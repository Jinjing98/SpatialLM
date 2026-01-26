# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

from typing import List, Optional, Tuple, Union
import math
import os

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# Global verbose flag for debug printouts
VERBOSE_3D_PE = os.environ.get("SPATIALLM_VERBOSE", "0").lower() in ("1", "true", "yes")

try:
    import torchsparse
    from torchsparse.utils.collate import sparse_collate
except ImportError:
    pass  # Ignore the import error if torchsparse is not installed for SpatialLM1.1

from spatiallm.model import PointBackboneType, ProjectorType

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)

# CCA utilities (only imported, no execution)
try:
    from spatiallm.model.cca_utils import (
        build_concentric_position_matrix,
        build_cca_position_ids,
        build_cca_attention_mask,
        project_pointcloud_to_2d_grid,
        CCA_GRID_SIZE,
    )
    _CCA_AVAILABLE = True
except ImportError:
    _CCA_AVAILABLE = False

# 3D Position Encoding utilities (RoPE and Sinusoidal)
try:
    from spatiallm.model.volumetric_pe import (
        RotaryEmbedding3D,
        apply_rotary_pos_emb_3d,
        compute_3d_sinusoidal_pe,
    )
    _3D_PE_AVAILABLE = True
    _3D_PE_ERROR = None
except ImportError as e:
    _3D_PE_AVAILABLE = False
    _3D_PE_ERROR = str(e)

# Custom Attention for 3D RoPE
try:
    from spatiallm.model.spatial_attention import SpatialLlamaAttention
    _SPATIAL_ATTENTION_AVAILABLE = True
    _SPATIAL_ATTENTION_ERROR = None
except ImportError as e:
    _SPATIAL_ATTENTION_AVAILABLE = False
    _SPATIAL_ATTENTION_ERROR = str(e)


class SpatialLMLlamaConfig(LlamaConfig):
    model_type = "spatiallm_llama"


class SpatialLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = SpatialLMLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.point_backbone_type = PointBackboneType(config.point_backbone)
        self.point_backbone = None
        point_config = config.point_config
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            from spatiallm.model.scenescript_encoder import PointCloudEncoder

            self.point_backbone = PointCloudEncoder(
                input_channels=point_config["input_channels"],
                d_model=point_config["embed_channels"],
                conv_layers=point_config["conv_layers"],
                num_bins=point_config["num_bins"],
            )
            embed_channels = point_config["embed_channels"]
        elif self.point_backbone_type == PointBackboneType.SONATA:
            from spatiallm.model.sonata_encoder import Sonata

            self.point_backbone = Sonata(
                in_channels=point_config["in_channels"],
                order=point_config["order"],
                stride=point_config["stride"],
                enc_depths=point_config["enc_depths"],
                enc_channels=point_config["enc_channels"],
                enc_num_head=point_config["enc_num_head"],
                enc_patch_size=point_config["enc_patch_size"],
                mlp_ratio=point_config["mlp_ratio"],
                mask_token=point_config["mask_token"],
                enc_mode=point_config["enc_mode"],
                enable_fourier_encode=True,
                num_bins=point_config["num_bins"],
                enable_flash=point_config.get("enable_flash", True),
                enable_rpe=point_config.get("enable_rpe", False),
            )
            embed_channels = point_config["enc_channels"][-1]
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

        self.projector_type = ProjectorType(getattr(config, "projector", "linear"))
        if self.projector_type == ProjectorType.LINEAR:
            self.point_proj = nn.Linear(embed_channels, config.hidden_size)
        elif self.projector_type == ProjectorType.MLP:
            self.point_proj = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.GELU(),
                nn.Linear(embed_channels, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {self.projector_type}")

        self.point_start_token_id = self.config.point_start_token_id
        self.point_end_token_id = self.config.point_end_token_id
        self.point_token_id = self.config.point_token_id

        # CCA (Concentric Causal Attention) for vlm_pe="CCA_2DProj"
        # Only initialize if vlm_pe is set to "CCA_2DProj"
        if _CCA_AVAILABLE and getattr(self.config, 'vlm_pe', None) == "CCA_2DProj":
            self._cca_concentric_pos = None  # Will be initialized on first use
            self._point_cca_positions = None  # CCA position for each point
        
        # 3D Position Encodings for vlm_pe
        # Initialize based on vlm_pe type with STRICT error checking
        vlm_pe = getattr(self.config, 'vlm_pe', None)
        
        # Handle string "None" as Python None (for command line compatibility)
        if vlm_pe == "None" or vlm_pe == "none":
            vlm_pe = None
        
        # === STRICT ERROR CHECKING: Raise errors if PE unavailable ===
        if vlm_pe == "CCA_2DProj" and not _CCA_AVAILABLE:
            raise ImportError(
                f"vlm_pe='CCA_2DProj' requested but cca_utils module failed to import. "
                f"Please ensure the module is available."
            )
        
        if vlm_pe in ["3D_RoPE", "3D_Sinusoidal"] and not _3D_PE_AVAILABLE:
            raise ImportError(
                f"vlm_pe='{vlm_pe}' requested but volumetric_pe module failed to import. "
                f"Error: {_3D_PE_ERROR}"
            )
        
        if vlm_pe == "3D_RoPE" and not _SPATIAL_ATTENTION_AVAILABLE:
            raise ImportError(
                f"vlm_pe='3D_RoPE' requested but spatial_attention module failed to import. "
                f"Error: {_SPATIAL_ATTENTION_ERROR}"
            )
        # ============================================================
        self.pcd_theta = getattr(config, 'pcd_theta', 10000)  # Base frequency for 3D PE (default: 10000)
        # 3D RoPE: vlm_pe="3D_RoPE"
        if vlm_pe == "3D_RoPE":
            # Read pcd_pe_merge_rule from config (passed from command line or default)
            pcd_pe_merge_rule = getattr(config, "pcd_pe_merge_rule", "3D_only")
            if pcd_pe_merge_rule not in ["3D_only", "3D_with_1D"]:
                raise ValueError(
                    f"[3D_RoPE] Invalid pcd_pe_merge_rule: '{pcd_pe_merge_rule}'. "
                    f"Must be '3D_only' or '3D_with_1D'."
                )
            
            # Initialize 3D RoPE module (applies per-head, all heads share same pattern)
            head_dim = config.hidden_size // config.num_attention_heads
            self.rotary_emb_3d = RotaryEmbedding3D(
                head_dim=head_dim,  # Apply per-head (e.g., 64)
                max_position_embeddings=config.max_position_embeddings,
                base=getattr(config, 'rope_theta', 10000),  # Use native RoPE base for 1D component
                base_3d=self.pcd_theta,  # Use pcd_theta for 3D components
                device=None,
                merge_rule=pcd_pe_merge_rule
            )
            self._point_patch_coords = None  # Will store patch coords for current batch (N, 3)
            self._point_3d_rope_cos = None  # Will store cos values (N, hidden_size)
            self._point_3d_rope_sin = None  # Will store sin values (N, hidden_size)
            self._point_token_positions = None  # Token positions of point cloud tokens in sequence
            
        else:
            self.rotary_emb_3d = None
        
        # 3D Sinusoidal PE: vlm_pe="3D_Sinusoidal"
        if vlm_pe == "3D_Sinusoidal":
            # Read pcd_pe_merge_rule from config (same as 3D_RoPE)
            pcd_pe_merge_rule = getattr(config, "pcd_pe_merge_rule", "3D_only")
            if pcd_pe_merge_rule not in ["3D_only", "3D_with_1D"]:
                raise ValueError(
                    f"[3D_Sinusoidal] Invalid pcd_pe_merge_rule: '{pcd_pe_merge_rule}'. "
                    f"Must be '3D_only' or '3D_with_1D'."
                )
            self.pcd_pe_merge_rule_sinusoidal = pcd_pe_merge_rule
            self._point_3d_sinusoidal_pe = None  # Will store PE for current batch (N, hidden_size)
            self._point_sinusoidal_positions = None  # Token positions for PE application
            logger.info(f"[3D_Sinusoidal] Initialized: pcd_pe_merge_rule={pcd_pe_merge_rule} pcd_theta={self.pcd_theta}")
        # Initialize weights and apply final processing
        self.post_init()
        
        # === Replace attention layers with Spatial variants for 3D_RoPE ===
        # This must happen AFTER post_init() which initializes the model
        if vlm_pe == "3D_RoPE":
            print("\n" + "="*80)
            print("[3D_RoPE] Replacing attention layers with SpatialLlamaAttention...")
            print(f"[3D_RoPE] pcd_pe_merge_rule = '{self.rotary_emb_3d.merge_rule}'")
            print("="*80)
            
            # Store merge_rule for forward pass
            self.pcd_pe_merge_rule = self.rotary_emb_3d.merge_rule
            
            num_replaced = 0
            for layer_idx, layer in enumerate(self.model.layers):
                # Replace the attention module
                old_attn = layer.self_attn
                old_type = type(old_attn).__name__
                new_attn = SpatialLlamaAttention(config, layer_idx)
                
                # Copy weights from old attention
                new_attn.load_state_dict(old_attn.state_dict())
                
                # Replace
                layer.self_attn = new_attn
                num_replaced += 1
                
                print(f"  Layer {layer_idx}: {old_type} → SpatialLlamaAttention ✓")
            
            print(f"[3D_RoPE] ✓ Replaced {num_replaced} attention layers")
            print("="*80 + "\n")
            
            # STRICT VERIFICATION: Check that replacement actually worked
            for layer_idx, layer in enumerate(self.model.layers):
                if not isinstance(layer.self_attn, SpatialLlamaAttention):
                    raise RuntimeError(
                        f"[3D_RoPE] FATAL ERROR: Layer {layer_idx} attention was NOT replaced! "
                        f"Expected SpatialLlamaAttention, got {type(layer.self_attn).__name__}. "
                        f"This means 3D_RoPE will NOT work!"
                    )
            print("[3D_RoPE] ✓ Verification passed: All attention layers are SpatialLlamaAttention\n")
        else:
            # vlm_pe is not "3D_RoPE" - use 100% original attention implementation
            print(f"[SpatialLM] vlm_pe={vlm_pe} - Using 100% original LlamaAttention (no custom modifications)\n")
            self.pcd_pe_merge_rule = None

    def forward_point_cloud(self, point_cloud, device, dtype):
        # point cloud has shape (n_points, n_features)
        # find the points that have nan values
        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        
        # === CCA: Compute 2D projection for point cloud tokens ===
        if _CCA_AVAILABLE and getattr(self.config, 'vlm_pe', None) == "CCA_2DProj":
            # feats[:, :3] contains normalized 3D coordinates in [0, 1]
            # TODO: Verify that feats[:, :3] is in [X, Y, Z] order
            grid_row, grid_col = project_pointcloud_to_2d_grid(
                feats[:, :3],
                grid_size=CCA_GRID_SIZE
            )
            
            # Build concentric position matrix (cached for efficiency)
            if self._cca_concentric_pos is None:
                self._cca_concentric_pos = build_concentric_position_matrix(
                    grid_size=CCA_GRID_SIZE,
                    device=device
                )
            elif self._cca_concentric_pos.device != device:
                # Move to correct device if needed
                self._cca_concentric_pos = self._cca_concentric_pos.to(device)
            
            # Lookup CCA position ID for each point from 2D grid
            self._point_cca_positions = self._cca_concentric_pos[grid_row, grid_col]
        # ========================================================
        
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            encoded_features = self.point_backbone(pc_sparse_tensor)
            # SceneScript doesn't provide patch coordinates, so we can't use 3D PE with it
            return self.point_proj(encoded_features["context"].to(dtype))
        
        elif self.point_backbone_type == PointBackboneType.SONATA:
            input_dict = {
                "coord": feats[:, :3].to(device),
                "grid_coord": coords.to(device),
                "feat": feats.to(device),
                "batch": torch.zeros(coords.shape[0], dtype=torch.long).to(device),
            }
            result = self.point_backbone(input_dict)
            
            # Sonata now returns a dict with "features" and "patch_coords"
            if isinstance(result, dict):
                encoded_features = result["features"]
                patch_coords = result["patch_coords"]  # (num_patches, 3) - actual patch centers!
                
                # Store patch coordinates for 3D PE computation
                # These are the ACTUAL semantic centers of encoded patches
                # No downsampling needed - perfect 1:1 match with encoded_features
                self._point_patch_coords = patch_coords
            else:
                # Backward compatibility: if Sonata returns tensor directly
                encoded_features = result
                self._point_patch_coords = None
            
            # add the batch dimension
            encoded_features = encoded_features.unsqueeze(0)
            return self.point_proj(encoded_features.to(dtype))
        
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        point_clouds: Optional[torch.Tensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            point_clouds (`torch.Tensor` of shape `(batch_size, n_points, n_features)`, *optional*):
                Point clouds to be used for the point cloud encoder.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("manycore-research/SpatialLM-Llama-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("manycore-research/SpatialLM-Llama-1B")

        >>> prompt = "<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"
        >>> conversation = [{"role": "user", "content": prompt}]
        >>> input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(input_ids, point_clouds=point_clouds, max_length=4096)
        >>> tokenizer.batch_decode(generate_ids, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # compute point cloud embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if (
            self.point_backbone is not None
            and (input_ids.shape[1] != 1 or self.training)
            and point_clouds is not None
        ):
            n_point_clouds = point_clouds.shape[0]
            point_features = []
            for i in range(n_point_clouds):  # * iterate over batch
                point_cloud = point_clouds[i]
                point_feature = self.forward_point_cloud(
                    point_cloud, inputs_embeds.device, inputs_embeds.dtype
                )
                point_features.append(point_feature)

            # Insert point cloud features into the input ids
            point_start_end_token_pos = []
            new_input_embeds = []
            new_attention_mask = []
            cur_point_idx = 0
            max_num_tokens = 0
            
            # === Compute 3D PE using actual patch coordinates ===
            # Use the patch center coordinates returned by Sonata encoder
            # These are the ACTUAL semantic centers of encoded patches (not downsampled!)
            if hasattr(self, '_point_patch_coords') and self._point_patch_coords is not None:
                patch_coords = self._point_patch_coords  # (num_patches, 3) - exact patch centers
                
                # Validation: Check for invalid values
                coords_valid = True
                if torch.isnan(patch_coords).any() or torch.isinf(patch_coords).any():
                    logger.warning("[3D PE] patch_coords contains NaN/Inf values! Skipping 3D PE.")
                    coords_valid = False
                
                if coords_valid:
                    # Always normalize coordinates using global scale (preserves aspect ratio)
                    coord_min_val = patch_coords.min(dim=0, keepdim=True)[0]  # (1, 3)
                    coord_max_val = patch_coords.max(dim=0, keepdim=True)[0]  # (1, 3)
                    coord_range = coord_max_val - coord_min_val                # (1, 3)
                    
                    # Use maximum range across all dimensions to preserve aspect ratio
                    max_range = coord_range.max()  # Scalar: largest dimension range
                    
                    logger.info(
                        f"[3D PE] Original patch_coords range: "
                        f"X=[{coord_min_val[0,0].item():.3f}, {coord_max_val[0,0].item():.3f}], "
                        f"Y=[{coord_min_val[0,1].item():.3f}, {coord_max_val[0,1].item():.3f}], "
                        f"Z=[{coord_min_val[0,2].item():.3f}, {coord_max_val[0,2].item():.3f}]"
                    )
                    
                    # Normalize with same scale for all dimensions (preserves spatial relationships)
                    patch_coords = (patch_coords - coord_min_val) / (max_range + 1e-8)
                    
                    logger.info(
                        f"[3D PE] Normalized with max_range={max_range.item():.3f}, "
                        f"final range: X=[{patch_coords[:,0].min().item():.3f}, {patch_coords[:,0].max().item():.3f}], "
                        f"Y=[{patch_coords[:,1].min().item():.3f}, {patch_coords[:,1].max().item():.3f}], "
                        f"Z=[{patch_coords[:,2].min().item():.3f}, {patch_coords[:,2].max().item():.3f}]"
                    )
                    
                    # Compute 3D RoPE if needed (returns (N, head_dim) for per-head application)
                    if self.rotary_emb_3d is not None:
                        print(f"\n[3D_RoPE] Computing 3D RoPE for {patch_coords.shape[0]} patches...")
                        print(f"[3D_RoPE] Merge rule: {self.rotary_emb_3d.merge_rule}")
                        
                        # Compute position_ids for point tokens (needed for "3D_with_1D" mode)
                        # We'll set these later when we know the actual sequence positions
                        # For now, just compute 3D component - position_ids will be added in forward pass
                        if self.rotary_emb_3d.merge_rule == "3D_with_1D":
                            # Store patch_coords for later use with position_ids
                            self._point_patch_coords_for_rope = patch_coords
                            # We need position_ids which we'll get from point_token_positions
                            # For now, use placeholder (will be recomputed in forward with real position_ids)
                            logger.warning(
                                "[3D_RoPE] TODO: merge_rule='3D_with_1D' currently uses sequence position for 1D component. "
                                "This should be reviewed to ensure correct positional encoding behavior."
                            )
                            # Create position_ids based on point token sequence positions
                            # We'll do this properly below after _point_token_positions is set
                            pass
                        
                        # For now, compute without position_ids (will recompute later for 3D_with_1D)
                        if self.rotary_emb_3d.merge_rule == "3D_only":
                            cos_3d, sin_3d = self.rotary_emb_3d(patch_coords, seq_len=patch_coords.shape[0], position_ids=None)
                            self._point_3d_rope_cos = cos_3d  # (num_patches, head_dim)
                            self._point_3d_rope_sin = sin_3d  # (num_patches, head_dim)
                            print(f"[3D_RoPE] ✓ Computed cos_3d.shape={cos_3d.shape}, sin_3d.shape={sin_3d.shape}")
                            
                            # STRICT CHECK: Verify dimensions
                            expected_head_dim = self.config.hidden_size // self.config.num_attention_heads
                            if cos_3d.shape[-1] != expected_head_dim:
                                raise RuntimeError(
                                    f"[3D_RoPE] FATAL ERROR: cos_3d has wrong dimension! "
                                    f"Expected head_dim={expected_head_dim}, got {cos_3d.shape[-1]}."
                                )
                        else:
                            # Will compute later with position_ids
                            self._point_3d_rope_cos = None
                            self._point_3d_rope_sin = None
                    
                    # Compute 3D Sinusoidal PE if needed
                    if hasattr(self, '_point_3d_sinusoidal_pe'):
                        print(f"\n[3D_Sinusoidal] Computing PE for {patch_coords.shape[0]} patches...")
                        print(f"[3D_Sinusoidal] Merge rule: {self.pcd_pe_merge_rule_sinusoidal}")
                        
                        if self.pcd_pe_merge_rule_sinusoidal == "3D_with_1D":
                            # Store patch_coords for later use with position_ids
                            self._point_patch_coords_for_sinusoidal = patch_coords
                            # Will compute later in forward() when we have position_ids
                            logger.warning(
                                "[3D_Sinusoidal] TODO: merge_rule='3D_with_1D' currently uses sequence position for 1D component. "
                                "This should be reviewed to ensure correct positional encoding behavior."
                            )
                            self._point_3d_sinusoidal_pe = None  # Will be computed later
                        else:  # "3D_only"
                            self._point_3d_sinusoidal_pe = compute_3d_sinusoidal_pe(
                                coords_3d=patch_coords,
                                hidden_size=self.config.hidden_size,
                                num_heads=self.config.num_attention_heads,
                                base=getattr(self.config, 'rope_theta', 10000),
                                base_3d=self.pcd_theta,
                                device=inputs_embeds.device,
                                merge_rule=self.pcd_pe_merge_rule_sinusoidal,
                                position_ids=None
                            )  # (num_patches, hidden_size)
                            print(f"[3D_Sinusoidal] ✓ Computed PE shape={self._point_3d_sinusoidal_pe.shape}")
            # =====================================================
            
            for cur_input_ids, cur_input_embeds, cur_attention_mask in zip(
                input_ids, inputs_embeds, attention_mask
            ):  # * input_ids: B, L; input_embeds: B, L, C
                cur_point_features = (
                    point_features[cur_point_idx]
                    .to(device=cur_input_embeds.device)
                    .squeeze(0)
                )
                num_patches = cur_point_features.shape[0]  # * number of point tokens
                num_point_start_tokens = (
                    (cur_input_ids == self.config.point_start_token_id).sum().item()
                )
                num_point_end_tokens = (
                    (cur_input_ids == self.config.point_end_token_id).sum().item()
                )
                # currently, we only support one point start and one point end token
                assert num_point_start_tokens == num_point_end_tokens == 1, (
                    "The number of point start tokens and point end tokens should be 1, "
                    f"but got {num_point_start_tokens} and {num_point_end_tokens}."
                )
                point_start_token_pos = torch.where(
                    cur_input_ids == self.config.point_start_token_id
                )[0][0]
                point_end_token_pos = torch.where(
                    cur_input_ids == self.config.point_end_token_id
                )[0][0]
                cur_new_input_embeds = torch.cat(
                    (
                        cur_input_embeds[: point_start_token_pos + 1],
                        cur_point_features,
                        cur_input_embeds[point_end_token_pos:],
                    ),
                    dim=0,
                )
                cur_new_attention_mask = torch.cat(
                    (
                        cur_attention_mask[: point_start_token_pos + 1],
                        torch.ones(num_patches, device=cur_attention_mask.device),
                        cur_attention_mask[point_end_token_pos:],
                    ),
                    dim=0,
                )

                cur_point_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                point_start_end_token_pos.append(
                    (point_start_token_pos, num_patches, point_end_token_pos)
                )
                if cur_new_input_embeds.shape[0] > max_num_tokens:
                    max_num_tokens = cur_new_input_embeds.shape[0]
            # pad the new input embeds and attention mask to the max dimension
            for i in range(len(new_input_embeds)):
                cur_input_embeds = new_input_embeds[i]
                last_row = cur_input_embeds[-1]
                padding = last_row.repeat(max_num_tokens - cur_input_embeds.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_input_embeds, padding], dim=0)

                cur_attention_mask = new_attention_mask[i]
                new_attention_mask[i] = F.pad(
                    cur_attention_mask,
                    (0, max_num_tokens - cur_attention_mask.shape[0]),
                    value=0,
                )
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            assert (
                attention_mask.shape[1] == inputs_embeds.shape[1]
            ), "The length of attention mask and inputs embeds should be the same"
            
            # === 3D RoPE: Track point token positions for attention patching ===
            if self.rotary_emb_3d is not None:
                # Store the positions of point cloud tokens in the sequence
                # This will be used by the attention layer to apply 3D RoPE
                # Format: list of (start_idx, end_idx) for each batch sample
                self._point_token_positions = []
                for i, (point_start, num_patches, point_end) in enumerate(point_start_end_token_pos):
                    # Point tokens are at [point_start+1, point_start+1+num_patches]
                    pt_start_idx = point_start + 1  # +1 to skip <point_start> token
                    pt_end_idx = pt_start_idx + num_patches
                    self._point_token_positions.append((pt_start_idx, pt_end_idx))
                
                # For "3D_with_1D" mode, now compute 3D RoPE with position_ids
                if self.rotary_emb_3d.merge_rule == "3D_with_1D" and hasattr(self, '_point_patch_coords_for_rope'):
                    # Compute position_ids from sequence positions
                    # Use the first batch's point token positions (assuming single batch for now)
                    if len(self._point_token_positions) > 0:
                        pt_start, pt_end = self._point_token_positions[0]
                        # Create position_ids: [pt_start, pt_start+1, ..., pt_end-1]
                        position_ids_point = torch.arange(pt_start, pt_end, dtype=torch.long, device=inputs_embeds.device)
                        
                        print(f"[3D_RoPE] Computing with position_ids for 3D_with_1D mode...")
                        print(f"[3D_RoPE] Position IDs range: [{pt_start}, {pt_end}) -> {position_ids_point.shape}")
                        
                        # Now compute 3D RoPE with position_ids
                        cos_3d, sin_3d = self.rotary_emb_3d(
                            self._point_patch_coords_for_rope,
                            seq_len=self._point_patch_coords_for_rope.shape[0],
                            position_ids=position_ids_point
                        )
                        self._point_3d_rope_cos = cos_3d  # (num_patches, head_dim)
                        self._point_3d_rope_sin = sin_3d  # (num_patches, head_dim)
                        print(f"[3D_RoPE] ✓ Computed cos_3d.shape={cos_3d.shape}, sin_3d.shape={sin_3d.shape}")
                        
                        # STRICT CHECK: Verify dimensions
                        expected_head_dim = self.config.hidden_size // self.config.num_attention_heads
                        if cos_3d.shape[-1] != expected_head_dim:
                            raise RuntimeError(
                                f"[3D_RoPE] FATAL ERROR: cos_3d has wrong dimension! "
                                f"Expected head_dim={expected_head_dim}, got {cos_3d.shape[-1]}."
                            )
                
                # Also compute 3D_Sinusoidal PE if needed for "3D_with_1D" mode
                if (hasattr(self, 'pcd_pe_merge_rule_sinusoidal') and 
                    self.pcd_pe_merge_rule_sinusoidal == "3D_with_1D" and
                    hasattr(self, '_point_patch_coords_for_sinusoidal')):
                    # Reuse the same position_ids we computed for 3D_RoPE
                    print(f"[3D_Sinusoidal] Computing with position_ids for 3D_with_1D mode...")
                    print(f"[3D_Sinusoidal] Position IDs range: [{pt_start}, {pt_end}) -> {position_ids_point.shape}")
                    
                    self._point_3d_sinusoidal_pe = compute_3d_sinusoidal_pe(
                        coords_3d=self._point_patch_coords_for_sinusoidal,
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_attention_heads,
                        base=getattr(self.config, 'rope_theta', 10000),
                        base_3d=self.pcd_theta,
                        device=inputs_embeds.device,
                        merge_rule=self.pcd_pe_merge_rule_sinusoidal,
                        position_ids=position_ids_point
                    )
                    print(f"[3D_Sinusoidal] ✓ Computed PE shape={self._point_3d_sinusoidal_pe.shape}")
            # ===================================================================
            
            # === 3D Sinusoidal PE: Apply to point cloud embeddings ===
            # Add 3D Sinusoidal PE directly to point cloud token embeddings
            # This uses the same per-head dimension split as 3D RoPE for consistency
            # IMPORTANT: Only apply during prefill (past_key_values is None), not during decoding
            if (hasattr(self, '_point_3d_sinusoidal_pe') and 
                self._point_3d_sinusoidal_pe is not None and
                past_key_values is None):  # Only during prefill!
                # Apply PE to each point cloud in the batch
                for i, (point_start, num_patches, point_end) in enumerate(point_start_end_token_pos):
                    # Point tokens are at [point_start+1, point_start+1+num_patches]
                    pt_start_idx = point_start + 1  # +1 to skip <point_start> token
                    pt_end_idx = pt_start_idx + num_patches
                    
                    # Add 3D Sinusoidal PE to point cloud embeddings
                    inputs_embeds[i, pt_start_idx:pt_end_idx, :] += self._point_3d_sinusoidal_pe
            # ==========================================================

        # === CCA: Build CCA position IDs and attention mask ===
        # Apply CCA (Concentric Causal Attention) if vlm_pe="CCA_2DProj"
        # This extends the 2D image CCA from modelling_fca_Feng.py to 3D point clouds
        # by projecting point cloud tokens onto a 2D plane (top-down view)
        # IMPORTANT: Only apply during prefill (past_key_values is None), not during decoding
        if (_CCA_AVAILABLE and 
            getattr(self.config, 'vlm_pe', None) == "CCA_2DProj" and 
            point_clouds is not None and 
            past_key_values is None):
            num_point_tokens = self._point_cca_positions.shape[0]
            batch_point_token_pos = []
            batch_cca_position_ids = []
            
            # Build CCA position IDs for each sample in batch
            for cur_input_ids in input_ids:
                point_start_pos = torch.where(
                    cur_input_ids == self.config.point_start_token_id
                )[0]
                
                if len(point_start_pos) > 0:
                    # This sample has point cloud
                    point_start_pos = point_start_pos[0].item()
                    batch_point_token_pos.append(point_start_pos)
                    
                    # Build CCA position IDs for this sample
                    cca_pos_ids = build_cca_position_ids(
                        position_ids=position_ids[0] if position_ids is not None else None,
                        point_token_pos=point_start_pos,
                        num_point_tokens=num_point_tokens,
                        concentric_pos=self._cca_concentric_pos,
                        device=inputs_embeds.device,
                        seq_len=inputs_embeds.shape[1],  # After embedding fusion
                        past_key_values=past_key_values
                    )
                    batch_cca_position_ids.append(cca_pos_ids.unsqueeze(0))
                else:
                    # Text-only sample (no point cloud)
                    batch_point_token_pos.append(-1)
                    batch_cca_position_ids.append(
                        torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
                    )
            
            # Stack batch
            batch_cca_position_ids = torch.cat(batch_cca_position_ids, dim=0)
            
            # Build CCA attention mask
            cca_attention_mask = build_cca_attention_mask(
                attention_mask=attention_mask,
                batch_point_token_pos=batch_point_token_pos,
                batch_cca_position_ids=batch_cca_position_ids,
                num_point_tokens=num_point_tokens,
                batch_size=inputs_embeds.shape[0],
                seq_len=inputs_embeds.shape[1],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                grid_size=CCA_GRID_SIZE
            )
            
            # Use CCA position IDs and mask
            position_ids = batch_cca_position_ids
            attention_mask = cca_attention_mask
        # =====================================================
        
        # === 3D RoPE: Set data on custom attention layers ===
        # Instead of monkey-patching, we use custom SpatialLlamaAttention layers
        # that check for _spatial_3d_rope_data attribute
        
        # Detect if this is prefill stage (cache is empty or None)
        is_prefill = (past_key_values is None)
        if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
            is_prefill = (past_key_values.get_seq_length(0) == 0)
        
        if (self.rotary_emb_3d is not None and 
            hasattr(self, '_point_3d_rope_cos') and 
            self._point_3d_rope_cos is not None and
            is_prefill):  # Only during prefill
            
            print(f"\n[3D_RoPE] Setting 3D RoPE data on attention layers...")
            print(f"  cos_3d: {self._point_3d_rope_cos.shape}")
            print(f"  sin_3d: {self._point_3d_rope_sin.shape}")
            print(f"  point_token_positions: {self._point_token_positions}")
            
            # Package data for attention layers
            spatial_rope_data = (
                self._point_3d_rope_cos,  # (num_patches, head_dim)
                self._point_3d_rope_sin,  # (num_patches, head_dim)
                self._point_token_positions  # List of (start, end) tuples
            )
            
            # Set data on all attention layers
            num_set = 0
            for layer in self.model.layers:
                layer.self_attn._spatial_3d_rope_data = spatial_rope_data
                num_set += 1
            
            print(f"[3D_RoPE] ✓ Set 3D RoPE data on {num_set} attention layers")
            print(f"[3D_RoPE] → Data contents: cos/sin shapes {cos_3d.shape}, {len(self._point_token_positions)} batches")
            print(f"[3D_RoPE] → Example: layer 0 has data? {hasattr(self.model.layers[0].self_attn, '_spatial_3d_rope_data')}")
            if hasattr(self.model.layers[0].self_attn, '_spatial_3d_rope_data'):
                print(f"[3D_RoPE] → Layer 0 data is None? {self.model.layers[0].self_attn._spatial_3d_rope_data is None}")
            
            # STRICT VERIFICATION: Check first layer
            first_attn = self.model.layers[0].self_attn
            if not hasattr(first_attn, '_spatial_3d_rope_data'):
                raise RuntimeError(
                    f"[3D_RoPE] FATAL ERROR: Failed to set _spatial_3d_rope_data on attention layers! "
                    f"Attention type: {type(first_attn).__name__}"
                )
            if first_attn._spatial_3d_rope_data is None:
                raise RuntimeError(
                    f"[3D_RoPE] FATAL ERROR: _spatial_3d_rope_data is None on attention layers!"
                )
            print(f"[3D_RoPE] ✓ Verification: Data successfully set on attention layers")
            print(f"[3D_RoPE] → Layer 0 attention ID: {id(self.model.layers[0].self_attn)}\n")
        elif self.rotary_emb_3d is not None and is_prefill:
            # We have 3D_RoPE enabled but no data during prefill - this is an error!
            raise RuntimeError(
                f"[3D_RoPE] FATAL ERROR: 3D_RoPE is enabled but no RoPE data available during prefill! "
                f"_point_3d_rope_cos exists: {hasattr(self, '_point_3d_rope_cos')}, "
                f"is not None: {getattr(self, '_point_3d_rope_cos', None) is not None}"
            )
        # ====================================================

        # === CRITICAL DEBUG: Check data RIGHT BEFORE model call ===
        if VERBOSE_3D_PE and self.rotary_emb_3d is not None and hasattr(self, '_point_3d_rope_cos') and self._point_3d_rope_cos is not None:
            print(f"\n[CRITICAL CHECK] RIGHT BEFORE self.model() call:")
            print(f"  Layer 0 attention type: {type(self.model.layers[0].self_attn).__name__}")
            print(f"  Layer 0 attention ID: {id(self.model.layers[0].self_attn)}")
            print(f"  Layer 0 has _spatial_3d_rope_data? {hasattr(self.model.layers[0].self_attn, '_spatial_3d_rope_data')}")
            if hasattr(self.model.layers[0].self_attn, '_spatial_3d_rope_data'):
                print(f"  Layer 0 data is None? {self.model.layers[0].self_attn._spatial_3d_rope_data is None}")
            print(f"  past_key_values type: {type(past_key_values)}")
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                print(f"  Cache length: {past_key_values.get_seq_length(0)}")
            print()
        # ==========================================================

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        # === 3D RoPE: Clear data from attention layers (explicit cleanup) ===
        # After prefill, we explicitly clear _spatial_3d_rope_data to make it clear that
        # 3D RoPE is ONLY applied during prefill. During decoding, point tokens already
        # have 3D RoPE cached in KV cache, and new text tokens use standard 1D RoPE.
        
        # Clear if we set the data earlier in this forward pass
        if (self.rotary_emb_3d is not None and 
            hasattr(self, '_point_3d_rope_cos') and 
            self._point_3d_rope_cos is not None and
            hasattr(self.model.layers[0].self_attn, '_spatial_3d_rope_data') and
            self.model.layers[0].self_attn._spatial_3d_rope_data is not None):
            # Clear the data after model call
            num_cleared = 0
            for layer in self.model.layers:
                if hasattr(layer.self_attn, '_spatial_3d_rope_data'):
                    layer.self_attn._spatial_3d_rope_data = None
                    num_cleared += 1
            print(f"[3D_RoPE] ✓ Cleared _spatial_3d_rope_data from {num_cleared} attention layers")
            print(f"[3D_RoPE] → Subsequent decoding will use cached point tokens (3D RoPE) + new text tokens (1D RoPE)\n")
        # =====================================================================

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # prepare new labels
            new_labels = []
            max_num_tokens = logits.shape[1]
            for i in range(len(point_start_end_token_pos)):
                cur_labels = labels[i]
                (
                    cur_point_start_token_pos,
                    num_patches,
                    cur_point_end_token_pos,
                ) = point_start_end_token_pos[i]
                cur_new_labels = torch.cat(
                    (
                        cur_labels[: cur_point_start_token_pos + 1],
                        torch.full(
                            (num_patches,),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                        ),
                        cur_labels[cur_point_end_token_pos:],
                    ),
                    dim=0,
                )
                cur_new_labels = F.pad(
                    cur_new_labels,
                    (0, max_num_tokens - cur_new_labels.shape[0]),
                    value=IGNORE_INDEX,
                )
                new_labels.append(cur_new_labels)
            labels = torch.stack(new_labels, dim=0)

            assert (
                labels.shape[1] == logits.shape[1]
            ), "The length of labels and logits should be the same"

            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs


AutoConfig.register("spatiallm_llama", SpatialLMLlamaConfig)
AutoModelForCausalLM.register(SpatialLMLlamaConfig, SpatialLMLlamaForCausalLM)
