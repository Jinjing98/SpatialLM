# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    Qwen2Model,
    Qwen2ForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

try:
    import torchsparse
    from torchsparse.utils.collate import sparse_collate
except ImportError:
    pass  # Ignore the import error if torchsparse is not installed for SpatialLM1.1

from spatiallm.model import PointBackboneType, ProjectorType

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)

# CCA utilities - STRICT IMPORT (will fail if not available)
from spatiallm.model.cca_utils import (
    build_concentric_position_matrix,
    build_cca_position_ids,
    build_cca_attention_mask,
    project_pointcloud_to_2d_grid,
    CCA_GRID_SIZE,
)


class CCASpatialLMQwenConfig(Qwen2Config):
    model_type = "cca_spatiallm_qwen"


class CCASpatialLMQwenForCausalLM(Qwen2ForCausalLM):
    config_class = CCASpatialLMQwenConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
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

        # CCA (Concentric Causal Attention) - STRICT MODE
        # Verify that VLM_PE is set to "CCA_2DProj"
        vlm_pe = getattr(self.config, 'VLM_PE', None)
        if vlm_pe != "CCA_2DProj":
            raise ValueError(
                f"This model requires VLM_PE='CCA_2DProj' for strict CCA mode, "
                f"but got VLM_PE='{vlm_pe}'. Please set config.VLM_PE='CCA_2DProj'."
            )
        
        # Initialize CCA structures
        self._cca_concentric_pos = None  # Will be initialized on first use
        self._point_cca_positions = None  # CCA position for each point

        # Initialize weights and apply final processing
        self.post_init()

    def _compute_cca_2d_projection(self, point_coords_3d, device):
        """
        Compute CCA 2D projection for point cloud tokens.
        
        Args:
            point_coords_3d: Normalized 3D coordinates [N, 3] in [0, 1] range, assumes [X, Y, Z] order
            device: Target device
            
        Returns:
            None. Updates self._point_cca_positions with CCA position IDs for each point.
        """
        print(f'[CCA] Computing 2D projection for {point_coords_3d.shape[0]} points')
        # Project 3D points to 2D grid
        grid_row, grid_col = project_pointcloud_to_2d_grid(
            point_coords_3d,
            grid_size=CCA_GRID_SIZE
        )
        
        # Build concentric position matrix (cached for efficiency)
        if self._cca_concentric_pos is None:
            self._cca_concentric_pos = build_concentric_position_matrix(
                grid_size=CCA_GRID_SIZE,
                device=device
            )
            print(f'[CCA] Built concentric position matrix with grid_size={CCA_GRID_SIZE}')
        elif self._cca_concentric_pos.device != device:
            # Move to correct device if needed
            self._cca_concentric_pos = self._cca_concentric_pos.to(device)
        
        # Lookup CCA position ID for each point from 2D grid
        self._point_cca_positions = self._cca_concentric_pos[grid_row, grid_col]
        print(f'[CCA] Point CCA positions shape: {self._point_cca_positions.shape}')

    def _build_cca_position_ids_and_mask(self, input_ids, inputs_embeds, attention_mask, position_ids, past_key_values):
        """
        Build CCA position IDs and attention mask for point cloud tokens.
        
        Args:
            input_ids: Input token IDs [B, L]
            inputs_embeds: Input embeddings after point cloud fusion [B, L', D]
            attention_mask: Attention mask [B, L']
            position_ids: Position IDs (may be None)
            past_key_values: Past key values for generation (None during prefill)
            
        Returns:
            Tuple of (cca_position_ids, cca_attention_mask)
        """
        if self._point_cca_positions is None:
            raise RuntimeError(
                "CCA positions not computed. Ensure forward_point_cloud was called "
                "and point cloud features were properly encoded."
            )
        
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
        
        return batch_cca_position_ids, cca_attention_mask

    def forward_point_cloud(self, point_cloud, device, dtype):
        # point cloud has shape (n_points, n_features)
        # find the points that have nan values
        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        
        # Compute CCA 2D projection for point cloud tokens
        # feats[:, :3] contains normalized 3D coordinates in [0, 1]
        self._compute_cca_2d_projection(feats[:, :3], device)
        
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            encoded_features = self.point_backbone(pc_sparse_tensor)
            return self.point_proj(encoded_features["context"].to(dtype))
        elif self.point_backbone_type == PointBackboneType.SONATA:
            input_dict = {
                "coord": feats[:, :3].to(device),
                "grid_coord": coords.to(device),
                "feat": feats.to(device),
                "batch": torch.zeros(coords.shape[0], dtype=torch.long).to(device),
            }
            encoded_features = self.point_backbone(input_dict)
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

        >>> model = AutoModelForCausalLM.from_pretrained("manycore-research/SpatialLM-Qwen-0.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("manycore-research/SpatialLM-Qwen-0.5B")

        >>> prompt = "<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"
        >>> conversation = [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}]
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
            print(f'[CCA] This is the prefill stage with input_ids shape: {input_ids.shape}')
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
            for cur_input_ids, cur_input_embeds, cur_attention_mask in zip(
                input_ids, inputs_embeds, attention_mask
            ):  # * input_ids: B, L; input_embeds: B, L, C
                cur_point_features = (
                    point_features[cur_point_idx]
                    .to(device=cur_input_embeds.device)
                    .squeeze(0)
                )
                num_patches = cur_point_features.shape[0]  # * number of point tokens
                print('[CCA] point_features num_patches: ', num_patches)
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
                print('[CCA] cur_input_ids: ',cur_input_ids.shape)
                point_start_token_pos = torch.where(
                    cur_input_ids == self.config.point_start_token_id
                )[0][0]
                point_end_token_pos = torch.where(
                    cur_input_ids == self.config.point_end_token_id
                )[0][0]
                print('[CCA] point_start_token_id/point_end_token_id: ', self.config.point_start_token_id, self.config.point_end_token_id)
                print('[CCA] point_start_token_pos: ', point_start_token_pos, 'point_end_token_pos: ', point_end_token_pos)
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
                print('[CCA] cur_new_input_embeds: ', cur_new_input_embeds.shape)
                print('[CCA] cur_new_attention_mask: ', cur_new_attention_mask.shape)

                cur_point_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                point_start_end_token_pos.append(
                    (point_start_token_pos, num_patches, point_end_token_pos)
                )
                if cur_new_input_embeds.shape[0] > max_num_tokens:
                    max_num_tokens = cur_new_input_embeds.shape[0]
            print(f'[CCA] max_num_tokens in the batch with b_size{n_point_clouds}: ', max_num_tokens)
            
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

        # Build CCA position IDs and attention mask
        # Apply during prefill (past_key_values is None), not during decoding
        if point_clouds is not None and past_key_values is None:
            print('[CCA] Building CCA position_ids and attention_mask')
            position_ids, attention_mask = self._build_cca_position_ids_and_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values
            )
            print(f'[CCA] CCA position_ids shape: {position_ids.shape}')
            print(f'[CCA] CCA attention_mask shape: {attention_mask.shape}')

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

        hidden_states = outputs[0]
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


AutoConfig.register("cca_spatiallm_qwen", CCASpatialLMQwenConfig)
AutoModelForCausalLM.register(CCASpatialLMQwenConfig, CCASpatialLMQwenForCausalLM)
