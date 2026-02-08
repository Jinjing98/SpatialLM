# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    # JJ: We will create custom versions with MixedRoPE3D
    # Qwen2Model,
    # Qwen2ForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

# JJ: Import our custom MixedRoPE3D versions
from spatiallm.model.rope3d.rope3dQwen2ForCausallm import (
    Qwen2ModelMixedRoPE3D,
    Qwen2ForCausalLMMixedRoPE3D,
)

try:
    import torchsparse
    from torchsparse.utils.collate import sparse_collate
except ImportError:
    pass  # Ignore the import error if torchsparse is not installed for SpatialLM1.1

from spatiallm.model import PointBackboneType, ProjectorType

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)


class MixedRoPE3DSpatialLMQwenConfig(Qwen2Config):
    model_type = "mixedRoPE3d_spatiallm_qwen"


class MixedRoPE3DSpatialLMQwenForCausalLM(Qwen2ForCausalLMMixedRoPE3D):
    config_class = MixedRoPE3DSpatialLMQwenConfig

    def __init__(self, config):
        super().__init__(config)
        # JJ: Use our custom Qwen2Model with MixedRoPE3D
        self.model = Qwen2ModelMixedRoPE3D(config)
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

        # Initialize weights and apply final processing
        self.post_init()

    def forward_point_cloud(self, point_cloud, device, dtype):
        # JJ: Return both embeddings and grid coordinates for MixedRoPE3D
        # Grid coordinates: each token corresponds to a voxel grid cell
        # point cloud has shape (n_points, n_features)
        
        # Find and remove NaN values
        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            encoded_output = self.point_backbone(pc_sparse_tensor)
            
            # JJ: SceneScript returns dict with "context" (embeddings) and "coords" (grid coordinates)
            # The forward() method calls vox_to_sequence() which returns:
            # - "seq": [B, maxlen, C] features
            # - "coords": [B, maxlen, 3] grid coordinates
            # - "mask": [B, maxlen] mask
            # But the return is processed as {"context": context, "context_mask": mask}
            
            # Extract context (embeddings)
            context = encoded_output["context"]  # [B, N_tokens, C]
            point_embeds = self.point_proj(context.to(dtype))
            
            # JJ: Extract grid coordinates from the sparse tensor BEFORE vox_to_sequence
            # We need to process the raw sparse output to get coords
            # SceneScript uses ResNet3DSparse which outputs torchsparse.SparseTensor
            # Re-run the encoder to get the sparse tensor output
            sparse_output = self.point_backbone.sparse_resnet(pc_sparse_tensor)
            
            # Extract coordinates from sparse tensor: .C is [N, 4] with [batch_idx, x, y, z]
            from spatiallm.model.scenescript_encoder import sparse_uncollate
            sparse_list = sparse_uncollate(sparse_output)
            grid_coords = sparse_list[0].C.float()  # [N_tokens, 3] - already dropped batch idx in sparse_uncollate
            
            
            return point_embeds, grid_coords.to(device)
            
        elif self.point_backbone_type == PointBackboneType.SONATA:
            input_dict = {
                "coord": feats[:, :3].to(device),
                "grid_coord": coords.to(device),
                "feat": feats.to(device),
                "batch": torch.zeros(coords.shape[0], dtype=torch.long).to(device),
            }
            encoded_features = self.point_backbone(input_dict)
            # Sonata returns context tensor [N_tokens, C]
            
            # JJ: To get grid_coord, we need to access the internal Point structure
            # Sonata forward returns context directly, but we need coords
            # Let's use return_coords=True if available
            if hasattr(self.point_backbone, 'forward') and 'return_coords' in self.point_backbone.forward.__code__.co_varnames:
                encoded_features, grid_coords_normalized = self.point_backbone(input_dict, return_coords=True)
                # grid_coords_normalized is normalized [0, 1], we need to denormalize
                # Use reduced_grid_size from encoder
                grid_coords = grid_coords_normalized * (self.point_backbone.reduced_grid_size - 1)
            else:
                # Fallback: re-run to get the point structure
                # This is less efficient but works
                from spatiallm.model.sonata_encoder import Point
                point = Point(input_dict)
                point = self.point_backbone.embedding(point)
                point.serialization(order=self.point_backbone.order, shuffle_orders=self.point_backbone.shuffle_orders)
                point.sparsify()
                point = self.point_backbone.enc(point)
                
                # Extract grid coordinates
                grid_coords = point["grid_coord"].float()  # [N_tokens, 3]
                
                # Use the already computed features
                encoded_features = self.point_backbone(input_dict)
            
            # Add the batch dimension
            encoded_features = encoded_features.unsqueeze(0)
            point_embeds = self.point_proj(encoded_features.to(dtype))
            
            return point_embeds, grid_coords.to(device)
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        return self.model

    # JJ HACK: explicitly 1D RoPE: generate position_ids based on attention_mask and past_key_values
    # ///////////////////////////////////////
    def _debug_position_ids_gen(self, attention_mask, past_key_values, inputs_embeds_shape, device):
        # Debug information
        # Generate position_ids based on inputs_embeds length, not attention_mask
        # During generation: inputs_embeds is [batch, 1, hidden] but attention_mask is [batch, cache_len+1]
        # batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        batch_size, seq_len = inputs_embeds_shape
        # Check if we have valid cached key-values
        has_cache = False
        cache_length = 0
        if past_key_values is not None:
            try:
                if hasattr(past_key_values, 'get_seq_length'):
                    cache_length = past_key_values.get_seq_length()
                    has_cache = cache_length > 0
                elif len(past_key_values) > 0:
                    cache_length = past_key_values[0][0].shape[2]
                    has_cache = True
            except (IndexError, KeyError, AttributeError):
                has_cache = False
                cache_length = 0
        
        if not has_cache:
            # First forward pass: position_ids based on cumsum of attention_mask
            position_ids_explicit_dbg = (attention_mask.long().cumsum(-1) - 1).masked_fill_(attention_mask == 0, 0)
            # But only keep the positions for the current inputs_embeds
            position_ids_explicit_dbg = position_ids_explicit_dbg[:, :seq_len]
        else:
            # Subsequent generation: position_ids start from cache_length
            position_ids_explicit_dbg = torch.arange(
                cache_length, cache_length + seq_len,
                dtype=torch.long,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
        print(f'[DftPE] position_ids_explicit_dbg: {position_ids_explicit_dbg.shape}')
        # print(f'[DftPE] position_ids_explicit_dbg: {position_ids_explicit_dbg[:10]}')
        # ///////////////////////////////////////
        return position_ids_explicit_dbg

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

        # JJ: Track point coordinates and masks for MixedRoPE3D
        point_coords_list = []
        point_token_mask = None
        
        # JJ: Optimization - only process point_clouds during prefill stage
        # Generation stage (input_ids.shape[1] == 1) should not process raw point clouds
        is_generation_stage = (input_ids is not None and input_ids.shape[1] == 1 and not self.training)
        
        if (
            self.point_backbone is not None
            and not is_generation_stage  # Skip point cloud processing in generation
            and point_clouds is not None
        ):
            n_point_clouds = point_clouds.shape[0]
            point_features = []
            for i in range(n_point_clouds):  # * iterate over batch
                point_cloud = point_clouds[i]
                point_feature, point_coords_raw = self.forward_point_cloud(
                    point_cloud, inputs_embeds.device, inputs_embeds.dtype
                )
                point_features.append(point_feature)
                point_coords_list.append(point_coords_raw)

            # Insert point cloud features into the input ids
            point_start_end_token_pos = []
            new_input_embeds = []
            new_attention_mask = []
            new_point_token_masks = []  # JJ: Track which tokens are point tokens
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
                
                # JJ: Create point token mask
                cur_point_token_mask = torch.zeros(cur_new_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device)
                cur_point_token_mask[point_start_token_pos + 1 : point_start_token_pos + 1 + num_patches] = True

                cur_point_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                new_point_token_masks.append(cur_point_token_mask)
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
                
                # JJ: Pad point token mask
                cur_point_token_mask = new_point_token_masks[i]
                new_point_token_masks[i] = F.pad(
                    cur_point_token_mask,
                    (0, max_num_tokens - cur_point_token_mask.shape[0]),
                    value=False,
                )
            
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            point_token_mask = torch.stack(new_point_token_masks, dim=0)  # [B, max_seq_len]
            
            # JJ: Use the first batch's point coordinates (assuming same scene structure)
            if len(point_coords_list) > 0:
                point_coords_list = point_coords_list[0]  # [N_point, 3]
            else:
                point_coords_list = None

            assert (
                attention_mask.shape[1] == inputs_embeds.shape[1]
            ), "The length of attention mask and inputs embeds should be the same"
        elif is_generation_stage:
            # Generation stage: skip point cloud processing
            pass

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        assert position_ids is None, f'position_ids is None for dft. we make it explicit in _debug_position_ids_gen'
        
        # JJ: Determine if this is prefill or generation stage
        # Prefill: input_ids contains full prompt (shape [B, L] with L > 1)
        # Generation: input_ids contains only 1 new token (shape [B, 1])
        # Better heuristic than checking past_key_values, as transformers may initialize cache before first forward
        is_prefill_stage = (input_ids is not None and input_ids.shape[1] > 1) or (past_key_values is None)
        
        # JJ: Pass point_coords and point_token_mask only during prefill
        # During generation, we don't apply 3D RoPE to new text tokens
        if is_prefill_stage:
            # Prefill stage: apply 3D RoPE to point tokens
            model_point_coords = point_coords_list if 'point_coords_list' in locals() else None
            model_point_token_mask = point_token_mask if 'point_token_mask' in locals() else None
        else:
            # Generation stage: only standard 1D RoPE for new text tokens
            model_point_coords = None
            model_point_token_mask = None
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,# only have dim of text_token_dim+accu_next_text_token_dim 1 during net token gen.. The attn for pcd does not saved here during net token gen.
            position_ids=position_ids,# JJ. by default None. It leverge the attn_mask in PREFILL to first constuct, then keep append during next token gen.
            # position_ids=self._debug_position_ids_gen(attention_mask, past_key_values, inputs_embeds.shape[:2], inputs_embeds.device),# JJ HACK: explicitly generate position_ids based on the same strategy. should be the same
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=None,  # Set to None to let position_ids take full control
            point_coords=model_point_coords,
            point_token_mask=model_point_token_mask,
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


AutoConfig.register("mixedRoPE3d_spatiallm_qwen", MixedRoPE3DSpatialLMQwenConfig)
AutoModelForCausalLM.register(MixedRoPE3DSpatialLMQwenConfig, MixedRoPE3DSpatialLMQwenForCausalLM)


if __name__ == "__main__":
    pass
# arkitscene 40753679
# [DftPE] point_features num_patches:  556
# [DftPE] cur_input_ids:  torch.Size([221])
# [DftPE] point_start_token_id/point_end_token_id:  151652 151653
# [DftPE] point_start_token_pos:  tensor(14, device='cuda:0') point_end_token_pos:  tensor(16, device='cuda:0')
# [DftPE] cur_new_input_embeds:  torch.Size([776, 896])
# [DftPE] cur_new_attention_mask:  torch.Size([776])
# [DftPE] max_num_tokens in the batch with b_size1:  776

# arkitscene 40753686
# [DftPE] point_features num_patches:  507
# [DftPE] cur_input_ids:  torch.Size([221])
# [DftPE] point_start_token_id/point_end_token_id:  151652 151653
# [DftPE] point_start_token_pos:  tensor(14, device='cuda:0') point_end_token_pos:  tensor(16, device='cuda:0')
# [DftPE] cur_new_input_embeds:  torch.Size([727, 896]) #507+(221-1)
# [DftPE] cur_new_attention_mask:  torch.Size([727])
# [DftPE] max_num_tokens in the batch with b_size1:  727
# [DftPE] attention_mask: torch.Size([1, 727]) True
# [DftPE] attention_mask: torch.Size([1, 222]) True
# [DftPE] attention_mask: torch.Size([1, 223]) True
# ......
# [DftPE] attention_mask: torch.Size([1, 718]) True
# [DftPE] attention_mask: torch.Size([1, 719]) True
# [DftPE] attention_mask: torch.Size([1, 720]) True