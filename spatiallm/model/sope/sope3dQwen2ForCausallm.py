# JJ: SOPE (Structured Orientation Positional Encoding) implementation for Qwen2
# This implements spherical coordinate-based 3D RoPE for point cloud tokens

import torch
import torch.nn as nn
import types
from typing import Optional, Tuple
from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Config
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast


class Qwen2ModelSope(Qwen2Model):
    """Qwen2Model with SOPE (Spherical Coordinate-Based Positional Encoding) support"""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # JJ: SOPE coordinate system configuration
        self.sope_coordinate_system = getattr(config, 'sope_coordinate_system', 'spherical')
        
        print(f"[SOPE] Using standard Qwen2DecoderLayer (position_embeddings computed at model level)")
        
        if self.sope_coordinate_system == 'spherical':
            self._setup_spherical_sope(config)
        else:
            raise ValueError(f"Unsupported coordinate system: {self.sope_coordinate_system}")
        
        # Re-initialize post init
        self.post_init()
    
    def _setup_spherical_sope(self, config: Qwen2Config):
        """
        Setup spherical coordinate system (SOPE core) by monkey patching rotary_emb.
        
        This validates freq_splits configuration and attaches SOPE functions to rotary_emb.
        """
        from spatiallm.model.sope import sope_core_utils
        
        # Validate freq_splits configuration
        head_dim = config.hidden_size // config.num_attention_heads
        freq_splits_config = config.freq_splits
        
        # Convert freq_splits from list [start, end] to tuple (start, end)
        # JJ: Divide by 2 because RoPE freqs dimension is head_dim // 2
        freq_splits = {}
        for key in ['t', 'r', 'theta', 'phi']:
            range_list = freq_splits_config.get(key, [0, head_dim // 4])
            if isinstance(range_list, (list, tuple)) and len(range_list) == 2:
                # Divide by 2 to convert from head_dim scale to freq_dim scale
                freq_splits[key] = (range_list[0] // 2, range_list[1] // 2)
            else:
                raise ValueError(
                    f"freq_splits['{key}'] must be a list/tuple of 2 elements [start, end], "
                    f"got: {range_list}"
                )
        
        # Validate ranges cover entire head_dim // 2 (RoPE frequency dimension) without gaps/overlaps
        freq_dim = head_dim // 2
        all_indices = set()
        for key in ['t', 'r', 'theta', 'phi']:
            start, end = freq_splits[key]
            if start >= end:
                raise ValueError(f"freq_splits['{key}'] invalid range: start ({start}) >= end ({end})")
            for i in range(start, end):
                if i in all_indices:
                    raise ValueError(f"freq_splits['{key}'] overlaps with previous ranges at index {i}")
                all_indices.add(i)
        
        # Check coverage
        expected_indices = set(range(freq_dim))
        if all_indices != expected_indices:
            missing = expected_indices - all_indices
            extra = all_indices - expected_indices
            raise ValueError(
                f"freq_splits does not cover freq_dim (head_dim // 2 = {freq_dim}) correctly.\n"
                f"  Missing indices: {sorted(missing) if missing else 'None'}\n"
                f"  Extra indices: {sorted(extra) if extra else 'None'}\n"
                f"  Got freq_splits: {freq_splits}"
            )
        
        # Monkey patch rotary_emb with SOPE functions
        self.rotary_emb.forward_original = self.rotary_emb.forward
        self.rotary_emb.forward = types.MethodType(sope_core_utils.forward, self.rotary_emb)
        self.rotary_emb._compute_3d_frequencies = types.MethodType(
            sope_core_utils._compute_3d_frequencies, self.rotary_emb
        )
        self.rotary_emb._cartesian_to_spherical = types.MethodType(
            sope_core_utils._cartesian_to_spherical, self.rotary_emb
        )
        self.rotary_emb._normalize_spherical_coordinates = types.MethodType(
            sope_core_utils._normalize_spherical_coordinates, self.rotary_emb
        )
        
        # Set freq_splits and normalization config
        self.rotary_emb.freq_splits = freq_splits
        self.rotary_emb.spherical_norm_strategy = getattr(config, 'spherical_norm_strategy', 'minmax')
        self.rotary_emb.spherical_scale_factor = getattr(config, 'spherical_scale_factor', 22.0)
        
        # Wrap _normalize_spherical_coordinates to use config parameters
        original_normalize = self.rotary_emb._normalize_spherical_coordinates
        def _normalize_spherical_coordinates_wrapper(spherical_coords, normalization_strategy=None, scale_factor=None):
            # Use config values if not provided
            norm_strat = normalization_strategy or self.rotary_emb.spherical_norm_strategy
            scale_fac = scale_factor or self.rotary_emb.spherical_scale_factor
            return original_normalize(spherical_coords, norm_strat, scale_fac)
        
        self.rotary_emb._normalize_spherical_coordinates = _normalize_spherical_coordinates_wrapper
        
        print(f"[SOPE] Monkey patched rotary_emb with spherical coordinate system")
        print(f"[SOPE] freq_splits: {freq_splits}, norm_strategy: {self.rotary_emb.spherical_norm_strategy}, scale_factor: {self.rotary_emb.spherical_scale_factor}")
    
    def _compute_spherical_position_embeddings(
        self,
        inputs_embeds: torch.FloatTensor,
        position_ids: torch.LongTensor,
        point_coords: torch.Tensor,
        point_token_pos: int,
        point_token_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute position embeddings using spherical coordinates (r, θ, φ) for SOPE.
        
        This method converts Cartesian (x,y,z) to spherical coordinates, then computes
        3D RoPE frequencies, and finally returns (cos, sin) for all tokens.
        
        Args:
            inputs_embeds: [B, seq_len, hidden_size]
            position_ids: [B, seq_len]
            point_coords: [N_point, 3] or [B, N_point, 3] cartesian coordinates
            point_token_pos: starting position of point tokens
            point_token_len: number of point tokens
        
        Returns:
            (cos, sin) tuple for position embeddings [B, seq_len, head_dim]
        """
        # sope_core_utils.forward returns freqs tensor [B, seq_len, head_dim // 2]
        freqs = self.rotary_emb(
            inputs_embeds,
            position_ids,
            coords_points=point_coords.unsqueeze(0) if point_coords.dim() == 2 else point_coords,  # [B, N, 3]
            point_token_pos=point_token_pos,
            point_token_len=point_token_len
        )
        
        # JJ: Expand freqs to head_dim (repeat interleave pattern)
        # Standard RoPE uses: emb = cat([freqs, freqs], dim=-1)
        # This is because rotate_half operation requires each frequency to appear twice
        freqs_expanded = torch.cat([freqs, freqs], dim=-1)  # [B, seq_len, head_dim]
        
        # Convert freqs to (cos, sin)
        cos = torch.cos(freqs_expanded)  # [B, seq_len, head_dim]
        sin = torch.sin(freqs_expanded)  # [B, seq_len, head_dim]
        
        # JJ: Ensure cos/sin have same dtype as model embeddings (important for BFloat16 inference)
        # This prevents dtype mismatch in attention during eval(fp16): query/key (after RoPE) vs value
        # Only convert if dtypes differ (no-op if already same dtype, e.g., float32 in training)
        target_dtype = inputs_embeds.dtype
        if cos.dtype != target_dtype:
            cos = cos.to(dtype=target_dtype)
            sin = sin.to(dtype=target_dtype)
        
        return (cos, sin)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Extended forward to support point_coords and point_token_mask for SOPE.
        
        Key modification: Compute position_embeddings once at model level using
        spherical coordinates for point tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, inputs_embeds.shape[:2], inputs_embeds, past_key_values
            )
        
        # Compute position embeddings
        position_embeddings = None
        assert hasattr(self, 'rotary_emb') and self.rotary_emb is not None, "rotary_emb is not found"
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, inputs_embeds.shape[1], dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        # JJ: SOPE - compute position_embeddings for spherical coordinate system
        if point_coords is not None and point_token_mask is not None:
            # Extract point token positions
            point_indices = point_token_mask[0].nonzero(as_tuple=True)[0]
            assert len(point_indices) > 0, f"point_token_mask is empty"
            
            point_token_pos = point_indices[0].item()
            point_token_len = len(point_indices)
            
            # Compute spherical position embeddings
            position_embeddings = self._compute_spherical_position_embeddings(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                point_coords=point_coords,
                point_token_pos=point_token_pos,
                point_token_len=point_token_len
            )
        else:
            # No point clouds, use standard 1D RoPE
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # Pass through all decoder layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values):
        """
        Prepare causal attention mask with proper handling of KV cache.
        
        Key insight: During generation with KV cache, attention_mask.shape[1] may not match
        the cached sequence length because point tokens were inserted during prefill.
        """
        batch_size, seq_length = input_shape
        
        # Get the past sequence length
        past_key_values_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                past_key_values_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                # Legacy cache format: list of (key, value) tuples
                past_key_values_length = past_key_values[0][0].shape[2]
        
        # Handle attention_mask mismatch during generation
        if attention_mask is not None and past_key_values_length > 0:
            expected_length = past_key_values_length + seq_length
            actual_length = attention_mask.shape[1]
            
            if actual_length < expected_length:
                # attention_mask is shorter than KV cache, need to expand it
                padding_length = expected_length - actual_length
                attention_mask = torch.nn.functional.pad(
                    attention_mask, 
                    (0, padding_length), 
                    value=1  # Can attend to point tokens
                )
        
        # Create causal mask: [batch_size, 1, tgt_len, src_len]
        if seq_length > 1:
            # Prefill or multi-token generation: create full causal mask
            causal_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        else:
            # Single token generation: no causal masking needed within this token
            causal_mask = None
        
        # Expand the user-provided attention_mask
        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(
                attention_mask, 
                inputs_embeds.dtype, 
                tgt_len=seq_length
            )
            
            if causal_mask is not None:
                combined_attention_mask = expanded_attn_mask + causal_mask
            else:
                combined_attention_mask = expanded_attn_mask
        else:
            combined_attention_mask = causal_mask
        
        return combined_attention_mask
    
    def _make_causal_mask(self, input_shape, dtype, device, past_key_values_length=0):
        """
        Make causal mask for autoregressive decoding.
        Returns: [batch_size, 1, tgt_len, src_len + past_key_values_length]
        """
        bsz, tgt_len = input_shape
        
        # Create causal mask: upper triangle is masked (filled with -inf)
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        
        # If we have past key values, prepend zeros for those positions (can attend to all past)
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        
        # Expand to batch: [bsz, 1, tgt_len, tgt_len + past_key_values_length]
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    def _expand_mask(self, mask, dtype, tgt_len=None):
        """
        Expand attention mask from [bsz, src_len] to [bsz, 1, tgt_len, src_len].
        Inverts the mask: 0 becomes -inf (masked), 1 stays 0 (unmasked).
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        
        # Expand: [bsz, src_len] -> [bsz, 1, tgt_len, src_len]
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        
        # Invert: 1.0 -> 0.0 (can attend), 0.0 -> -inf (cannot attend)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class Qwen2ForCausalLMSope(Qwen2ForCausalLM):
    """Qwen2ForCausalLM with SOPE support"""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace model with our custom one
        self.model = Qwen2ModelSope(config)
        # Re-initialize weights
        self.post_init()


if __name__ == "__main__":
    import torch
    from transformers import AutoConfig
    
    # Test loading
    print("Testing SOPE Qwen2 implementation...")
    
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Add SOPE config
    config.sope_coordinate_system = "spherical"
    config.freq_splits = {
        't': [0, 16],
        'r': [16, 32],
        'theta': [32, 48],
        'phi': [48, 64]
    }
    config.spherical_norm_strategy = "minmax"
    config.spherical_scale_factor = 22.0
    
    print(f"Config: {config}")
    print("✓ Configuration ready for SOPE")
