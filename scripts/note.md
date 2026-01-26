- 3D_RoPE only happend during prefill, during decoding it _spatial_3d_rope_data will be None.
- Does theta matters and how to set?  We set 3D_RoPE as 1000000(follow qwen1D); 3D_sinusodal as 10000.
    rope_theta = getattr(config, 'rope_theta', 1000000)  # Qwen2: 1000000, Llama: 10000
