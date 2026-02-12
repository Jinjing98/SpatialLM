import torch

@torch.no_grad()
def forward(self, x, position_ids, coords_points=None, point_token_pos=None, point_token_len=None):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    device = x.device
    batch_size, seq_len = position_ids.shape

    if coords_points is None or point_token_pos is None or point_token_len is None:
        # next token gen stage
        return super().forward(x, position_ids)

    if point_token_pos + point_token_len > seq_len:
        raise ValueError(f"out of bound: {point_token_pos} + {point_token_len} > {seq_len}")

    if coords_points.shape[1] != point_token_len:
        raise ValueError(f"not match: coords_points.shape[1]={coords_points.shape[1]} != point_token_len={point_token_len}")

    # print('Seq_len:',seq_len)
    # print('Point_token_pos:',point_token_pos)
    # print('Point_token_len:',point_token_len)
    assert seq_len > point_token_pos + point_token_len, f"seq_len={seq_len} > point_token_pos={point_token_pos} + point_token_len={point_token_len}"
    if point_token_pos > 0:
        pos_pre = position_ids[:, :point_token_pos] # (B, pre_len)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        pos_pre_expanded = pos_pre[:, None, :].float()
        freqs_pre = (inv_freq_expanded @ pos_pre_expanded).transpose(1, 2) # (B, pre_len, dim)
    else:
        assert False, "should not reach here"
        freqs_pre = torch.empty(batch_size, 0, len(self.inv_freq), device=device)

    # pos_points = position_ids[:, point_token_pos:point_token_pos + point_token_len] # (B, point_len)

    t_tensor = torch.arange(point_token_len, device=device, dtype=torch.float32)# JJ. prefill so can start 0
    freqs_points_single = self._compute_3d_frequencies(coords_points, t_tensor, device) # (point_len, dim)

    freqs_points = freqs_points_single.unsqueeze(0).expand(batch_size, -1, -1) # (B, point_len, dim)

    if point_token_pos + point_token_len < seq_len:
        pos_last = position_ids[:, point_token_pos + point_token_len:] # (B, last_len)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        pos_last_expanded = pos_last[:, None, :].float()
        freqs_last = (inv_freq_expanded @ pos_last_expanded).transpose(1, 2) # (B, last_len, dim)
    else:
        assert False, "should not reach here"
        freqs_last = torch.empty(batch_size, 0, len(self.inv_freq), device=device)

    freqs = torch.cat([freqs_pre, freqs_points, freqs_last], dim=1) # (B, seq_len, dim)

    # Note: device_type logic truncated in image
    return freqs


def _compute_3d_frequencies(self, coords_points, t_tensor, device):
    batch_size, seq_len_points, _ = coords_points.shape

    spherical_coords = self._cartesian_to_spherical(coords_points) # (B, seq_len_points, 3)

    normalized_spherical = self._normalize_spherical_coordinates(spherical_coords) # (B, seq_len_points, 3)

    coords_mean = normalized_spherical.mean(dim=0) # (seq_len_points, 3)
    r_coords = coords_mean[:, 0]
    theta_coords = coords_mean[:, 1]
    phi_coords = coords_mean[:, 2]

    t_start, t_end = self.freq_splits['t']
    r_start, r_end = self.freq_splits['r']
    theta_start, theta_end = self.freq_splits['theta']
    phi_start, phi_end = self.freq_splits['phi']

    inv_freq_t = self.inv_freq[t_start:t_end]
    
    inv_freq_r = self.inv_freq[r_start:r_end]
    inv_freq_theta = self.inv_freq[theta_start:theta_end]
    inv_freq_phi = self.inv_freq[phi_start:phi_end]

    freqs_t = torch.outer(t_tensor.float(), inv_freq_t)             # (seq_len_points, t_dim)

    freqs_r = torch.outer(r_coords.float(), inv_freq_r)             # (seq_len_points, r_dim)
    freqs_theta = torch.outer(theta_coords.float(), inv_freq_theta) # (seq_len_points, theta_dim)
    freqs_phi = torch.outer(phi_coords.float(), inv_freq_phi)       # (seq_len_points, phi_dim)

    freqs_combined = torch.cat([freqs_t, freqs_r, freqs_theta, freqs_phi], dim=1)

    return freqs_combined


def _cartesian_to_spherical(self, coords):
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    r = torch.sqrt(x**2 + y**2 + z**2)

    theta = torch.atan2(y, x)

    phi = torch.where(r > 1e-8, torch.acos(torch.clamp(z / r, -1, 1)), torch.zeros_like(r))

    spherical_coords = torch.stack([r, theta, phi], dim=-1)

    return spherical_coords


def _normalize_spherical_coordinates(self, spherical_coords, normalization_strategy="minmax", scale_factor=22.0):
    r, theta, phi = spherical_coords[..., 0], spherical_coords[..., 1], spherical_coords[..., 2]

    if normalization_strategy == "minmax":
        r_min = r.min(dim=1, keepdim=True)[0] # (B, 1)
        r_max = r.max(dim=1, keepdim=True)[0] # (B, 1)
        r_range = torch.clamp(r_max - r_min, min=1e-6)
        r_norm = (r - r_min) / r_range * scale_factor

        theta_norm = (theta + torch.pi) / (2 * torch.pi) * scale_factor
        phi_norm = phi / torch.pi * scale_factor

    elif normalization_strategy == "zscore":
        r_mean = r.mean(dim=1, keepdim=True)
        r_std = torch.clamp(r.std(dim=1, keepdim=True), min=1e-6)
        r_norm = torch.clamp((r - r_mean) / r_std * 3.67 + 11, 0, scale_factor)

        theta_norm = (theta + torch.pi) / (2 * torch.pi) * scale_factor
        phi_norm = phi / torch.pi * scale_factor

    else: # "raw"
        r_norm = r
        theta_norm = theta
        phi_norm = phi

    normalized_coords = torch.stack([r_norm, theta_norm, phi_norm], dim=-1)
    return normalized_coords