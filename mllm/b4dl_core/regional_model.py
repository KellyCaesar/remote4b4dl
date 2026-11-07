"""Region-aware spatiotemporal projection for the original B3 visual path."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .temporal import TemporalBlock, TemporalConfig, relative_geometry, validate_frames


@dataclass(frozen=True)
class RegionalConfig:
    input_width: int = 128
    width: int = 768
    heads: int = 8
    layers: int = 2
    ff_width: int = 1536
    tokens_per_frame: int = 1
    use_temporal: bool = True
    use_geometry: bool = True
    distance_scale_m: float = 10.0
    time_scale_s: float = 4.0
    radius_m: float = 10.0
    neighbors: int = 8
    centroid_policy: str = "strict_cell"

    def __post_init__(self):
        integers = (self.input_width, self.width, self.heads, self.layers, self.ff_width, self.neighbors,
                    self.tokens_per_frame)
        if (any(type(value) is not int for value in integers)
                or min(self.input_width, self.width, self.heads, self.layers, self.ff_width, self.neighbors) < 1
                or self.width % self.heads or self.tokens_per_frame not in (1, 5)
                or type(self.use_temporal) is not bool or type(self.use_geometry) is not bool
                or self.centroid_policy not in ("strict_cell", "observed_assignment")):
            raise ValueError("invalid regional architecture")
        if not all(type(value) in (int, float) and 0 < value < float("inf") for value in
                   (self.distance_scale_m, self.time_scale_s, self.radius_m)):
            raise ValueError("invalid regional physical scales")

    def as_dict(self):
        return asdict(self)


def region_ids(indices):
    """Map SST row-major 80x80 indices to row-major 8x8 regions."""
    y = torch.div(indices, 80, rounding_mode="floor")
    x = indices.remainder(80)
    return torch.div(y, 10, rounding_mode="floor") * 8 + torch.div(x, 10, rounding_mode="floor")


def visual_token_mask(frame_valid, indices, cell_valid, tokens_per_frame):
    if tokens_per_frame == 1:
        return frame_valid.unsqueeze(-1)
    if tokens_per_frame != 5:
        raise ValueError("regional projection supports one or five tokens per frame")
    region = region_ids(indices.clamp(0, 6399))
    quadrant = torch.div(region, 32, rounding_mode="floor") * 2 + (region.remainder(8) >= 4).long()
    occupied = torch.zeros(*frame_valid.shape, 4, dtype=torch.long, device=indices.device)
    occupied.scatter_add_(2, quadrant, cell_valid.long())
    return torch.cat((frame_valid.unsqueeze(-1), occupied.bool() & frame_valid.unsqueeze(-1)), dim=-1)


def _validate_cells(features, xyz, indices, valid, frame_valid, ego_from_sst, width, centroid_policy="strict_cell"):
    batch, frames = frame_valid.shape
    if (features.ndim != 4 or features.shape[:2] != (batch, frames) or features.shape[-1] != width
            or xyz.shape != (*features.shape[:3], 3) or indices.shape != features.shape[:3]
            or valid.shape != features.shape[:3] or valid.dtype != torch.bool
            or indices.dtype != torch.long or ego_from_sst.shape != (batch, frames, 4, 4)):
        raise ValueError("invalid regional cell tensors")
    if any(value.device != features.device for value in (xyz, indices, valid, frame_valid, ego_from_sst)):
        raise ValueError("regional cell tensors must share a device")
    if (valid & ~frame_valid.unsqueeze(-1)).any() or ((indices[valid] < 0) | (indices[valid] >= 6400)).any():
        raise ValueError("invalid regional cell index or padded frame")
    if not torch.isfinite(features[valid]).all() or not torch.isfinite(xyz[valid]).all():
        raise ValueError("regional cells must be finite")
    selected = indices[valid]
    x = selected.remainder(80).to(xyz.dtype) * .5
    y = torch.div(selected, 80, rounding_mode="floor").to(xyz.dtype) * .5 - 20
    point = xyz[valid]
    if centroid_policy == "strict_cell" and (((point[:, 0] < x - 1e-4) | (point[:, 0] > x + .5001)
         | (point[:, 1] < y - 1e-4) | (point[:, 1] > y + .5001)
         | (point[:, 2] < -2.0001) | (point[:, 2] > 4.0001)).any()):
        raise ValueError("regional centroid outside its SST cell")
    poses = ego_from_sst[frame_valid].double()
    eye = torch.eye(3, dtype=poses.dtype, device=poses.device)
    bottom = poses.new_tensor([0, 0, 0, 1]).expand(len(poses), -1)
    rotation = poses[:, :3, :3]
    if (not torch.isfinite(poses).all() or not torch.allclose(poses[:, 3], bottom, atol=1e-5)
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation, eye.expand_as(rotation), atol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation), poses.new_ones(len(poses)), atol=1e-4)):
        raise ValueError("invalid SST to ego transform")
    # Duplicate grid cells would change both the local attention and position average.
    ordered = indices.masked_fill(~valid, 6400).sort(dim=-1).values
    if ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] < 6400)).any():
        raise ValueError("duplicate regional cell index")


class LocalRegionPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        local_width = min(config.width, 128)
        self.features = nn.Linear(config.input_width, local_width)
        self.position = nn.Linear(3, local_width)
        self.score = nn.Linear(local_width, 1)
        self.output = nn.Linear(local_width, config.width)

    def forward(self, features, xyz, indices, valid):
        batch, frames, count, _ = features.shape
        if count == 0:
            raise ValueError("regional tensors require at least one padded cell slot")
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        xyz = xyz.masked_fill(~valid.unsqueeze(-1), 0)
        groups = region_ids(indices.clamp(0, 6399))
        region_x = groups.remainder(8).to(xyz.dtype) * 5 + 2.5
        region_y = torch.div(groups, 8, rounding_mode="floor").to(xyz.dtype) * 5 - 17.5
        offset = torch.stack(((xyz[..., 0] - region_x) / 5, (xyz[..., 1] - region_y) / 5,
                              xyz[..., 2] / 6), dim=-1)
        hidden = torch.tanh(self.features(features) + self.position(offset.to(features.dtype)))
        logits = self.score(hidden).squeeze(-1).float()
        flat_groups = groups.reshape(batch * frames, count)
        flat_valid = valid.reshape(batch * frames, count)
        flat_logits = logits.reshape(batch * frames, count)
        floor = torch.finfo(flat_logits.dtype).min
        maxima = flat_logits.new_full((batch * frames, 64), floor)
        maxima.scatter_reduce_(1, flat_groups, flat_logits.masked_fill(~flat_valid, floor),
                               reduce="amax", include_self=True)
        centred = (flat_logits - maxima.gather(1, flat_groups)).masked_fill(~flat_valid, 0)
        weights = torch.exp(centred).masked_fill(~flat_valid, 0)
        denominator = weights.new_zeros(batch * frames, 64)
        denominator.scatter_add_(1, flat_groups, weights)
        weights = weights / denominator.gather(1, flat_groups).clamp_min(1e-12)
        local_width = hidden.shape[-1]
        pooled = hidden.new_zeros(batch * frames, 64, local_width)
        pooled.scatter_add_(1, flat_groups.unsqueeze(-1).expand(-1, -1, local_width),
                            hidden.reshape(batch * frames, count, local_width) * weights.unsqueeze(-1).to(hidden.dtype))
        centres = xyz.new_zeros(batch * frames, 64, 3)
        centres.scatter_add_(1, flat_groups.unsqueeze(-1).expand(-1, -1, 3),
                             xyz.reshape(batch * frames, count, 3) * weights.unsqueeze(-1).to(xyz.dtype))
        occupied = denominator.gt(0)
        tokens = self.output(pooled).masked_fill(~occupied.unsqueeze(-1), 0)
        return tokens.reshape(batch, frames, 64, -1), centres.reshape(batch, frames, 64, 3), occupied.reshape(batch, frames, 64)


class RegionalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_norm = nn.LayerNorm(config.width)
        self.spatial = nn.MultiheadAttention(config.width, config.heads, dropout=0, batch_first=True)
        self.spatial_ff_norm = nn.LayerNorm(config.width)
        self.spatial_ff = nn.Sequential(nn.Linear(config.width, config.ff_width), nn.GELU(),
                                        nn.Linear(config.ff_width, config.width))
        if config.use_temporal:
            self.temporal_norm = nn.LayerNorm(config.width)
            self.temporal = nn.MultiheadAttention(config.width, config.heads, dropout=0, batch_first=True)
            self.temporal_bias = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, config.heads))
            self.global_block = TemporalBlock(TemporalConfig(width=config.width, heads=config.heads,
                layers=1, ff_width=config.ff_width, use_geometry=config.use_geometry,
                distance_scale_m=config.distance_scale_m, time_scale_s=config.time_scale_s))

    def forward(self, global_tokens, regions, centres, occupied, times, poses, frame_valid):
        batch, frames, count, width = regions.shape
        all_tokens = torch.cat((global_tokens.unsqueeze(2), regions), dim=2)
        all_valid = torch.cat((frame_valid.unsqueeze(-1), occupied), dim=-1)
        flat = all_tokens.reshape(batch * frames, count + 1, width)
        mask = all_valid.reshape(batch * frames, count + 1).clone()
        mask[:, 0] = True  # A padded frame needs one finite attention key.
        normalized = self.spatial_norm(flat)
        flat = flat + self.spatial(normalized, normalized, normalized, key_padding_mask=~mask, need_weights=False)[0]
        flat = flat + self.spatial_ff(self.spatial_ff_norm(flat))
        all_tokens = flat.reshape(batch, frames, count + 1, width).masked_fill(~all_valid.unsqueeze(-1), 0)
        global_tokens, regions = all_tokens[:, :, 0], all_tokens[:, :, 1:]
        if not self.config.use_temporal:
            return global_tokens, regions
        relation = relative_geometry(times, poses, frame_valid, TemporalConfig(width=self.config.width,
            heads=self.config.heads, layers=1, ff_width=self.config.ff_width,
            use_geometry=self.config.use_geometry, distance_scale_m=self.config.distance_scale_m,
            time_scale_s=self.config.time_scale_s))
        global_tokens = self.global_block(global_tokens, relation, frame_valid)
        if frames == 1 or not occupied.any():
            return global_tokens, regions

        # Each region attends to at most eight regions from adjacent frames plus itself.
        candidates = []
        candidate_centres = []
        candidate_valid = []
        candidate_dt = []
        candidate_yaw = []
        for offset in (-1, 1):
            source = torch.arange(frames, device=regions.device) + offset
            in_range = (source >= 0) & (source < frames)
            source = source.clamp(0, frames - 1)
            candidates.append(regions[:, source])
            candidate_centres.append(centres[:, source])
            candidate_valid.append(occupied[:, source] & in_range[None, :, None])
            candidate_dt.append((times[:, source] - times).float().unsqueeze(-1).expand(-1, -1, count))
            if self.config.use_geometry:
                candidate_yaw.append(relation[:, torch.arange(frames, device=regions.device), source, 3:5]
                                     .unsqueeze(2).expand(-1, -1, count, -1))
            else:
                candidate_yaw.append(regions.new_zeros(batch, frames, count, 2))
        candidates = torch.cat(candidates, dim=2)
        candidate_centres = torch.cat(candidate_centres, dim=2)
        candidate_valid = torch.cat(candidate_valid, dim=2)
        dt = torch.cat(candidate_dt, dim=2)
        yaw = torch.cat(candidate_yaw, dim=2)
        delta = candidate_centres.unsqueeze(2) - centres.unsqueeze(3)
        if self.config.use_geometry:
            distance = torch.linalg.vector_norm(delta[..., :2], dim=-1)
            allowed = distance <= self.config.radius_m
        else:
            grid = torch.arange(count, device=regions.device)
            y = torch.div(grid, 8, rounding_mode="floor")
            x = grid.remainder(8)
            grid_dist = (x[:, None] - x[None, :]).abs() + (y[:, None] - y[None, :]).abs()
            distance = grid_dist.repeat(1, 2).to(delta.dtype).expand(batch, frames, -1, -1)
            allowed = torch.ones_like(distance, dtype=torch.bool)
        allowed &= candidate_valid.unsqueeze(2) & occupied.unsqueeze(-1)
        distance = distance.masked_fill(~allowed, float("inf"))
        k = min(self.config.neighbors, 2 * count)
        nearest_distance, nearest = distance.topk(k, dim=-1, largest=False, sorted=True)
        gather = nearest.unsqueeze(-1).expand(-1, -1, -1, -1, width)
        selected = candidates.unsqueeze(2).expand(-1, -1, count, -1, -1).gather(3, gather)
        selected_delta = delta.gather(3, nearest.unsqueeze(-1).expand(-1, -1, -1, -1, 3))
        selected_dt = dt.unsqueeze(2).expand(-1, -1, count, -1).gather(3, nearest)
        selected_yaw = yaw.unsqueeze(2).expand(-1, -1, count, -1, -1).gather(
            3, nearest.unsqueeze(-1).expand(-1, -1, -1, -1, 2))
        query = regions.reshape(-1, 1, width)
        keys = torch.cat((regions.unsqueeze(3), selected), dim=3).reshape(-1, k + 1, width)
        key_valid = torch.cat((occupied.unsqueeze(-1), torch.isfinite(nearest_distance)), dim=-1).reshape(-1, k + 1)
        key_valid[:, 0] = True
        if self.config.use_geometry:
            # Centres are expressed relative to the first global pose; rotate offsets into each query ego frame.
            local = torch.einsum("btij,btrkj->btrki", poses[..., :3, :3].transpose(-1, -2).float(), selected_delta.float())
            values = torch.cat((local[..., :3] / self.config.distance_scale_m,
                                selected_yaw.float(), selected_dt.unsqueeze(-1) / self.config.time_scale_s), dim=-1)
            bias = self.temporal_bias(values.to(regions.dtype)).permute(0, 1, 2, 4, 3)
            bias = F.pad(bias, (1, 0)).reshape(-1, self.config.heads, 1, k + 1)
        else:
            bias = regions.new_zeros(batch * frames * count, self.config.heads, 1, k + 1)
        bias = bias.masked_fill(~key_valid[:, None, None, :], float("-inf")).reshape(-1, 1, k + 1)
        norm_query, norm_keys = self.temporal_norm(query), self.temporal_norm(keys)
        update = self.temporal(norm_query, norm_keys, norm_keys, attn_mask=bias, need_weights=False)[0]
        regions = (regions + update.reshape(batch, frames, count, width)).masked_fill(~occupied.unsqueeze(-1), 0)
        return global_tokens, regions


class RegionalProjector(nn.Module):
    def __init__(self, config, language_width):
        super().__init__()
        self.config = config
        self.local_pool = LocalRegionPool(config)
        self.blocks = nn.ModuleList(RegionalBlock(config) for _ in range(config.layers))
        self.residual = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU(),
                                      nn.Linear(config.width, language_width))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def encode(self, features, times, poses, valid, cells, xyz, indices, cell_valid, ego_from_sst):
        validate_frames(features, times, poses, valid, geometry=True)
        _validate_cells(cells, xyz, indices, cell_valid, valid, ego_from_sst, self.config.input_width,
                        self.config.centroid_policy)
        if features.shape[-1] != self.config.width:
            raise ValueError("global feature width mismatch")
        regions, centres_sst, occupied = self.local_pool(cells, xyz, indices, cell_valid)
        identity = torch.eye(4, device=poses.device, dtype=torch.float64)
        poses = torch.where(valid[..., None, None], poses.double(), identity)
        ego_from_sst = torch.where(valid[..., None, None], ego_from_sst.double(), identity)
        times = torch.where(valid, times, times[:, :1])
        transform = poses @ ego_from_sst
        centres = torch.einsum("btij,btrj->btri", transform[..., :3, :3], centres_sst.double())
        centres = centres + transform[..., None, :3, 3] - poses[:, :1, None, :3, 3].double()
        centres = centres.float().masked_fill(~occupied.unsqueeze(-1), 0)
        global_tokens = features.masked_fill(~valid.unsqueeze(-1), 0)
        for block in self.blocks:
            global_tokens, regions = block(global_tokens, regions, centres, occupied, times, poses, valid)
        return global_tokens, regions, occupied

    def forward(self, features, times, poses, valid, cells, xyz, indices, cell_valid, ego_from_sst, projector,
                output_dtype=None):
        global_tokens, regions, occupied = self.encode(features, times, poses, valid, cells, xyz, indices,
                                                       cell_valid, ego_from_sst)
        dtype = output_dtype or projector.weight.dtype
        weight = projector.weight.to(dtype)
        bias = projector.bias.to(dtype) if projector.bias is not None else None
        baseline = features.new_zeros(*features.shape[:2], weight.shape[0], dtype=dtype)
        baseline[valid] = F.linear(features[valid].to(dtype), weight, bias)
        global_output = baseline + self.residual(global_tokens).to(dtype)
        token_mask = visual_token_mask(valid, indices, cell_valid, self.config.tokens_per_frame)
        if self.config.tokens_per_frame == 1:
            return global_output.unsqueeze(2), token_mask
        region = torch.arange(64, device=features.device)
        quadrant = torch.div(region, 32, rounding_mode="floor") * 2 + (region.remainder(8) >= 4).long()
        pooled = regions.new_zeros(*regions.shape[:2], 4, regions.shape[-1])
        counts = regions.new_zeros(*regions.shape[:2], 4, 1)
        group = quadrant.reshape(1, 1, 64, 1).expand(*regions.shape[:3], regions.shape[-1])
        pooled.scatter_add_(2, group, regions * occupied.unsqueeze(-1))
        counts.scatter_add_(2, quadrant.reshape(1, 1, 64, 1).expand(*regions.shape[:3], 1), occupied.unsqueeze(-1).to(regions.dtype))
        pooled = pooled / counts.clamp_min(1)
        local_output = F.linear(pooled.to(dtype), weight, bias)
        return torch.cat((global_output.unsqueeze(2), local_output), dim=2), token_mask
