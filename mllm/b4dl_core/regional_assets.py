"""Extract independent, versioned SST grid assets without changing model inputs.

The v2 format records the coordinates actually used by the encoder.  It is
deliberately separate from regional_data's v1 training loader.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys

import numpy as np

from .audit import raw_pose
from .inputs import GeometryDataset
from .io import atomic_output, digest, iter_array, read_json, read_jsonl, safe_file, sha256, unique_index, write_json
from .regional_data import CAMERA_FROM_SST, _verify_sources, ego_from_sst
from .sst_source import load_context, retained_points


SCHEMA = "b4dl-regional-features-v2"
GRID_SIZE = 80
POINT_CLOUD_RANGE = np.array([0., -20., -2., 40., 20., 4.])
ARRAY_FIELDS = {
    "global_features", "offsets", "indices", "values", "xyz", "point_counts", "out_of_range_counts",
    "global_from_lidar_ego", "global_from_camera_ego", "lidar_ego_from_lidar", "camera_ego_from_camera",
    "ego_from_sst",
}


def check_encoder_config(path):
    """Require the existing configuration for which the BEV scatter is defined."""
    config = runpy.run_path(str(path))
    model = config["model"]
    layer, middle, backbone = model["voxel_layer"], model["middle_encoder"], model["backbone"]
    if (tuple(layer["voxel_size"]) != (0.5, 0.5, 6)
            or list(layer["point_cloud_range"]) != POINT_CLOUD_RANGE.tolist()
            or layer["max_num_points"] != -1 or tuple(layer["max_voxels"]) != (-1, -1)
            or middle["type"] != "SSTInputLayerV2" or tuple(middle["sparse_shape"]) != (80, 80, 1)
            or backbone["type"] != "SSTv2" or tuple(backbone["output_shape"]) != (80, 80)
            or backbone["conv_out_channel"] != 128 or backbone["num_attached_conv"] != 0):
        raise ValueError("unsupported SST configuration for regional v2")


def verify_grid_sources(provenance, encoder_root):
    """Bind voxelization, surviving coordinates and scatter to the receipt."""
    files = ("sst/mmdet3d/ops/voxel/voxelize.py",
             "sst/mmdet3d/models/middle_encoders/sst_input_layer_v2.py",
             "sst/mmdet3d/models/backbones/sst_v2.py")
    checked = {}
    for name in files:
        matches = [value for path, value in provenance["source_files_sha256"].items()
                   if path.replace("\\", "/").endswith("/" + name)]
        actual = sha256(Path(encoder_root) / name)
        if not matches or len(set(matches)) != 1 or actual != matches[0]:
            raise ValueError("regional grid source mismatch: " + name)
        checked[name] = actual
    return checked


def verify_runtime_grid(sst, encoder_root):
    """Reject an already-imported or substituted encoder implementation."""
    expected = ("sst/mmdet3d/ops/voxel/voxelize.py",
                "sst/mmdet3d/models/middle_encoders/sst_input_layer_v2.py",
                "sst/mmdet3d/models/backbones/sst_v2.py")
    for module, name in zip((sst.voxel_layer, sst.middle_encoder, sst.backbone), expected):
        source = getattr(sys.modules[module.__class__.__module__], "__file__", "")
        if not source or Path(source).resolve() != (Path(encoder_root) / name).resolve():
            raise ValueError("unexpected runtime SST grid implementation: " + name)
    voxel_module = sys.modules[sst.voxel_layer.__class__.__module__]
    shim = sys.modules["lidarclip.model._mmdet3d_compat"]
    if getattr(voxel_module, "dynamic_voxelize", None) is not shim._dynamic_voxelize_py:
        raise ValueError("unexpected runtime SST dynamic voxelization")


@contextmanager
def capture_grid(sst):
    """Observe the existing single-frame forward; hooks do not modify tensors."""
    observed = {"point_coors": [], "kept_coors": []}

    def voxel_hook(_module, _args, output):
        observed["point_coors"].append(output.detach().cpu().numpy().copy())

    def middle_hook(_module, _args, output):
        observed["kept_coors"].append(output["voxel_coors"].detach().cpu().numpy().copy())

    handles = [sst.voxel_layer.register_forward_hook(voxel_hook),
               sst.middle_encoder.register_forward_hook(middle_hook)]
    try:
        yield observed
    finally:
        for handle in handles:
            handle.remove()


def cells_from_grid(bev, points, point_coors, kept_coors):
    """Join the observed per-point voxel IDs to the post-drop BEV cells.

    Coords are [z,y,x] per input point and [batch,z,y,x] per surviving
    voxel.  The point mean uses the original, unclipped SST coordinates of
    *all* points assigned to each surviving voxel by the encoder.
    """
    bev, points, point_coors, kept_coors = map(np.asarray, (bev, points, point_coors, kept_coors))
    if (bev.shape != (128, 80, 80) or points.ndim != 2 or points.shape[1] != 4
            or point_coors.shape != (len(points), 3) or kept_coors.ndim != 2 or kept_coors.shape[1] != 4
            or not np.issubdtype(point_coors.dtype, np.integer) or not np.issubdtype(kept_coors.dtype, np.integer)
            or not np.isfinite(bev).all() or not np.isfinite(points).all()):
        raise ValueError("invalid SST output, input points or observed voxel coordinates")
    # Some voxelizers use [-1,-1,-1] for rejected points.  Partial/invalid
    # coordinates are rejected instead of silently assigning them to a cell.
    rejected = np.all(point_coors == -1, axis=1)
    assigned = point_coors[~rejected]
    if ((assigned[:, 0] != 0).any() or (assigned[:, 1:] < 0).any()
            or (assigned[:, 1:] >= GRID_SIZE).any()):
        raise ValueError("invalid observed point voxel coordinate")
    if (len(kept_coors) == 0 or (kept_coors[:, :2] != 0).any()
            or (kept_coors[:, 2:] < 0).any() or (kept_coors[:, 2:] >= GRID_SIZE).any()):
        raise ValueError("invalid or empty surviving SST grid")
    ids = (kept_coors[:, 2] * GRID_SIZE + kept_coors[:, 3]).astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate surviving SST grid coordinates")
    ids.sort()
    point_ids = (assigned[:, 1] * GRID_SIZE + assigned[:, 2]).astype(np.int64)
    counts = np.bincount(point_ids, minlength=GRID_SIZE ** 2)
    if (counts[ids] == 0).any():
        raise ValueError("surviving SST cell has no assigned input points")
    xyz = points[~rejected, :3].astype(np.float64)
    outside = ((xyz < POINT_CLOUD_RANGE[:3]) | (xyz >= POINT_CLOUD_RANGE[3:])).any(axis=1)
    sums = np.zeros((GRID_SIZE ** 2, 3), dtype=np.float64)
    np.add.at(sums, point_ids, xyz)
    outside_counts = np.bincount(point_ids[outside], minlength=GRID_SIZE ** 2)
    flat = bev.reshape(128, GRID_SIZE ** 2).T
    not_kept = np.ones(GRID_SIZE ** 2, dtype=bool)
    not_kept[ids] = False
    if np.any(flat[not_kept] != 0):
        raise ValueError("BEV contains values outside the observed surviving grid")
    values = flat[ids].astype(np.float16)
    if not np.isfinite(values).all():
        raise ValueError("SST cell values overflow FP16 cache")
    original = flat[ids].astype(np.float32)
    error = original - values.astype(np.float32)
    norm = float(np.linalg.norm(original))
    stats = {
        "input_points": len(points), "rejected_points": int(rejected.sum()),
        "input_out_of_range_points": int(((points[:, :3] < POINT_CLOUD_RANGE[:3]) |
                                          (points[:, :3] >= POINT_CLOUD_RANGE[3:])).any(axis=1).sum()),
        "assigned_points_in_kept_cells": int(counts[ids].sum()),
        "out_of_range_points_in_kept_cells": int(outside_counts[ids].sum()),
        "input_occupied_cells": int(np.count_nonzero(counts)), "kept_cells": len(ids),
        "max_cell_abs_error": float(np.max(np.abs(error))),
        "cell_relative_l2": float(np.linalg.norm(error) / norm) if norm else 0.,
    }
    arrays = {"indices": ids.astype(np.int16), "values": values,
              "xyz": (sums[ids] / counts[ids, None]).astype(np.float32),
              "point_counts": counts[ids].astype(np.int32),
              "out_of_range_counts": outside_counts[ids].astype(np.int32)}
    return arrays, stats


def frame_metadata(context, sample_token):
    """Return sensor identities, timestamps and transforms for one keyframe."""
    sample = context["sample"][sample_token]
    lidar = context["sample_data"][sample["data"]["LIDAR_TOP"]]
    camera = context["sample_data"][sample["data"]["CAM_BACK_RIGHT"]]
    sensors = {"lidar": lidar, "camera": camera}
    meta, transforms = {"sample_token": sample_token}, {}
    for name, sensor in sensors.items():
        if (sensor.get("sample_token") != sample_token
                or sensor.get("token") != sample["data"]["LIDAR_TOP" if name == "lidar" else "CAM_BACK_RIGHT"]):
            raise ValueError(name + " sample identity mismatch")
        pose = context["ego_pose"][sensor["ego_pose_token"]]
        if (type(sensor["timestamp"]) is not int or type(pose.get("timestamp")) is not int
                or pose["timestamp"] != sensor["timestamp"]):
            raise ValueError(name + " sensor/ego timestamp mismatch")
        meta[name] = {"sample_data_token": sensor["token"], "timestamp_us": sensor["timestamp"],
                      "ego_pose_token": sensor["ego_pose_token"],
                      "calibrated_sensor_token": sensor["calibrated_sensor_token"]}
        transforms["global_from_" + name + "_ego"] = raw_pose(pose)
        transforms[name + "_ego_from_" + name] = raw_pose(
            context["calibrated_sensor"][sensor["calibrated_sensor_token"]])
    transforms["ego_from_sst"] = ego_from_sst(context, sample_token)
    return meta, transforms


def validate_scene(arrays, frames):
    """Validate the v2 on-disk arrays and frame identity/transform contract."""
    if set(arrays) != ARRAY_FIELDS or not isinstance(frames, list) or not frames:
        raise ValueError("invalid regional v2 fields or frame metadata")
    n = len(frames)
    offsets, indices = arrays["offsets"], arrays["indices"]
    count = len(indices)
    expected = {"global_features": ((n, 768), np.float32), "offsets": ((n + 1,), np.int64),
                "indices": ((count,), np.int16), "values": ((count, 128), np.float16),
                "xyz": ((count, 3), np.float32), "point_counts": ((count,), np.int32),
                "out_of_range_counts": ((count,), np.int32)}
    for key in ("global_from_lidar_ego", "global_from_camera_ego", "lidar_ego_from_lidar",
                "camera_ego_from_camera", "ego_from_sst"):
        expected[key] = ((n, 4, 4), np.float64)
    for key, (shape, dtype) in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
            raise ValueError("invalid regional v2 array: " + key)
    if (offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) <= 0)
            or (indices < 0).any() or (indices >= GRID_SIZE ** 2).any()
            or (arrays["point_counts"] <= 0).any() or (arrays["out_of_range_counts"] < 0).any()
            or (arrays["out_of_range_counts"] > arrays["point_counts"]).any()):
        raise ValueError("invalid regional v2 offsets, indices or point counts")
    inside_only = arrays["out_of_range_counts"] == 0
    selected = indices.astype(np.int64)
    x = (selected % GRID_SIZE) * .5
    y = (selected // GRID_SIZE) * .5 - 20.
    xyz = arrays["xyz"]
    outside_cell = ((xyz[:, 0] < x - 1e-4) | (xyz[:, 0] > x + .5001)
                    | (xyz[:, 1] < y - 1e-4) | (xyz[:, 1] > y + .5001)
                    | (xyz[:, 2] < -2.0001) | (xyz[:, 2] > 4.0001))
    if (inside_only & outside_cell).any():
        raise ValueError("inside-only regional v2 centroid outside its cell")
    for start, end in zip(offsets[:-1], offsets[1:]):
        if np.any(np.diff(indices[start:end].astype(np.int64)) <= 0):
            raise ValueError("regional v2 indices must be unique and sorted within each frame")
    for key in expected:
        if key.endswith("_ego") or key in ("lidar_ego_from_lidar", "camera_ego_from_camera", "ego_from_sst"):
            matrices = arrays[key]
            rotation = matrices[:, :3, :3]
            if (not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6, rtol=0)
                    or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
                raise ValueError("invalid regional v2 rigid transform: " + key)
    expected_transform = (np.linalg.inv(arrays["global_from_lidar_ego"]) @ arrays["global_from_camera_ego"]
                          @ arrays["camera_ego_from_camera"] @ CAMERA_FROM_SST)
    if not np.allclose(arrays["ego_from_sst"], expected_transform, atol=1e-8, rtol=0):
        raise ValueError("regional v2 ego_from_sst chain mismatch")
    sample_ids, lidar_ids, camera_ids, lidar_times = [], [], [], []
    for frame_index, frame in enumerate(frames):
        sample_ids.append(frame["sample_token"])
        for name, ids in (("lidar", lidar_ids), ("camera", camera_ids)):
            sensor = frame[name]
            ids.append(sensor["sample_data_token"])
            if (type(sensor["timestamp_us"]) is not int
                    or not all(isinstance(sensor[k], str) and sensor[k] for k in
                               ("sample_data_token", "ego_pose_token", "calibrated_sensor_token"))):
                raise ValueError("invalid regional v2 sensor metadata")
        lidar_times.append(frame["lidar"]["timestamp_us"])
        start, end = offsets[frame_index:frame_index + 2]
        counts = arrays["point_counts"][start:end]
        outside_counts = arrays["out_of_range_counts"][start:end]
        integer_fields = ("raw_points", "support_in_range_points", "input_points", "rejected_points",
                          "input_out_of_range_points", "assigned_points_in_kept_cells",
                          "out_of_range_points_in_kept_cells", "input_occupied_cells", "kept_cells")
        if (any(type(frame.get(key)) is not int or frame[key] < 0 for key in integer_fields)
                or frame["raw_points"] < frame["input_points"]
                or frame["input_points"] - frame["input_out_of_range_points"] != frame["support_in_range_points"]
                or frame["rejected_points"] > frame["input_points"]
                or frame["kept_cells"] != int(end - start)
                or frame["input_occupied_cells"] < frame["kept_cells"]
                or frame["assigned_points_in_kept_cells"] != int(counts.sum())
                or frame["assigned_points_in_kept_cells"] > frame["input_points"] - frame["rejected_points"]
                or frame["out_of_range_points_in_kept_cells"] != int(outside_counts.sum())
                or frame["out_of_range_points_in_kept_cells"] > frame["input_out_of_range_points"]):
            raise ValueError("regional v2 frame point/count metadata mismatch")
        image_size = frame.get("camera_image_size")
        raw_hash = frame.get("raw_lidar_sha256")
        if (not isinstance(image_size, list) or len(image_size) != 2
                or any(type(value) is not int or value <= 0 for value in image_size)
                or not isinstance(raw_hash, str) or len(raw_hash) != 64
                or any(char not in "0123456789abcdef" for char in raw_hash)):
            raise ValueError("invalid regional v2 raw input metadata")
        for key in ("global_max_abs_error", "global_cosine", "max_cell_abs_error", "cell_relative_l2"):
            if not isinstance(frame.get(key), (int, float)) or not np.isfinite(frame[key]):
                raise ValueError("invalid regional v2 error metadata")
    if (any(not isinstance(s, str) or not s for s in sample_ids)
            or any(len(ids) != len(set(ids)) for ids in (sample_ids, lidar_ids, camera_ids))
            or any(b <= a for a, b in zip(lidar_times, lidar_times[1:]))):
        raise ValueError("invalid regional v2 frame identities or order")


def load_scene(root, scene_id):
    """Read and verify one v2 asset; never routes it to a training loader."""
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != digest(
            {k: v for k, v in manifest.items() if k != "dataset_id"})
            or manifest.get("scene_count") != len(manifest.get("scenes", {}))):
        raise ValueError("invalid regional v2 manifest")
    record = manifest["scenes"][scene_id]
    path = safe_file(root, record["file"])
    if sha256(path) != record["sha256"]:
        raise ValueError("regional v2 scene hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    validate_scene(arrays, record["frames"])
    return arrays, record["frames"], manifest


def extract(geometry_root, table_root, data_root, encoder_root, support_path, output, *, device="cuda:0", max_scenes=None):
    """Create a new v2 asset directory with source and frame identities."""
    import torch

    geometry = GeometryDataset(geometry_root)
    provenance, checkpoint = _verify_sources(geometry, table_root, encoder_root)
    if sha256(support_path) != geometry.manifest.get("input_scope", {}).get("support_sha256"):
        raise ValueError("regional extraction input-support hash mismatch")
    support = unique_index(read_jsonl(support_path), "sample_token")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("regional SST extraction requires CUDA")
    config_path = Path(encoder_root) / "lidarclip/model/sst_encoder_only_config.py"
    check_encoder_config(config_path)
    grid_sources = verify_grid_sources(provenance, encoder_root)
    scene_ids = sorted(geometry.manifest["scenes"])
    if max_scenes is not None:
        if type(max_scenes) is not int or max_scenes < 1:
            raise ValueError("max_scenes must be positive")
        scene_ids = scene_ids[:max_scenes]
    selected = {token for scene in scene_ids for token in geometry.manifest["scenes"][scene]["frame_tokens"]}
    data = {row["token"]: row for row in iter_array(Path(table_root) / "sample_data.json")
            if row["token"] in selected}
    if set(data) != selected:
        raise ValueError("missing LiDAR sample_data for regional extraction")
    context = load_context(table_root, {row["sample_token"] for row in data.values()})
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new regional feature directory")
    sys.path.insert(0, str(Path(encoder_root).resolve()))
    from lidarclip.model.sst import LidarEncoderSST
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = {key[len("lidar_encoder."):]: value for key, value in payload["state_dict"].items()
             if key.startswith("lidar_encoder.")}
    encoder = LidarEncoderSST(str(config_path), 768)
    verify_runtime_grid(encoder._sst, encoder_root)
    loaded = encoder.load_state_dict(state, strict=False)
    if loaded.missing_keys:
        raise ValueError("regional extraction encoder is incomplete")
    encoder.to(device).eval()
    output.mkdir(parents=True)
    (output / "scenes").mkdir()
    records = {}
    try:
        for ident in scene_ids:
            source = geometry.manifest["scenes"][ident]
            reference = geometry.scene(ident)
            if reference["features"].dtype != np.float32:
                raise ValueError("historical global features must be float32 for regional v2")
            cell_parts = {key: [] for key in ("indices", "values", "xyz", "point_counts", "out_of_range_counts")}
            transform_parts = {key: [] for key in ("global_from_lidar_ego", "global_from_camera_ego",
                               "lidar_ego_from_lidar", "camera_ego_from_camera", "ego_from_sst")}
            offsets, frames = [0], []
            for frame_index, token in enumerate(source["frame_tokens"]):
                sample_token = data[token]["sample_token"]
                meta, transforms = frame_metadata(context, sample_token)
                if (meta["lidar"]["sample_data_token"] != token
                        or meta["lidar"]["timestamp_us"] != source["timestamps_us"][frame_index]
                        or not np.allclose(transforms["global_from_lidar_ego"], reference["frame_geometry"][frame_index], atol=1e-8, rtol=0)
                        or not np.allclose(transforms["lidar_ego_from_lidar"], reference["ego_from_lidar"][frame_index], atol=1e-8, rtol=0)):
                    raise ValueError("regional frame or LiDAR pose does not match geometry scene")
                points, _, receipt = retained_points(context, sample_token, data_root)
                retained = support.get(sample_token, {})
                if (retained.get("scene_id") != ident or retained.get("frame_index") != frame_index
                        or any(retained.get(key) != receipt[key] for key in
                               ("raw_lidar_sha256", "lidar_sample_data_token", "camera_sample_data_token",
                                "raw_points", "camera_crop_points", "retained_points"))):
                    raise ValueError("retained LiDAR input support mismatch")
                torch.manual_seed(0)
                with torch.no_grad(), capture_grid(encoder._sst) as observed:
                    bev = encoder._sst.extract_feat([torch.from_numpy(points).to(device)], None)[0]
                    global_feature, _ = encoder._pooler(bev)
                if len(observed["point_coors"]) != 1 or len(observed["kept_coors"]) != 1:
                    raise ValueError("expected one observed SST voxelization and middle-layer output")
                actual = global_feature[0].float().cpu().numpy()
                expected = reference["features"][frame_index]
                norm = np.linalg.norm(actual) * np.linalg.norm(expected)
                cosine = float(np.dot(actual, expected) / norm) if norm else 0.
                error = float(np.max(np.abs(actual - expected)))
                if not np.allclose(actual, expected, atol=5e-4, rtol=1e-4) or cosine <= .999999:
                    raise ValueError(f"historical global feature mismatch in {ident} frame {frame_index}: {error}")
                cells, stats = cells_from_grid(bev[0].float().cpu().numpy(), points,
                                               observed["point_coors"][0], observed["kept_coors"][0])
                for key in cell_parts:
                    cell_parts[key].append(cells[key])
                for key in transform_parts:
                    transform_parts[key].append(transforms[key])
                offsets.append(offsets[-1] + len(cells["indices"]))
                frames.append({**meta, "raw_lidar_sha256": receipt["raw_lidar_sha256"],
                               "camera_image_size": receipt["camera_image_size"],
                               "raw_points": receipt["raw_points"], "support_in_range_points": receipt["retained_points"],
                               "global_max_abs_error": error, "global_cosine": cosine, **stats})
            arrays = {key: np.concatenate(parts) for key, parts in cell_parts.items()}
            arrays.update({key: np.asarray(parts, dtype=np.float64) for key, parts in transform_parts.items()})
            arrays.update(global_features=np.asarray(reference["features"], dtype=np.float32),
                          offsets=np.asarray(offsets, dtype=np.int64))
            validate_scene(arrays, frames)
            filename = f"scenes/{ident}.npz"
            path = safe_file(output, filename)
            with atomic_output(path, "wb") as stream:
                np.savez(stream, **arrays)
            records[ident] = {"file": filename, "sha256": sha256(path), "frames": frames,
                              "geometry_scene_sha256": source["sha256"],
                              "source_feature_sha256": source["source_feature_sha256"]}
            print(f"regional v2 {len(records)}/{len(scene_ids)} {ident}", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "extraction_failed", "type": type(error).__name__,
                   "message": str(error), "completed_scenes": sorted(records)})
        raise
    manifest = {"schema": SCHEMA, "geometry_dataset_id": geometry.manifest["dataset_id"],
        "geometry_manifest_sha256": sha256(Path(geometry_root) / "manifest.json"),
        "encoder_provenance_id": provenance["provenance_id"], "encoder_checkpoint_sha256": sha256(checkpoint),
        "support_sha256": sha256(support_path), "complete": len(scene_ids) == len(geometry.manifest["scenes"]),
        "scene_count": len(records), "scenes": records, "encoder_grid_source_sha256": grid_sources,
        "implementation_sha256": {"regional_assets.py": sha256(Path(__file__)),
            "regional_data.py": sha256(Path(__file__).with_name("regional_data.py")),
            "sst_source.py": sha256(Path(__file__).with_name("sst_source.py")),
            "audit.py": sha256(Path(__file__).with_name("audit.py"))},
        "runtime": {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                    "numpy": np.__version__, "device": str(torch.device(device)),
                    "device_name": torch.cuda.get_device_name(torch.device(device)), "batch_size": 1},
        "preprocessing": {"camera": "CAM_BACK_RIGHT", "axes": "camera-z,-camera-x,-camera-y",
            "min_camera_depth_m_exclusive": .5, "image_crop": "center_square",
            "point_cloud_range": POINT_CLOUD_RANGE.tolist(), "voxel_size": [.5, .5, 6.],
            "point_assignment": "observed_voxel_layer_output", "valid_cells": "middle_encoder_voxel_coors_after_drop",
            "centroid": "mean_of_original_sst_xyz_of_all_assigned_points", "seed_per_frame": 0,
            "encoder_mode": "eval", "cell_values_dtype": "float16", "global_values_dtype": "float32"},
        "global_tolerance": {"atol": 5e-4, "rtol": 1e-4, "min_cosine": .999999}}
    manifest["dataset_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry-dataset", "tables", "data-root", "encoder-root", "support", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--gpu-lock", required=True)
    args = parser.parse_args(argv)
    from .b3 import device_lease
    with device_lease(args.device, args.gpu_lock):
        result = extract(args.geometry_dataset, args.tables, args.data_root, args.encoder_root, args.support,
                         args.output, device=args.device, max_scenes=args.max_scenes)
    print(f"regional v2 {result['dataset_id']} scenes={result['scene_count']} complete={result['complete']}")


if __name__ == "__main__":
    main()
