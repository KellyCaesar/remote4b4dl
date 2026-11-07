"""Extract and load versioned pre-pooling SST cells aligned to geometry scenes."""
import argparse
from collections import OrderedDict
from pathlib import Path, PurePosixPath
import sys

import numpy as np

from .audit import raw_pose
from .data import TABLES
from .inputs import GeometryDataset
from .io import atomic_output, digest, iter_array, read_json, read_jsonl, safe_file, sha256, unique_index, write_json
from .sst_source import load_context, retained_points


SCHEMA = "b4dl-regional-features-v1"
CAMERA_FROM_SST = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                            [1., 0., 0., 0.], [0., 0., 0., 1.]])


def ego_from_sst(context, sample_token):
    """Map SST input axes at camera time into the LiDAR-time ego frame."""
    sample = context["sample"][sample_token]
    lidar = context["sample_data"][sample["data"]["LIDAR_TOP"]]
    camera = context["sample_data"][sample["data"]["CAM_BACK_RIGHT"]]
    global_from_lidar_ego = raw_pose(context["ego_pose"][lidar["ego_pose_token"]])
    global_from_camera_ego = raw_pose(context["ego_pose"][camera["ego_pose_token"]])
    camera_ego_from_camera = raw_pose(context["calibrated_sensor"][camera["calibrated_sensor_token"]])
    return np.linalg.inv(global_from_lidar_ego) @ global_from_camera_ego @ camera_ego_from_camera @ CAMERA_FROM_SST


def sparse_cells(bev, points):
    """Keep occupied SST output cells and means of their actual input points."""
    bev, points = np.asarray(bev), np.asarray(points)
    if bev.shape != (128, 80, 80) or points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("expected 128x80x80 SST output and Nx4 input points")
    if not np.isfinite(bev).all() or not np.isfinite(points).all():
        raise ValueError("nonfinite SST features or input points")
    inside = ((points[:, :3] >= [0., -20., -2.]) & (points[:, :3] < [40., 20., 4.])).all(axis=1)
    selected = points[inside, :3].astype(np.float64)
    x = np.floor(selected[:, 0] / .5).astype(np.int64)
    y = np.floor((selected[:, 1] + 20.) / .5).astype(np.int64)
    point_ids = y * 80 + x
    counts = np.bincount(point_ids, minlength=6400)
    sums = np.zeros((6400, 3), dtype=np.float64)
    np.add.at(sums, point_ids, selected)
    flat = bev.reshape(128, 6400).T
    indices = np.flatnonzero(np.any(flat != 0, axis=1))
    if not len(indices) or (counts[indices] == 0).any():
        raise ValueError("SST output contains empty or unsupported occupied cells")
    values = flat[indices].astype(np.float16)
    if not np.isfinite(values).all():
        raise ValueError("SST values overflow FP16 cache")
    xyz = (sums[indices] / counts[indices, None]).astype(np.float32)
    return indices.astype(np.int16), values, xyz


def _verify_sources(geometry, table_root, encoder_root):
    table_root, encoder_root = Path(table_root), Path(encoder_root)
    for name in TABLES:
        matches = [value for path, value in geometry.manifest["sources"].items()
                   if PurePosixPath(path.replace("\\", "/")).name == name + ".json"]
        if len(matches) != 1 or sha256(table_root / (name + ".json")) != matches[0]:
            raise ValueError("raw source hash mismatch: " + name)
    record = geometry.manifest.get("encoder_provenance")
    if not record or sha256(record["path"]) != record["sha256"]:
        raise ValueError("verified encoder provenance receipt required")
    provenance = read_json(record["path"])
    if (provenance.get("schema") != "b4dl-sst-provenance-v1" or not provenance.get("verified")
            or provenance.get("provenance_id") != record["provenance_id"]
            or provenance["provenance_id"] != digest({k: v for k, v in provenance.items() if k != "provenance_id"})):
        raise ValueError("invalid encoder provenance")
    checkpoint = encoder_root / "ckpt_anneal/lidarclip_mm/last.ckpt"
    if sha256(checkpoint) != provenance["checkpoint"]["sha256"]:
        raise ValueError("regional extraction encoder checkpoint mismatch")
    names = ("lidarclip/model/sst.py", "lidarclip/model/sst_encoder_only_config.py",
             "lidarclip/model/attention_pool.py", "lidarclip/model/_mmdet3d_compat.py")
    for name in names:
        matches = [value for path, value in provenance["source_files_sha256"].items()
                   if path.replace("\\", "/").endswith("/" + name)]
        if not matches or len(set(matches)) != 1 or sha256(encoder_root / name) != matches[0]:
            raise ValueError("regional extraction source mismatch: " + name)
    return provenance, checkpoint


def extract(geometry_root, table_root, data_root, encoder_root, support_path, output, *, device="cuda:0", max_scenes=None):
    """Extract scenes into a new immutable directory and seal a complete manifest."""
    import torch
    geometry = GeometryDataset(geometry_root)
    provenance, checkpoint = _verify_sources(geometry, table_root, encoder_root)
    scope = geometry.manifest.get("input_scope", {})
    if sha256(support_path) != scope.get("support_sha256"):
        raise ValueError("regional extraction input-support hash mismatch")
    support = unique_index(read_jsonl(support_path), "sample_token")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("regional SST extraction requires CUDA")
    scene_ids = sorted(geometry.manifest["scenes"])
    if max_scenes is not None:
        if type(max_scenes) is not int or max_scenes < 1:
            raise ValueError("max_scenes must be positive")
        scene_ids = scene_ids[:max_scenes]
    selected_tokens = {token for ident in scene_ids for token in geometry.manifest["scenes"][ident]["frame_tokens"]}
    data = {row["token"]: row for row in iter_array(Path(table_root) / "sample_data.json")
            if row["token"] in selected_tokens}
    if set(data) != selected_tokens:
        raise ValueError("missing LiDAR sample_data for regional extraction")
    context = load_context(table_root, {row["sample_token"] for row in data.values()})
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new regional feature directory")
    output.mkdir(parents=True)
    (output / "scenes").mkdir()
    sys.path.insert(0, str(Path(encoder_root).resolve()))
    from lidarclip.model.sst import LidarEncoderSST
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = {key[len("lidar_encoder."):]: value for key, value in payload["state_dict"].items()
             if key.startswith("lidar_encoder.")}
    encoder = LidarEncoderSST(str(Path(encoder_root) / "lidarclip/model/sst_encoder_only_config.py"), 768)
    loaded = encoder.load_state_dict(state, strict=False)
    if loaded.missing_keys:
        raise ValueError("regional extraction encoder is incomplete")
    encoder.to(device).eval()
    records = {}
    try:
        for ident in scene_ids:
            scene_record = geometry.manifest["scenes"][ident]
            reference = geometry.scene(ident)["features"]
            offsets, indices, values, xyz, transforms, raw_hashes = [0], [], [], [], [], []
            max_error, max_cell_abs_error, max_cell_relative_l2 = 0., 0., 0.
            for frame, token in enumerate(scene_record["frame_tokens"]):
                sample_token = data[token]["sample_token"]
                sample = context["sample"][sample_token]
                if sample["data"]["LIDAR_TOP"] != token or data[token]["timestamp"] != scene_record["timestamps_us"][frame]:
                    raise ValueError("regional frame does not match geometry scene")
                points, _, receipt = retained_points(context, sample_token, data_root)
                retained = support.get(sample_token, {})
                if (receipt["lidar_sample_data_token"] != token or retained.get("scene_id") != ident
                        or retained.get("frame_index") != frame
                        or any(retained.get(key) != receipt[key] for key in
                               ("raw_lidar_sha256", "lidar_sample_data_token", "camera_sample_data_token",
                                "raw_points", "camera_crop_points", "retained_points"))):
                    raise ValueError("retained LiDAR input support mismatch")
                torch.manual_seed(0)
                with torch.no_grad():
                    bev = encoder._sst.extract_feat([torch.from_numpy(points).to(device)], None)[0]
                    global_feature, _ = encoder._pooler(bev)
                global_feature = global_feature[0].float().cpu().numpy()
                expected = reference[frame]
                norm = np.linalg.norm(global_feature) * np.linalg.norm(expected)
                cosine = float(np.dot(global_feature, expected) / norm) if norm else 0.
                error = float(np.max(np.abs(global_feature - expected)))
                if not np.allclose(global_feature, expected, atol=5e-4, rtol=1e-4) or cosine <= .999999:
                    raise ValueError(f"historical global feature mismatch in {ident} frame {frame}: {error}")
                bev_numpy = bev[0].float().cpu().numpy()
                cell_index, cell_value, cell_xyz = sparse_cells(bev_numpy, points)
                original_cells = bev_numpy.reshape(128, 6400).T[cell_index.astype(np.int64)]
                error_cells = original_cells - cell_value.astype(np.float32)
                max_cell_abs_error = max(max_cell_abs_error, float(np.max(np.abs(error_cells))))
                magnitude = float(np.linalg.norm(original_cells))
                if magnitude:
                    max_cell_relative_l2 = max(max_cell_relative_l2, float(np.linalg.norm(error_cells)) / magnitude)
                indices.append(cell_index); values.append(cell_value); xyz.append(cell_xyz)
                transforms.append(ego_from_sst(context, sample_token)); raw_hashes.append(receipt["raw_lidar_sha256"])
                offsets.append(offsets[-1] + len(cell_index))
                max_error = max(max_error, error)
            filename = f"scenes/{ident}.npz"
            path = safe_file(output, filename)
            with atomic_output(path, "wb") as stream:
                np.savez(stream, offsets=np.asarray(offsets, dtype=np.int64), indices=np.concatenate(indices),
                         values=np.concatenate(values), xyz=np.concatenate(xyz),
                         ego_from_sst=np.asarray(transforms, dtype=np.float64))
            records[ident] = {"file": filename, "sha256": sha256(path), "frames": len(reference),
                "frame_tokens": scene_record["frame_tokens"], "raw_lidar_sha256": raw_hashes,
                "source_feature_sha256": scene_record["sha256"], "max_global_abs_error": max_error,
                "max_cell_abs_error": max_cell_abs_error, "max_cell_relative_l2": max_cell_relative_l2}
            print(f"regional features {len(records)}/{len(scene_ids)} {ident}", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "extraction_failed", "type": type(error).__name__,
                   "message": str(error), "completed_scenes": sorted(records)})
        raise
    manifest = {"schema": SCHEMA, "geometry_dataset_id": geometry.manifest["dataset_id"],
        "geometry_manifest_sha256": sha256(Path(geometry_root) / "manifest.json"),
        "encoder_provenance_id": provenance["provenance_id"], "encoder_checkpoint_sha256": sha256(checkpoint),
        "support_sha256": sha256(support_path),
        "source_sha256": sha256(Path(__file__)), "complete": len(scene_ids) == len(geometry.manifest["scenes"]),
        "scene_count": len(records), "scenes": records,
        "global_tolerance": {"atol": 5e-4, "rtol": 1e-4, "min_cosine": .999999},
        "cell_values": "float16", "coordinate_system": "camera-z,-camera-x,-camera-y"}
    manifest["dataset_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


class RegionalGeometry:
    def __init__(self, geometry, root, *, cache_scenes=2):
        self.base, self.regional_root = geometry, Path(root)
        self.root = geometry.root
        self.manifest = geometry.manifest
        self.requests, self.targets = geometry.requests, geometry.targets
        self.regional_manifest = read_json(self.regional_root / "manifest.json")
        regional = self.regional_manifest
        if (regional.get("schema") not in (SCHEMA, "b4dl-regional-features-v2") or regional.get("dataset_id") != digest(
                {key: value for key, value in regional.items() if key != "dataset_id"})
                or regional.get("geometry_dataset_id") != self.manifest["dataset_id"]
                or regional.get("geometry_manifest_sha256") != sha256(geometry.root / "manifest.json")
                or regional.get("encoder_provenance_id") != self.manifest.get("encoder_provenance", {}).get("provenance_id")
                or regional.get("support_sha256") != self.manifest.get("input_scope", {}).get("support_sha256")
                or regional.get("scene_count") != len(regional.get("scenes", {}))):
            raise ValueError("regional feature manifest identity mismatch")
        if (not set(regional["scenes"]) <= set(self.manifest["scenes"])
                or type(regional.get("complete")) is not bool
                or regional["complete"] != (set(regional["scenes"]) == set(self.manifest["scenes"]))):
            raise ValueError("regional feature coverage does not match completeness")
        if regional["schema"] == "b4dl-regional-features-v2":
            preprocessing = regional.get("preprocessing", {})
            if (preprocessing.get("point_assignment") != "observed_voxel_layer_output"
                    or preprocessing.get("valid_cells") != "middle_encoder_voxel_coors_after_drop"
                    or preprocessing.get("centroid") != "mean_of_original_sst_xyz_of_all_assigned_points"):
                raise ValueError("unsupported regional v2 coordinate contract")
        if type(cache_scenes) is not int or cache_scenes < 0:
            raise ValueError("invalid regional scene cache size")
        self.cache_scenes, self._cache, self._verified = cache_scenes, OrderedDict(), {}

    def scene(self, ident):
        if ident not in self.regional_manifest["scenes"]:
            raise ValueError("regional features missing scene: " + ident)
        record, original = self.regional_manifest["scenes"][ident], self.manifest["scenes"][ident]
        is_v2 = self.regional_manifest["schema"] == "b4dl-regional-features-v2"
        if is_v2:
            frames = record["frames"]
            if (len(frames) != original["frames"]
                    or [frame["lidar"]["sample_data_token"] for frame in frames] != original["frame_tokens"]
                    or [frame["lidar"]["timestamp_us"] for frame in frames] != original["timestamps_us"]
                    or record["geometry_scene_sha256"] != original["sha256"]
                    or record["source_feature_sha256"] != original["source_feature_sha256"]):
                raise ValueError("regional v2 and geometry scene identities disagree")
        elif (record["frames"] != original["frames"] or record["frame_tokens"] != original["frame_tokens"]
              or record["source_feature_sha256"] != original["sha256"]):
            raise ValueError("regional and geometry scene identities disagree")
        path = safe_file(self.regional_root, record["file"])
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if self._verified.get(ident) != stamp:
            if sha256(path) != record["sha256"]:
                raise ValueError("regional scene hash mismatch")
            self._verified[ident] = stamp
            self._cache.pop(ident, None)
        if ident not in self._cache:
            if is_v2:
                from .regional_assets import validate_scene
                # The manifest was authenticated in __init__, and the scene
                # hash was checked above. Avoid reparsing the full manifest
                # for every randomly sampled training example.
                with np.load(path, allow_pickle=False) as source:
                    cells = {key: source[key] for key in source.files}
                validate_scene(cells, record["frames"])
                reference = self.base.scene(ident)
                if (not np.array_equal(cells["global_features"], reference["features"])
                        or not np.allclose(cells["global_from_lidar_ego"], reference["frame_geometry"], atol=1e-8, rtol=0)
                        or not np.allclose(cells["lidar_ego_from_lidar"], reference["ego_from_lidar"], atol=1e-8, rtol=0)):
                    raise ValueError("regional v2 global features or LiDAR transforms disagree with geometry")
                cells = {key: cells[key] for key in ("offsets", "indices", "values", "xyz", "ego_from_sst")}
            else:
                with np.load(path, allow_pickle=False) as source:
                    cells = {key: source[key] for key in source.files}
            if set(cells) != {"offsets", "indices", "values", "xyz", "ego_from_sst"}:
                raise ValueError("unknown regional scene fields")
            offsets, indices, values, xyz, transform = (cells[key] for key in
                ("offsets", "indices", "values", "xyz", "ego_from_sst"))
            frames = original["frames"]
            if (offsets.shape != (frames + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)
                    or int(offsets[-1]) != len(indices) or values.shape != (len(indices), 128)
                    or xyz.shape != (len(indices), 3) or transform.shape != (frames, 4, 4)
                    or not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer)
                    or (indices < 0).any() or (indices >= 6400).any()
                    or not np.isfinite(values).all() or not np.isfinite(xyz).all() or not np.isfinite(transform).all()):
                raise ValueError("invalid regional scene arrays")
            x = (indices.astype(np.int64) % 80) * .5
            y = (indices.astype(np.int64) // 80) * .5 - 20.
            if not is_v2 and (((xyz[:, 0] < x - 1e-4) | (xyz[:, 0] > x + .5001)
                 | (xyz[:, 1] < y - 1e-4) | (xyz[:, 1] > y + .5001)
                 | (xyz[:, 2] < -2.0001) | (xyz[:, 2] > 4.0001)).any()):
                raise ValueError("regional centroid outside its SST cell")
            sizes = np.diff(offsets)
            width = int(sizes.max())
            padded = {"cell_features": np.zeros((frames, width, 128), np.float32),
                      "cell_xyz": np.zeros((frames, width, 3), np.float32),
                      "cell_indices": np.zeros((frames, width), np.int64),
                      "cell_valid_mask": np.zeros((frames, width), bool), "ego_from_sst": transform}
            for frame, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                length = int(end - start)
                if len(np.unique(indices[start:end])) != length:
                    raise ValueError("duplicate regional indices")
                padded["cell_features"][frame, :length] = values[start:end]
                padded["cell_xyz"][frame, :length] = xyz[start:end]
                padded["cell_indices"][frame, :length] = indices[start:end]
                padded["cell_valid_mask"][frame, :length] = True
            self._cache[ident] = padded
            while len(self._cache) > max(self.cache_scenes, 1):
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(ident)
        result = self.base.scene(ident)
        result.update({key: value.copy() for key, value in self._cache[ident].items()})
        if self.cache_scenes == 0:
            self._cache.pop(ident)
        return result


class RegionalDataset:
    def __init__(self, dataset, root):
        self.base, self.geometry = dataset, RegionalGeometry(dataset.geometry, root)
        self.regional_manifest = self.geometry.regional_manifest
        self.requests, self.targets = dataset.requests, dataset.targets
        self.parent, self.exclusions = dataset.parent, dataset.exclusions
        self.manifest, self.root = dataset.manifest, dataset.root
        required = {request["scene_id"] for request in self.requests}
        if not required <= set(self.regional_manifest["scenes"]):
            raise ValueError("regional features do not cover requested split")

    def model_input(self, request, control=None):
        item = self.base.model_input(request, control)
        ident = request["scene_id"]
        condition = control["condition"] if control else "identity"
        source = control["donor_scene_id"] if condition == "cross_scene" else ident
        scene = self.geometry.scene(source)
        for key in ("cell_features", "cell_xyz", "cell_indices", "cell_valid_mask", "ego_from_sst"):
            value = scene[key]
            if condition == "repeat_first":
                value = np.repeat(value[:1], len(value), axis=0)
            elif condition == "shuffle_features":
                value = value[control["permutation"]].copy()
            elif condition == "zero" and key != "ego_from_sst":
                value = np.zeros_like(value)
            item[key] = value
        return item


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
        result = extract(args.geometry_dataset, args.tables, args.data_root, args.encoder_root, args.support, args.output,
                         device=args.device, max_scenes=args.max_scenes)
    print(f"regional dataset {result['dataset_id']} scenes={result['scene_count']} complete={result['complete']}")


if __name__ == "__main__":
    main()
