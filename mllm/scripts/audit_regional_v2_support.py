"""Audit out-of-range point support for annotation boxes in regional v2 scenes.

This diagnostic does not alter supervision or claim that every supported box
is a selected QA target. It considers only points assigned to surviving cells.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from b4dl_core.inputs import GeometryDataset
from b4dl_core.io import iter_array, read_jsonl, sha256, unique_index, write_json
from b4dl_core.regional_assets import load_scene
from b4dl_core.sst_source import box_support, load_context, retained_points


def surviving_outside_mask(points, kept_indices):
    """Reproduce the bound encoder's float32, clamped dynamic voxel IDs."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4 or points.dtype != np.float32 or not np.isfinite(points).all():
        raise ValueError("expected finite float32 SST points")
    xyz = points[:, :3]
    lower = np.array([0., -20., -2.], dtype=np.float32)
    upper = np.array([40., 20., 4.], dtype=np.float32)
    outside = ((xyz < lower) | (xyz >= upper)).any(axis=1)
    x = np.floor((xyz[:, 0] - lower[0]) / np.float32(.5)).astype(np.int64).clip(0, 79)
    y = np.floor((xyz[:, 1] - lower[1]) / np.float32(.5)).astype(np.int64).clip(0, 79)
    ids = y * 80 + x
    return outside & np.isin(ids, kept_indices), ids


def audit(geometry_root, regional_root, table_root, data_root, support_path, output, split="train"):
    geometry = GeometryDataset(geometry_root)
    regional_root, table_root, output = map(Path, (regional_root, table_root, output))
    if output.exists():
        raise FileExistsError("choose a new support audit output")
    from b4dl_core.io import read_json
    manifest = read_json(regional_root / "manifest.json")
    if (manifest.get("geometry_dataset_id") != geometry.manifest["dataset_id"]
            or manifest.get("geometry_manifest_sha256") != sha256(Path(geometry_root) / "manifest.json")
            or sha256(support_path) != manifest.get("support_sha256")
            or manifest.get("support_sha256") != geometry.manifest["input_scope"]["support_sha256"]):
        raise ValueError("regional support audit input identity mismatch")
    scenes = [ident for ident in sorted(manifest["scenes"]) if geometry.manifest["scenes"][ident]["split"] == split]
    if not scenes:
        raise ValueError("no regional scenes in requested split")
    sample_ids = {frame["sample_token"] for ident in scenes for frame in manifest["scenes"][ident]["frames"]}
    annotations = defaultdict(list)
    for row in iter_array(table_root / "sample_annotation.json"):
        if row["sample_token"] in sample_ids:
            annotations[row["sample_token"]].append(row)
    context = load_context(table_root, sample_ids)
    support = unique_index(read_jsonl(support_path), "sample_token")
    totals, scene_reports = Counter(), []
    for ident in scenes:
        arrays, frames, _ = load_scene(regional_root, ident)
        original = geometry.manifest["scenes"][ident]
        if ([frame["lidar"]["sample_data_token"] for frame in frames] != original["frame_tokens"]
                or manifest["scenes"][ident]["geometry_scene_sha256"] != original["sha256"]):
            raise ValueError("audit scene and parent geometry disagree")
        counts = Counter()
        for index, frame in enumerate(frames):
            sample = frame["sample_token"]
            points, inside_global, receipt = retained_points(context, sample, data_root)
            historical = support[sample]
            if (historical["scene_id"] != ident or historical["frame_index"] != index
                    or receipt["raw_lidar_sha256"] != frame["raw_lidar_sha256"]
                    or receipt["raw_lidar_sha256"] != historical["raw_lidar_sha256"]):
                raise ValueError("raw sample or support identity mismatch")
            start, end = arrays["offsets"][index:index + 2]
            ids = arrays["indices"][start:end].astype(np.int64)
            mask, point_ids = surviving_outside_mask(points, ids)
            actual_counts = np.bincount(point_ids[mask], minlength=6400)[ids]
            if not np.array_equal(actual_counts, arrays["out_of_range_counts"][start:end]):
                raise ValueError("recomputed outside cell assignments differ from captured v2")
            matrix = arrays["global_from_lidar_ego"][index] @ arrays["ego_from_sst"][index]
            outside_global = points[mask, :3].astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
            rows = annotations[sample]
            in_box = box_support(inside_global, rows)
            if in_box != historical["annotation_counts"]:
                raise ValueError("recomputed historical box support differs")
            outside_box = box_support(outside_global, rows)
            counts.update(frames=1, annotations=len(rows), outside_points=int(mask.sum()),
                inside_supported=sum(value > 0 for value in in_box.values()),
                outside_supported=sum(value > 0 for value in outside_box.values()),
                outside_only=sum(in_box[token] == 0 and count > 0 for token, count in outside_box.items()),
                both_supported=sum(in_box[token] > 0 and count > 0 for token, count in outside_box.items()))
        totals.update(counts)
        scene_reports.append({"scene_id": ident, **dict(counts)})
        print(f"support audit {len(scene_reports)}/{len(scenes)} {ident}", flush=True)
    report = {"schema": "b4dl-regional-v2-support-audit-v1", "geometry_dataset_id": geometry.manifest["dataset_id"],
              "regional_dataset_id": manifest["dataset_id"], "support_sha256": sha256(support_path),
              "split": split, "scenes": scene_reports, "totals": dict(totals),
              "scope": "annotation-box point support; not a QA target or label error count"}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry-dataset", "regional-features", "tables", "data-root", "support", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("train", "internal_dev", "test"), default="train")
    args = parser.parse_args()
    print(audit(args.geometry_dataset, args.regional_features, args.tables, args.data_root, args.support,
                args.output, args.split), flush=True)


if __name__ == "__main__":
    main()
