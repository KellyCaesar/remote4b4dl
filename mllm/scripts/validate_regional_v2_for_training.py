"""Validate every complete v2 scene against its parent before formal training."""
import argparse
from pathlib import Path

from b4dl_core.inputs import GeometryDataset
from b4dl_core.io import sha256, write_json
from b4dl_core.regional_data import RegionalGeometry


def validate(geometry, regional_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new validation receipt")
    regional = RegionalGeometry(geometry, regional_root)
    manifest = regional.regional_manifest
    if manifest["schema"] != "b4dl-regional-features-v2" or not manifest["complete"]:
        raise ValueError("formal training requires complete regional v2 assets")
    total_frames, total_cells = 0, 0
    for index, ident in enumerate(sorted(geometry.manifest["scenes"])):
        scene = regional.scene(ident)
        total_frames += len(scene["features"])
        total_cells += int(scene["cell_valid_mask"].sum())
        if (index + 1) % 25 == 0:
            print(f"validated regional scenes {index + 1}/{len(geometry.manifest['scenes'])}", flush=True)
    receipt = {"schema": "b4dl-regional-v2-full-validation-v1", "status": "validated",
               "geometry_dataset_id": geometry.manifest["dataset_id"], "regional_dataset_id": manifest["dataset_id"],
               "regional_manifest_sha256": sha256(Path(regional_root) / "manifest.json"),
               "scene_count": len(geometry.manifest["scenes"]), "frame_count": total_frames,
               "valid_cell_count": total_cells}
    write_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dataset", required=True)
    parser.add_argument("--regional-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(validate(GeometryDataset(args.geometry_dataset), args.regional_features, args.output), flush=True)


if __name__ == "__main__":
    main()
