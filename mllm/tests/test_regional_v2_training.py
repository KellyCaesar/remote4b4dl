"""V2 regional training preserves observed coordinates and source identity."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from b4dl_core.io import digest, sha256, write_json
from b4dl_core.regional_data import CAMERA_FROM_SST, RegionalGeometry
from b4dl_core.regional_model import _validate_cells


class V2TrainingTests(unittest.TestCase):
    def fixture(self, root):
        geo = root / "geometry"
        geo.mkdir()
        write_json(geo / "manifest.json", {"fixture": True})
        reference = {"features": np.ones((1, 768), np.float32), "frame_times": np.array([0.]),
                     "frame_geometry": np.eye(4)[None], "ego_from_lidar": np.eye(4)[None]}
        original = {"frames": 1, "frame_tokens": ["lidar"], "timestamps_us": [100],
                    "sha256": "scene-hash", "source_feature_sha256": "feature-hash"}

        class Geometry:
            pass

        geometry = Geometry()
        geometry.root = geo
        geometry.manifest = {"dataset_id": "geometry-id", "scenes": {"scene": original},
                             "encoder_provenance": {"provenance_id": "encoder-id"},
                             "input_scope": {"support_sha256": "support-id"}}
        geometry.requests, geometry.targets = [], {}
        geometry.scene = lambda ident: {key: value.copy() for key, value in reference.items()}
        folder = root / "regional"
        folder.mkdir()
        arrays = {"global_features": reference["features"].copy(), "offsets": np.array([0, 1], np.int64),
                  "indices": np.array([0], np.int16), "values": np.zeros((1, 128), np.float16),
                  "xyz": np.array([[-.1, -20.1, 0.]], np.float32),
                  "point_counts": np.array([1], np.int32), "out_of_range_counts": np.array([1], np.int32),
                  "global_from_lidar_ego": np.eye(4)[None], "global_from_camera_ego": np.eye(4)[None],
                  "lidar_ego_from_lidar": np.eye(4)[None],
                  "camera_ego_from_camera": np.linalg.inv(CAMERA_FROM_SST)[None], "ego_from_sst": np.eye(4)[None]}
        frame = {"sample_token": "sample", "lidar": {"sample_data_token": "lidar", "timestamp_us": 100,
                 "ego_pose_token": "lidar-pose", "calibrated_sensor_token": "lidar-cal"},
                 "camera": {"sample_data_token": "camera", "timestamp_us": 100,
                 "ego_pose_token": "camera-pose", "calibrated_sensor_token": "camera-cal"},
                 "raw_lidar_sha256": "a" * 64, "camera_image_size": [1600, 900],
                 "raw_points": 1, "support_in_range_points": 0, "input_points": 1,
                 "rejected_points": 0, "input_out_of_range_points": 1,
                 "assigned_points_in_kept_cells": 1, "out_of_range_points_in_kept_cells": 1,
                 "input_occupied_cells": 1, "kept_cells": 1, "global_max_abs_error": 0.,
                 "global_cosine": 1., "max_cell_abs_error": 0., "cell_relative_l2": 0.}
        np.savez(folder / "scene.npz", **arrays)
        manifest = {"schema": "b4dl-regional-features-v2", "geometry_dataset_id": "geometry-id",
                    "geometry_manifest_sha256": sha256(geo / "manifest.json"), "encoder_provenance_id": "encoder-id",
                    "support_sha256": "support-id", "complete": True, "scene_count": 1,
                    "preprocessing": {"point_assignment": "observed_voxel_layer_output",
                        "valid_cells": "middle_encoder_voxel_coors_after_drop",
                        "centroid": "mean_of_original_sst_xyz_of_all_assigned_points"},
                    "scenes": {"scene": {"file": "scene.npz", "sha256": sha256(folder / "scene.npz"),
                        "frames": [frame], "geometry_scene_sha256": "scene-hash", "source_feature_sha256": "feature-hash"}}}
        manifest["dataset_id"] = digest(manifest)
        write_json(folder / "manifest.json", manifest)
        return geometry, folder, arrays, manifest

    def test_real_assignment_is_preserved_and_strict_mode_still_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry, folder, arrays, _ = self.fixture(Path(directory))
            scene = RegionalGeometry(geometry, folder).scene("scene")
            np.testing.assert_array_equal(scene["cell_xyz"][0, 0], arrays["xyz"][0])
            self.assertTrue(scene["cell_valid_mask"][0, 0])  # zero-valued cell remains valid
            args = (torch.tensor(scene["cell_features"])[None], torch.tensor(scene["cell_xyz"])[None],
                    torch.tensor(scene["cell_indices"])[None], torch.tensor(scene["cell_valid_mask"])[None],
                    torch.ones((1, 1), dtype=torch.bool), torch.tensor(scene["ego_from_sst"])[None], 128)
            with self.assertRaisesRegex(ValueError, "outside"):
                _validate_cells(*args)
            _validate_cells(*args, centroid_policy="observed_assignment")

    def test_rejects_parent_frame_or_feature_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry, folder, arrays, manifest = self.fixture(Path(directory))
            geometry.manifest["scenes"]["scene"]["frame_tokens"] = ["other"]
            with self.assertRaisesRegex(ValueError, "identities disagree"):
                RegionalGeometry(geometry, folder).scene("scene")
            geometry.manifest["scenes"]["scene"]["frame_tokens"] = ["lidar"]
            arrays["global_features"][0, 0] = 5
            np.savez(folder / "scene.npz", **arrays)
            manifest["scenes"]["scene"]["sha256"] = sha256(folder / "scene.npz")
            manifest["dataset_id"] = digest({k: v for k, v in manifest.items() if k != "dataset_id"})
            write_json(folder / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "global features"):
                RegionalGeometry(geometry, folder).scene("scene")


if __name__ == "__main__":
    unittest.main()
