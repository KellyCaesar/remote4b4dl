"""Independent v2 asset, coordinate-chain, and SST capture checks."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
from typing import Tuple
import unittest
from unittest.mock import patch

import numpy as np
import torch

from b4dl_core.io import digest, sha256, write_json, write_jsonl
from b4dl_core.regional_assets import (SCHEMA, capture_grid, cells_from_grid, check_encoder_config,
                                       extract, frame_metadata, load_scene, validate_scene, verify_grid_sources)


def context_fixture():
    root_half = 2 ** -.5
    return {
        "sample": {"s": {"data": {"LIDAR_TOP": "l", "CAM_BACK_RIGHT": "c"}}},
        "sample_data": {
            "l": {"token": "l", "sample_token": "s", "timestamp": 100,
                  "ego_pose_token": "pl", "calibrated_sensor_token": "lc"},
            "c": {"token": "c", "sample_token": "s", "timestamp": 160,
                  "ego_pose_token": "pc", "calibrated_sensor_token": "cc"}},
        "ego_pose": {
            "pl": {"timestamp": 100, "translation": [10., 0., 0.], "rotation": [root_half, 0., 0., root_half]},
            "pc": {"timestamp": 160, "translation": [8., 1., 0.], "rotation": [0., 0., 0., 1.]}},
        "calibrated_sensor": {
            "lc": {"translation": [1., 0., 0.], "rotation": [root_half, 0., 0., root_half]},
            "cc": {"translation": [0., 1., 0.], "rotation": [root_half, 0., 0., root_half]}}}


def frame_arrays():
    meta, matrices = frame_metadata(context_fixture(), "s")
    meta.update(raw_lidar_sha256="a" * 64, camera_image_size=[1600, 900], raw_points=3,
                support_in_range_points=1, input_points=2, rejected_points=0, input_out_of_range_points=1,
                assigned_points_in_kept_cells=2, out_of_range_points_in_kept_cells=1,
                input_occupied_cells=1, kept_cells=1, global_max_abs_error=0., global_cosine=1.,
                max_cell_abs_error=0., cell_relative_l2=0.)
    arrays = {key: value[None].astype(np.float64) for key, value in matrices.items()}
    arrays.update(global_features=np.ones((1, 768), np.float32), offsets=np.array([0, 1], np.int64),
                  indices=np.array([0], np.int16), values=np.zeros((1, 128), np.float16),
                  xyz=np.array([[-.05, -20.45, 2.5]], np.float32),
                  point_counts=np.array([2], np.int32), out_of_range_counts=np.array([1], np.int32))
    return arrays, [meta]


class RegionalAssetsTests(unittest.TestCase):
    def test_current_sst_configuration_matches_contract(self):
        root = Path(__file__).resolve().parents[2]
        check_encoder_config(root / "encoders/lidarclip/lidarclip/model/sst_encoder_only_config.py")

    def test_grid_sources_must_match_the_encoder_provenance(self):
        encoder_root = Path(__file__).resolve().parents[2] / "encoders/lidarclip"
        names = ("sst/mmdet3d/ops/voxel/voxelize.py",
                 "sst/mmdet3d/models/middle_encoders/sst_input_layer_v2.py",
                 "sst/mmdet3d/models/backbones/sst_v2.py")
        provenance = {"source_files_sha256": {"/original/encoder/" + name: sha256(encoder_root / name)
                                              for name in names}}
        self.assertEqual(len(verify_grid_sources(provenance, encoder_root)), 3)
        provenance["source_files_sha256"]["/original/encoder/" + names[1]] = "changed"
        with self.assertRaisesRegex(ValueError, "grid source mismatch"):
            verify_grid_sources(provenance, encoder_root)

    def test_camera_axes_and_sensor_time_chain_with_rotation(self):
        meta, matrices = frame_metadata(context_fixture(), "s")
        self.assertEqual(meta["lidar"]["timestamp_us"], 100)
        self.assertEqual(meta["camera"]["timestamp_us"], 160)
        # Independent hand-computed path: lidar [1,0,2] -> lidar ego [1,1,2]
        # -> global [9,1,2] -> camera ego [-1,0,2] -> camera [-1,1,2]
        # -> SST [2,1,-1].  Both ego poses and both calibrations rotate.
        lidar = np.array([1., 0., 2., 1.])
        camera = np.array([-1., 1., 2., 1.])
        sst = np.array([2., 1., -1., 1.])
        ego = matrices["lidar_ego_from_lidar"] @ lidar
        np.testing.assert_allclose(ego, [1., 1., 2., 1.], atol=1e-12)
        np.testing.assert_allclose(matrices["global_from_lidar_ego"] @ ego, [9., 1., 2., 1.], atol=1e-12)
        camera_ego = matrices["camera_ego_from_camera"] @ camera
        np.testing.assert_allclose(camera_ego, [-1., 0., 2., 1.], atol=1e-12)
        np.testing.assert_allclose(matrices["global_from_camera_ego"] @ camera_ego, [9., 1., 2., 1.], atol=1e-12)
        np.testing.assert_allclose(matrices["ego_from_sst"] @ sst, ego, atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(matrices["ego_from_sst"]) @ ego, sst, atol=1e-12)
        self.assertFalse(np.allclose(matrices["lidar_ego_from_lidar"] @ sst, ego))
        context = context_fixture()
        context["ego_pose"]["pc"]["timestamp"] = 100
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            frame_metadata(context, "s")
        context = context_fixture()
        context["sample_data"]["c"]["sample_token"] = "other"
        with self.assertRaisesRegex(ValueError, "sample identity"):
            frame_metadata(context, "s")

    def test_cells_follow_actual_assignment_and_surviving_coordinates(self):
        points = np.array([[.1, -19.9, 0., 1.], [-.2, -21., 5., 1.],
                           [1.6, -18.9, 1., 1.], [1.7, -18.8, 3., 1.],
                           [100., 0., 0., 1.]], np.float32)
        # The second point is clamped into cell 0 by the existing shim; the
        # last point is assigned but its cell is dropped by the middle layer.
        point_coors = np.array([[0, 0, 0], [0, 0, 0], [0, 2, 3], [0, 2, 3], [0, 40, 79]], np.int32)
        kept = np.array([[0, 0, 2, 3], [0, 0, 0, 0]], np.int64)
        bev = np.zeros((128, 80, 80), np.float32)
        bev[:, 2, 3] = 2
        cells, stats = cells_from_grid(bev, points, point_coors, kept)
        np.testing.assert_array_equal(cells["indices"], [0, 163])
        np.testing.assert_array_equal(cells["point_counts"], [2, 2])
        np.testing.assert_array_equal(cells["out_of_range_counts"], [1, 0])
        np.testing.assert_allclose(cells["xyz"], [[-.05, -20.45, 2.5], [1.65, -18.85, 2.]], atol=1e-5)
        self.assertTrue(np.all(cells["values"][0] == 0))  # Zero is still a valid output cell.
        self.assertEqual(stats["input_occupied_cells"], 3)
        self.assertEqual(stats["assigned_points_in_kept_cells"], 4)
        self.assertEqual(stats["input_out_of_range_points"], 2)
        bad = kept.copy()
        bad[0] = [0, 0, 1, 1]
        with self.assertRaisesRegex(ValueError, "no assigned"):
            cells_from_grid(bev, points, point_coors, bad)
        bev[:, 1, 1] = 3
        with self.assertRaisesRegex(ValueError, "outside the observed"):
            cells_from_grid(bev, points, point_coors, kept)

    def test_checked_in_voxel_shim_assigns_outside_points_to_boundary(self):
        # Load the actual pure-torch function without importing the shim's
        # optional mmcv/mmdet stack in the CPU-only test environment.
        root = Path(__file__).resolve().parents[2]
        path = root / "encoders/lidarclip/lidarclip/model/_mmdet3d_compat.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_dynamic_voxelize_py")
        scope = {"torch": torch, "Tuple": Tuple}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        points = torch.tensor([[.1, -19.9, 0., 1.], [-.2, -21., 5., 1.],
                               [40., 20., 4., 1.]], dtype=torch.float32)
        coors = torch.zeros((len(points), 3), dtype=torch.int32)
        scope["_dynamic_voxelize_py"](points, coors, (.5, .5, 6.), (0, -20, -2, 40, 20, 4))
        np.testing.assert_array_equal(coors.numpy(), [[0, 0, 0], [0, 0, 0], [0, 79, 79]])
        bev = np.zeros((128, 80, 80), dtype=np.float32)
        kept = np.array([[0, 0, 0, 0], [0, 0, 79, 79]], dtype=np.int64)
        cells, stats = cells_from_grid(bev, points.numpy(), coors.numpy(), kept)
        np.testing.assert_array_equal(cells["out_of_range_counts"], [1, 1])
        self.assertEqual(stats["input_out_of_range_points"], 2)

    def test_capture_hooks_preserve_forward_and_are_removed(self):
        class Voxel(torch.nn.Module):
            def forward(self, value):
                return value.to(torch.int32)

        class Middle(torch.nn.Module):
            def forward(self, value):
                return {"voxel_coors": value.to(torch.int64), "feature": value + 1}

        class FakeSST(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.voxel_layer, self.middle_encoder = Voxel(), Middle()

            def forward(self, value):
                return self.middle_encoder(self.voxel_layer(value))["feature"]

        model = FakeSST()
        value = torch.tensor([[0, 1, 2, 3]])
        expected = model(value)
        with capture_grid(model) as observed:
            actual = model(value)
        torch.testing.assert_close(actual, expected)
        np.testing.assert_array_equal(observed["point_coors"][0], value.numpy())
        np.testing.assert_array_equal(observed["kept_coors"][0], value.numpy())
        self.assertFalse(model.voxel_layer._forward_hooks)
        self.assertFalse(model.middle_encoder._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with capture_grid(model):
                raise RuntimeError("fixture failure")
        self.assertFalse(model.voxel_layer._forward_hooks)
        self.assertFalse(model.middle_encoder._forward_hooks)

    def test_v2_scene_roundtrip_and_integrity(self):
        arrays, frames = frame_arrays()
        validate_scene(arrays, frames)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "scene.npz", **arrays)
            manifest = {"schema": SCHEMA, "scene_count": 1,
                        "scenes": {"scene": {"file": "scene.npz", "sha256": sha256(root / "scene.npz"),
                                             "frames": frames}}}
            manifest["dataset_id"] = digest(manifest)
            write_json(root / "manifest.json", manifest)
            loaded, loaded_frames, _ = load_scene(root, "scene")
            np.testing.assert_array_equal(loaded["xyz"], arrays["xyz"])
            self.assertEqual(loaded_frames, frames)
            (root / "scene.npz").write_bytes((root / "scene.npz").read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_scene(root, "scene")
        arrays["ego_from_sst"][0, 0, 3] += 1
        with self.assertRaisesRegex(ValueError, "chain mismatch"):
            validate_scene(arrays, frames)

    def test_extraction_writes_loadable_scene_with_frame_provenance(self):
        context = context_fixture()
        _, transforms = frame_metadata(context, "s")
        points = np.array([[.1, -19.9, 0., 1.], [-.2, -21., 5., 1.]], np.float32)
        receipt = {"raw_lidar_sha256": "a" * 64, "raw_points": 3, "camera_crop_points": 2,
                   "retained_points": 1, "lidar_sample_data_token": "l", "camera_sample_data_token": "c",
                   "camera_image_size": [1600, 900]}

        class FakeVoxel(torch.nn.Module):
            def forward(self, values):
                return torch.zeros((len(values), 3), dtype=torch.int32)

        class FakeMiddle(torch.nn.Module):
            def forward(self, _coors):
                return {"voxel_coors": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)}

        class FakeSST:
            def __init__(self):
                self.voxel_layer, self.middle_encoder = FakeVoxel(), FakeMiddle()

            def extract_feat(self, tensors, _metas):
                self.middle_encoder(self.voxel_layer(tensors[0]))
                return [torch.zeros((1, 128, 80, 80), dtype=torch.float32)]

        class FakeEncoder:
            def __init__(self, *_args):
                self._sst = FakeSST()

            def load_state_dict(self, *_args, **_kwargs):
                return SimpleNamespace(missing_keys=[])

            def to(self, _device):
                return self

            def eval(self):
                return self

            def _pooler(self, _bev):
                return torch.ones((1, 768), dtype=torch.float32), None

        class DeviceInput:
            def __init__(self, tensor):
                self.tensor = tensor

            def to(self, _device):
                return self.tensor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, geometry_root, encoder_root = (root / name for name in ("tables", "geometry", "encoder"))
            for path in (tables, geometry_root, encoder_root):
                path.mkdir()
            write_json(tables / "sample_data.json", [context["sample_data"]["l"]])
            write_json(geometry_root / "manifest.json", {"fixture": True})
            support = root / "support.jsonl"
            write_jsonl(support, [{"sample_token": "s", "scene_id": "scene", "frame_index": 0, **receipt}])
            checkpoint = root / "encoder.ckpt"
            checkpoint.write_bytes(b"fixture")
            scene = {"frame_tokens": ["l"], "timestamps_us": [100], "sha256": "geometry-hash",
                     "source_feature_sha256": "original-feature-hash"}
            geometry = SimpleNamespace(manifest={"dataset_id": "geometry-id", "scenes": {"scene": scene},
                                                "input_scope": {"support_sha256": sha256(support)}},
                                       scene=lambda _ident: {"features": np.ones((1, 768), np.float32),
                                           "frame_geometry": transforms["global_from_lidar_ego"][None],
                                           "ego_from_lidar": transforms["lidar_ego_from_lidar"][None]})
            fake_module = ModuleType("lidarclip.model.sst")
            fake_module.LidarEncoderSST = FakeEncoder
            original_from_numpy = torch.from_numpy
            with (patch("b4dl_core.regional_assets.GeometryDataset", return_value=geometry),
                  patch("b4dl_core.regional_assets._verify_sources", return_value=({"provenance_id": "p"}, checkpoint)),
                  patch("b4dl_core.regional_assets.check_encoder_config"),
                  patch("b4dl_core.regional_assets.verify_grid_sources", return_value={}),
                  patch("b4dl_core.regional_assets.verify_runtime_grid"),
                  patch("b4dl_core.regional_assets.load_context", return_value=context),
                  patch("b4dl_core.regional_assets.retained_points", return_value=(points, None, receipt)),
                  patch.object(torch.cuda, "is_available", return_value=True),
                  patch.object(torch.cuda, "get_device_name", return_value="synthetic CPU fixture"),
                  patch.object(torch, "load", return_value={"state_dict": {}}),
                  patch.object(torch, "from_numpy", side_effect=lambda x: DeviceInput(original_from_numpy(x))),
                  patch.dict("sys.modules", {"lidarclip.model.sst": fake_module})):
                manifest = extract(geometry_root, tables, root, encoder_root, support, root / "asset")
            self.assertTrue(manifest["complete"])
            arrays, frames, loaded_manifest = load_scene(root / "asset", "scene")
            self.assertEqual(loaded_manifest["geometry_dataset_id"], "geometry-id")
            self.assertEqual(frames[0]["sample_token"], "s")
            self.assertEqual(frames[0]["camera"]["timestamp_us"], 160)
            self.assertEqual(frames[0]["out_of_range_points_in_kept_cells"], 1)
            np.testing.assert_allclose(arrays["xyz"], [[-.05, -20.45, 2.5]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
