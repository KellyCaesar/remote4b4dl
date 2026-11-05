"""Regression checks for regional inputs, projection and B3 integration."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from b4dl_core.integration import attach_regional, language_model
from b4dl_core.io import digest, sha256, write_json
from b4dl_core.regional_data import RegionalDataset, ego_from_sst, sparse_cells
from b4dl_core.regional_model import RegionalConfig, RegionalProjector
from b4dl_core.reasoning.data import ReasoningDataset, build_data
from b4dl_core.reasoning.policy import extension_state, load_checkpoint, save_checkpoint, sequence_batch
from b4dl_core.data import build_dataset
from test_reasoning_data import raw_fixture
from test_reasoning_training import CharacterTokenizer, item
from vtimellm.model.vtimellm_llama import VTimeLLMConfig, VTimeLLMLlamaForCausalLM


def regional_item():
    row = item()
    indices = np.tile(np.array([0, 40, 3200, 3240], dtype=np.int64), (3, 1))
    xyz = np.tile(np.array([[[.25, -19.75, 0.], [20.25, -19.75, 0.],
                             [.25, .25, 0.], [20.25, .25, 0.]]], np.float32), (3, 1, 1))
    row.update(cell_features=np.ones((3, 4, 128), np.float32),
        cell_xyz=xyz, cell_indices=indices,
        cell_valid_mask=np.ones((3, 4), dtype=bool),
        ego_from_sst=np.tile(np.eye(4, dtype=np.float64), (3, 1, 1)))
    return row


def regional_model(tokens_per_frame=1):
    torch.manual_seed(5)
    config = VTimeLLMConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=2, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    raw = VTimeLLMLlamaForCausalLM(config)
    raw.get_model().mm_projector = torch.nn.Linear(768, 32)
    regional = RegionalConfig(layers=1, ff_width=32, tokens_per_frame=tokens_per_frame)
    model, options = attach_regional(raw, regional, lora_rank=4)
    return model, options


def write_regional_fixture(geometry, output):
    output.mkdir()
    records = {}
    for scene_id, source in geometry.manifest["scenes"].items():
        frames = source["frames"]
        path = output / (scene_id + ".npz")
        np.savez(path, offsets=np.arange(frames + 1, dtype=np.int64),
                 indices=np.zeros(frames, dtype=np.int16), values=np.ones((frames, 128), dtype=np.float16),
                 xyz=np.tile(np.array([[.1, -19.9, 0]], np.float32), (frames, 1)),
                 ego_from_sst=np.tile(np.eye(4), (frames, 1, 1)))
        records[scene_id] = {"file": path.name, "sha256": sha256(path), "frames": frames,
                             "frame_tokens": source["frame_tokens"], "source_feature_sha256": source["sha256"]}
    manifest = {"schema": "b4dl-regional-features-v1", "geometry_dataset_id": geometry.manifest["dataset_id"],
        "geometry_manifest_sha256": sha256(geometry.root / "manifest.json"),
        "encoder_provenance_id": geometry.manifest.get("encoder_provenance", {}).get("provenance_id"),
        "support_sha256": geometry.manifest.get("input_scope", {}).get("support_sha256"),
        "complete": True, "scene_count": len(records), "scenes": records}
    manifest["dataset_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)


class RegionalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_sparse_cells_follow_y_x_layout_and_input_centroids(self):
        bev = np.zeros((128, 80, 80), np.float32)
        bev[:, 2, 3] = 2
        bev[:, 0, 0] = 1
        points = np.array([[.1, -19.9, 0., 1.], [1.6, -18.9, 1., 1.],
                           [1.7, -18.8, 3., 1.], [100., 0., 0., 1.]], np.float32)
        indices, values, xyz = sparse_cells(bev, points)
        np.testing.assert_array_equal(indices, [0, 163])
        np.testing.assert_allclose(xyz, [[.1, -19.9, 0], [1.65, -18.85, 2]], atol=1e-5)
        np.testing.assert_array_equal(values[:, 0], [1, 2])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            sparse_cells(bev, points[:1])

    def test_camera_and_lidar_ego_time_are_both_used(self):
        pose = lambda x: {"rotation": [1., 0., 0., 0.], "translation": [x, 0., 0.]}
        context = {"sample": {"s": {"data": {"LIDAR_TOP": "l", "CAM_BACK_RIGHT": "c"}}},
            "sample_data": {"l": {"ego_pose_token": "pl"},
                            "c": {"ego_pose_token": "pc", "calibrated_sensor_token": "cc"}},
            "ego_pose": {"pl": pose(10.), "pc": pose(12.)}, "calibrated_sensor": {"cc": pose(3.)}}
        transform = ego_from_sst(context, "s")
        np.testing.assert_allclose(transform @ [2., 1., -1., 1.], [4., 1., 2., 1.])
        np.testing.assert_allclose(np.linalg.inv(transform) @ transform, np.eye(4), atol=1e-12)

    def test_projection_masks_padding_and_uses_geometry(self):
        config = RegionalConfig(input_width=4, width=8, heads=2, layers=1, ff_width=16, tokens_per_frame=5)
        net, projector = RegionalProjector(config, 12), torch.nn.Linear(8, 12)
        features = torch.randn(1, 3, 8)
        times = torch.tensor([[0., .5, 0.]], dtype=torch.float64)
        poses = torch.eye(4, dtype=torch.float64).repeat(1, 3, 1, 1)
        valid = torch.tensor([[True, True, False]])
        cells = torch.randn(1, 3, 4, 4)
        xyz = torch.tensor([[[[.25, -19.75, 0.], [20.25, -19.75, 0.],
                              [.25, .25, 0.], [20.25, .25, 0.]]]]).repeat(1, 3, 1, 1)
        indices = torch.tensor([0, 40, 3200, 3240]).repeat(1, 3, 1)
        mask = valid.unsqueeze(-1).expand(-1, -1, 4).clone()
        transform = torch.eye(4, dtype=torch.float64).repeat(1, 3, 1, 1)
        cells[:, 2] = float("nan")
        xyz[:, 2] = float("nan")
        features[:, 2] = float("nan")
        times[:, 2] = float("nan")
        poses[:, 2] = float("nan")
        transform[:, 2] = float("nan")
        out, visual_mask = net(features, times, poses, valid, cells, xyz, indices, mask, transform, projector)
        torch.testing.assert_close(out[:, :, 0][valid], projector(features[valid]), rtol=0, atol=0)
        self.assertEqual(visual_mask.sum().item(), 10)
        out[visual_mask].square().mean().backward()
        for name, parameter in net.named_parameters():
            self.assertTrue(parameter.grad is None or torch.isfinite(parameter.grad).all(), name)
        duplicate_xyz = torch.tensor([.25, -19.75, 0.]).reshape(1, 1, 1, 3).expand_as(xyz)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            net(features, times, poses, valid, cells, duplicate_xyz, torch.zeros_like(indices), mask, transform, projector)

    def test_tiny_language_model_forward_generation_and_checkpoint(self):
        tokenizer = CharacterTokenizer()
        for count in (1, 5):
            with self.subTest(tokens=count), tempfile.TemporaryDirectory() as directory:
                model, options = regional_model(count)
                row = regional_item()
                batch = sequence_batch(model, tokenizer, [row], [[100, 101, 2]])
                result = model(**batch, use_cache=False)
                self.assertTrue(torch.isfinite(result.loss))
                result.loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if ".regional_adapter.residual." in n))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                generated = model.generate(**sequence_batch(model, tokenizer, [row]), max_new_tokens=2,
                                           do_sample=False, pad_token_id=2)
                self.assertEqual(generated.shape[0], 1)
                state = extension_state(model)
                path = Path(directory) / "checkpoint.pt"
                save_checkpoint(path, model, {"fixture": 1}, "sft", {"fixture": True}, 1)
                restored, restored_options = regional_model(count)
                self.assertEqual(options, restored_options)
                load_checkpoint(path, restored, {"fixture": 1}, "sft")
                for name, value in extension_state(restored).items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                language_model(model).config.tokenizer_model_max_length = 10
                with self.assertRaisesRegex(ValueError, "context exceeds"):
                    sequence_batch(model, tokenizer, [row], [[100, 101, 2]])

    def test_regional_geometry_identity_and_input_interventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables, features, support = raw_fixture(root)
            build_dataset(tables, root / "metadata.json", features, root / "splits.json", root / "geometry", input_support=support)
            build_data(root / "geometry", tables, support, root / "reasoning")
            dataset = ReasoningDataset(root / "reasoning", root / "geometry", "train")
            write_regional_fixture(dataset.geometry, root / "regional")
            regional = RegionalDataset(dataset, root / "regional")
            request = regional.requests[0]
            normal = regional.model_input(request)
            self.assertEqual(normal["cell_features"].shape[-1], 128)
            self.assertTrue(normal["cell_valid_mask"].all())
            control = {"dataset_id": dataset.geometry.manifest["dataset_id"],
                "request_id": request["request_id"], "input_fingerprint": dataset.parent[request["request_id"]]["input_fingerprint"],
                "condition": "zero", "status": "ready"}
            zero = regional.model_input(request, control)
            self.assertFalse(zero["cell_valid_mask"].any())
            self.assertFalse(zero["features"].any())
            np.testing.assert_array_equal(zero["frame_geometry"], normal["frame_geometry"])
            scene = regional.geometry.regional_manifest["scenes"][request["scene_id"]]
            path = root / "regional" / scene["file"]
            path.write_bytes(path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                regional.geometry.scene(request["scene_id"])


if __name__ == "__main__":
    unittest.main()
