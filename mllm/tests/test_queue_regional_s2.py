"""Check the S1 completion and S2 matching gates without launching training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from b4dl_core.io import digest, sha256, write_json, write_jsonl
from b4dl_core.regional_model import RegionalConfig
from scripts.queue_regional_s2 import run, train_command, validate_s1_binding, verify_training


class QueueRegionalS2Tests(unittest.TestCase):
    def fixture(self, root):
        output = root / "run"
        output.mkdir()
        identity = {"stage": "sft", "recipe": {"seed": 42, "engineering": False},
                    "complete_budget": True, "common": {"implementation": "source"}}
        payload = {"schema": "b4dl-reasoning-checkpoint-v1", "stage": "sft", "step": 2, "identity": identity,
                   "parameters": {"p": torch.tensor([1.])},
                   "optimizer": {"state": {0: {"step": torch.tensor(2.), "exp_avg": torch.tensor([.5])}}}}
        torch.save(payload, output / "step-2.pt")
        checkpoint = sha256(output / "step-2.pt")
        write_json(output / "latest.json", {"step": 2, "checkpoint": "step-2.pt", "sha256": checkpoint})
        write_json(output / "run.json", {"identity": identity, "requested_updates": 2, "total_updates": 2})
        write_json(output / "completion.json", {"status": "training_complete", "updates": 2,
                   "identity": identity, "full_budget": True, "checkpoint_sha256": checkpoint})
        write_jsonl(output / "training.jsonl", [{"step": 1, "loss": 1.}, {"step": 2, "loss": .5}])
        return output, identity, payload

    def test_verified_endpoint_and_rejects_incomplete_or_tampered(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, _ = self.fixture(Path(directory))
            result = verify_training(output, "training_complete", 2)
            self.assertEqual(result["updates"], 2)
            (output / "training.jsonl").write_text(json.dumps({"step": 1, "loss": 1.}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "log count"):
                verify_training(output, "training_complete", 2)
            (output / "step-2.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_training(output, "training_complete", 2)

    def test_rejects_nonfinite_optimizer_and_incorrect_step(self):
        for key, value in (("exp_avg", torch.tensor([float("nan")])), ("step", torch.tensor(1.))):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                output, _, payload = self.fixture(Path(directory))
                payload["optimizer"]["state"][0][key] = value
                torch.save(payload, output / "step-2.pt")
                checksum = sha256(output / "step-2.pt")
                latest = json.loads((output / "latest.json").read_text())
                completion = json.loads((output / "completion.json").read_text())
                latest["sha256"] = completion["checkpoint_sha256"] = checksum
                write_json(output / "latest.json", latest)
                write_json(output / "completion.json", completion)
                with self.assertRaises(ValueError):
                    verify_training(output, "training_complete", 2)

    def test_s2_configuration_changes_only_temporal_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, prep = root / "baseline", root / "prep"
            baseline.write_bytes(b"base")
            prep.write_bytes(b"prep")
            s1config = RegionalConfig(use_temporal=False, use_geometry=False,
                                      centroid_policy="observed_assignment").as_dict()
            config = RegionalConfig(use_temporal=True, use_geometry=False, centroid_policy="observed_assignment")
            common = {"options": {"regional": s1config}, "implementation": "source",
                      "baseline_sha256": sha256(baseline), "preparation_sha256": sha256(prep),
                      "regional_dataset_id": "asset"}
            s1 = {"full_budget": True, "identity": {"stage": "sft", "recipe": {"seed": 42}, "common": common}}
            manifest = {"dataset_id": "asset", "geometry_dataset_id": "geometry", "complete": True}
            validation = {"status": "validated", "regional_dataset_id": "asset", "geometry_dataset_id": "geometry"}
            validate_s1_binding(s1, config, {"seed": 42}, baseline, prep, manifest, validation, "source")
            with self.assertRaisesRegex(ValueError, "only by temporal"):
                validate_s1_binding(s1, RegionalConfig(use_temporal=True, use_geometry=True),
                                    {"seed": 42}, baseline, prep, manifest, validation, "source")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_s1_binding(s1, config, {"seed": 43}, baseline, prep, manifest, validation, "source")

    def test_training_starts_from_baseline_without_resume(self):
        args = SimpleNamespace(dataset="data", geometry="geo", baseline="baseline", preparation="prep", recipe="recipe",
            gpu_lock="lock", small_regional="small", regional="full", s2_config="s2.json")
        smoke = train_command(args, "smoke", engineering=True)
        formal = train_command(args, "formal")
        self.assertIn("--engineering-region-subset", smoke)
        self.assertNotIn("--engineering", formal)
        self.assertNotIn("--resume", smoke + formal)
        self.assertEqual(formal[formal.index("--regional-features") + 1], "full")

    def test_failed_upstream_never_launches_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "s1-pipeline.pid").write_text("123", encoding="utf-8")
            (root / "s1-pipeline-exit").write_text("1", encoding="utf-8")
            args = SimpleNamespace(parent=root, output=root / "queue", s1=root / "s1", poll_seconds=1)
            fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            with patch.dict("sys.modules", {"fcntl": fake_fcntl}), patch("subprocess.run") as launch:
                with self.assertRaisesRegex(RuntimeError, "S1 pipeline failed"):
                    run(args)
                launch.assert_not_called()
            status = json.loads((root / "queue" / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stage"], "waiting_for_s1")

    def test_failed_gpu_smoke_does_not_launch_formal_training(self):
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "s1-pipeline.pid").write_text("123", encoding="utf-8")
            (root / "s1-pipeline-exit").write_text("0", encoding="utf-8")
            (root / "s1-pipeline-stage").write_text("s1_formal_completed", encoding="utf-8")
            for name in ("dataset", "geometry", "regional"):
                (root / name).mkdir()
            write_json(root / "dataset" / "manifest.json", {"dataset_id": "data"})
            write_json(root / "geometry" / "manifest.json", {"dataset_id": "geo"})
            manifest = {"geometry_dataset_id": "geo"}
            manifest["dataset_id"] = digest(manifest)
            write_json(root / "regional" / "manifest.json", manifest)
            write_json(root / "validation.json", {"regional_manifest_sha256": sha256(root / "regional" / "manifest.json")})
            write_json(root / "s2.json", {"schema": "b4dl-regional-config-v1", "use_temporal": True,
                                           "use_geometry": False, "centroid_policy": "observed_assignment"})
            args = SimpleNamespace(parent=root, output=root / "queue", s1=root / "s1", poll_seconds=1,
                updates=2, regional=root / "regional", validation=root / "validation.json", dataset=root / "dataset",
                geometry=root / "geometry", s2_config=root / "s2.json", recipe=root / "recipe",
                baseline=root / "base", preparation=root / "prep", source=root, gpu_lock=root / "lock",
                small_regional=root / "small")
            result = {"identity": {"common": {"reasoning_dataset_id": "data"}}}
            fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            with patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                    patch("scripts.queue_regional_s2.verify_training", return_value=result), \
                    patch("scripts.queue_regional_s2.validate_s1_binding"), \
                    patch("scripts.queue_regional_s2.recipe_for"), \
                    patch("scripts.queue_regional_s2.implementation_identity"), \
                    patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "smoke")) as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    run(args)
                self.assertEqual(launch.call_count, 1)
            status = json.loads((root / "queue" / "status.json").read_text())
            self.assertEqual((status["status"], status["stage"]), ("failed", "s2_gpu_smoke"))
            self.assertFalse((root / "queue" / "s2-formal").exists())


if __name__ == "__main__":
    unittest.main()
