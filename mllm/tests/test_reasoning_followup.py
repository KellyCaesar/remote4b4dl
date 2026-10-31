"""Fault and sequencing checks for the unattended post-evaluation queue."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/queue_reasoning_followup.py"
spec = importlib.util.spec_from_file_location("reasoning_followup", SCRIPT)
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    followup.write(path, value)


def prediction_fixture(directory, *, arm=None):
    plan = {"checkpoint_sha256": "ckpt", "baseline_sha256": "base", "preparation_sha256": "prep",
            "implementation": "impl", "protocol_id": "proto", "dataset_id": "data", "seed": 42,
            "max_new_tokens": 512}
    expected = {"a": "fingerprint-a", "b": "fingerprint-b"}
    run = {"checkpoint_sha256": "ckpt", "condition": "identity", "split": "internal_dev",
           "dataset_id": "data", "seed": 42, "max_new_tokens": 512, "decoding": "greedy"}
    if arm:
        run = dict(plan, oracle_assisted_metadata=False, model_kind="reasoning" if arm == "sft" else "original_b3",
                   model_artifact_sha256="ckpt" if arm == "sft" else "base")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "predictions.jsonl").write_text("".join(json.dumps({"request_id": k, "input_fingerprint": v}) + "\n"
                                                        for k, v in expected.items()), encoding="utf-8")
    put(directory / "run.json", run)
    receipt = {"status": "six_task_inference_complete" if arm else "reasoning_inference_complete", "run": run,
               "requests": 2, "predictions_sha256": followup.sha(directory / "predictions.jsonl")}
    put(directory / "completion.json", receipt)
    return plan, expected


def predecessor_fixture(root):
    plan = {"evaluation": str(root / "evaluation"), "run_root": str(root), "checkpoint_sha256": "ckpt", "protocol_id": "proto"}
    put(root / "evaluation/status.json", {"status": "geometry_evaluation_complete", "checkpoint_sha256": "ckpt",
        "completed_stages": [{"name": name, "returncode": 0} for name in
            ("identity-infer", "identity-evaluate", "zero-infer", "zero-evaluate", "intervention-compare")]})
    put(root / "evaluation/six-task/status.json", {"status": "six_task_comparison_complete",
        "checkpoint_sha256": "ckpt", "protocol_id": "proto", "completed_stages": [{"stage": name, "exit_code": 0}
            for name in ("sft-infer", "sft-metrics", "b3-infer", "b3-metrics", "compare")]})
    return plan


class FollowupTests(unittest.TestCase):
    def test_plan_cannot_switch_diagnostics_to_test(self):
        plan = followup.read(SCRIPT.parents[1] / "configs/reasoning_followup_20261101.json")
        plan["split"] = "test"
        with self.assertRaisesRegex(ValueError, "development data"):
            followup.preflight(plan, check_implementation=False)

    def test_plan_cannot_silently_drop_controls(self):
        plan = followup.read(SCRIPT.parents[1] / "configs/reasoning_followup_20261101.json")
        plan["conditions"] = ["identity"]
        with self.assertRaisesRegex(ValueError, "conditions changed"):
            followup.preflight(plan, check_implementation=False)

    def test_both_predecessors_must_complete_all_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = predecessor_fixture(root)
            self.assertTrue(followup.predecessor_ready(plan))
            path = root / "evaluation/six-task/status.json"
            state = followup.read(path)
            state["completed_stages"].pop()
            put(path, state)
            with self.assertRaisesRegex(ValueError, "stages incomplete"):
                followup.predecessor_ready(plan)

    def test_failed_or_missing_controller_cannot_release_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = predecessor_fixture(root)
            path = root / "evaluation/six-task/status.json"
            state = followup.read(path)
            for status in ("failed", "running"):
                state.update(status=status, pid=999999999)
                put(path, state)
                with self.assertRaises(ValueError):
                    followup.predecessor_ready(plan)

    def test_changed_predecessor_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = predecessor_fixture(Path(tmp))
            plan["checkpoint_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                followup.predecessor_ready(plan)

    def test_prediction_content_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan, expected = prediction_fixture(directory)
            followup.verify_predictions(directory, expected, plan, condition="identity", split="internal_dev")
            with (directory / "predictions.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                followup.verify_predictions(directory, expected, plan, condition="identity", split="internal_dev")

    def test_duplicate_predictions_rejected_even_with_updated_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan, expected = prediction_fixture(directory)
            path = directory / "predictions.jsonl"
            path.write_text(json.dumps({"request_id": "a", "input_fingerprint": expected["a"]}) + "\n" +
                            json.dumps({"request_id": "a", "input_fingerprint": expected["a"]}) + "\n")
            receipt = followup.read(directory / "completion.json")
            receipt["predictions_sha256"] = followup.sha(path)
            put(directory / "completion.json", receipt)
            with self.assertRaisesRegex(ValueError, "IDs/fingerprints"):
                followup.verify_predictions(directory, expected, plan, condition="identity", split="internal_dev")

    def test_six_task_wrong_model_arm_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan, expected = prediction_fixture(directory, arm="sft")
            followup.verify_predictions(directory, expected, plan, arm="sft")
            with self.assertRaisesRegex(ValueError, "model arm"):
                followup.verify_predictions(directory, expected, plan, arm="b3")

    def test_valid_answer_with_unsupported_steps_is_separate(self):
        template = {"scene_id": "scene", "task": "motion", "answer": "Yes", "truth": "Yes", "status": "ok", "absolute_errors": {}}
        report = {"run": {"split": "internal_dev"}, "rows": [
            dict(template, request_id="a", valid=True, correct=True, step_scores=[1, 0]),
            dict(template, request_id="b", valid=True, correct=True, step_scores=[1, 1]),
            dict(template, request_id="c", valid=False, correct=False, step_scores=[]),
            dict(template, request_id="d", valid=True, correct=False, step_scores=[1, 1])]}
        audit = followup.error_audit(report)
        self.assertEqual(set(audit["counts"]["motion"].values()), {1})
        self.assertEqual(len(audit["counts"]["motion"]), 4)

    def test_directional_regression_is_not_automatic_training(self):
        comparison = {"per_task_metrics": {t: {m: {"base": .5, "candidate": .6, "delta": .1}} for t, m in followup.PRIMARY.items()}}
        dev = {"by_task": {"motion": {"whole_trace_pass_rate": .1, "n": 2}}}
        decision = followup.recommendations(comparison, dev, {})
        self.assertEqual(decision["priority"], "grounded_reasoning")
        comparison["per_task_metrics"]["time_grounding"]["miou"]["delta"] = -.01
        decision = followup.recommendations(comparison, dev, {})
        self.assertEqual(decision["priority"], "six_task_retention")
        self.assertFalse(decision["training_launched"])
        self.assertTrue(decision["grpo_gate"].startswith("blocked"))

    def test_completed_stage_is_not_repeated_and_hash_change_stops_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            put(path, {"ok": True})
            state = {"completed_stages": {"stage": {"args": ["infer"], "files": {str(path): followup.sha(path)}}}}
            queue = followup.Queue({"output": tmp}, state, 0)
            with patch.object(followup.subprocess, "Popen") as launch:
                queue.stage("stage", ["infer"], [path])
                launch.assert_not_called()
                put(path, {"ok": False})
                with self.assertRaisesRegex(ValueError, "outputs changed"):
                    queue.stage("stage", ["infer"], [path])
                launch.assert_not_called()

    def test_stop_file_prevents_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "STOP").touch()
            queue = followup.Queue({"output": tmp}, {"completed_stages": {}}, 0)
            with patch.object(followup.subprocess, "Popen") as launch:
                with self.assertRaises(InterruptedError):
                    queue.stage("new", ["infer"], [])
                launch.assert_not_called()

    def test_wait_does_not_launch_work_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = followup.Queue({"output": tmp, "wait_timeout_hours": 1, "poll_seconds": .01}, {"completed_stages": {}}, 0)
            with patch.object(followup, "predecessor_ready", return_value=False), patch.object(followup.time, "sleep", side_effect=InterruptedError), \
                    patch.object(followup.subprocess, "Popen") as launch:
                with self.assertRaises(InterruptedError):
                    queue.run()
                launch.assert_not_called()
                self.assertEqual(followup.read(Path(tmp) / "status.json")["status"], "waiting_for_evaluation")

    def test_timeout_terminates_only_the_spawned_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"output": tmp, "run_root": tmp, "python": "unused", "stage_timeout_hours": 1e-12, "poll_seconds": .01}
            queue = followup.Queue(plan, {"completed_stages": {}}, 0)
            process = MagicMock(pid=12345)
            process.poll.return_value = None
            with patch.object(followup.time, "monotonic", side_effect=[0, 1]), \
                    patch.object(followup.subprocess, "Popen", return_value=process), \
                    patch.object(followup.os, "killpg", create=True) as kill:
                with self.assertRaises(TimeoutError):
                    queue.stage("infer", ["infer"], [])
                kill.assert_called_once_with(12345, followup.signal.SIGTERM)
                self.assertNotIn("stage_pid", queue.state)
                self.assertEqual(queue.state["completed_stages"], {})

    @unittest.skipIf(os.name != "posix", "Linux flock is the deployment contract")
    def test_duplicate_queue_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with followup.queue_lock(Path(tmp) / ".lock"):
                with self.assertRaises(BlockingIOError):
                    with followup.queue_lock(Path(tmp) / ".lock"):
                        self.fail("duplicate queue acquired the lease")


if __name__ == "__main__":
    unittest.main()
