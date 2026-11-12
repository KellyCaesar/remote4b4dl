"""Test the T-arm evaluation queue without GPU work."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import queue_upcore_t_arm_evaluation as queue


def make_plan(root, arms=5):
    for name in ("dataset", "geometry", "regional", "mixed_plan", "source"):
        (root / name).mkdir(exist_ok=True)
    for name in ("baseline", "preparation", "python"):
        (root / name).write_text("{}", encoding="utf-8")
    checkpoints = []
    for index in range(arms):
        directory = root / ("arm%d" % index)
        directory.mkdir(exist_ok=True)
        checkpoint = directory / "step-1.pt"
        checkpoint.write_bytes(b"checkpoint-%d" % index)
        (directory / "completion.json").write_text(json.dumps({
            "status": "training_complete", "updates": 13851, "full_budget": True,
            "checkpoint_sha256": queue.sha256(checkpoint)}), encoding="utf-8")
        checkpoints.append({"name": "arm%d" % index, "checkpoint": str(checkpoint),
                            "checkpoint_sha256": queue.sha256(checkpoint), "regional": str(root / "regional"),
                            "cost_rank": index})
    return {"schema": queue.SCHEMA, "source": str(root / "source"), "dataset": str(root / "dataset"),
            "geometry": str(root / "geometry"), "baseline": str(root / "baseline"),
            "preparation": str(root / "preparation"), "mixed_plan": str(root / "mixed_plan"),
            "python": str(root / "python"), "gpu_lock": str(root / "gpu.lock"),
            "controller_lock": str(root / "controller.lock"), "updates": 13851, "requests": 13602,
            "split": "internal_dev", "condition": "identity", "max_new_tokens": 512, "seed": 42,
            "minimum_free_gib": 5, "arms": checkpoints}


class TArmEvaluationTests(unittest.TestCase):
    def test_plan_requires_five_arms_and_fixed_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan(root)
            queue.validate_plan(plan)
            plan["arms"] = plan["arms"][:4]
            with self.assertRaisesRegex(ValueError, "five distinct"):
                queue.validate_plan(plan)
            plan = make_plan(root)
            plan["seed"] = 43
            with self.assertRaisesRegex(ValueError, "fixed decoding"):
                queue.validate_plan(plan)

    def test_checkpoint_requires_complete_budget_and_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan(root)
            arm = plan["arms"][0]
            self.assertEqual(queue.validate_checkpoint(plan, arm), arm["checkpoint_sha256"])
            arm["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "frozen hash"):
                queue.validate_checkpoint(plan, arm)

    def test_report_must_be_complete_and_cover_four_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            data = {"complete_prediction_set": True, "summary": {"n": 13602},
                    "by_task": {name: {"accuracy": 0.5, "whole_trace_pass_rate": 0.25} for name in queue.TASKS}}
            report.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(queue.validate_report(report, 13602)["summary"]["n"], 13602)
            data["summary"]["n"] = 13601
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                queue.validate_report(report, 13602)
            data["summary"]["n"] = 13602
            data["by_task"].pop("motion")
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing geometry"):
                queue.validate_report(report, 13602)

    def test_ranking_prefers_trace_then_accuracy_then_lower_cost(self):
        rows = [
            {"name": "t3", "trace_macro": 0.30, "accuracy_macro": 0.60, "cost_rank": 3},
            {"name": "t1", "trace_macro": 0.30, "accuracy_macro": 0.62, "cost_rank": 1},
            {"name": "t0", "trace_macro": 0.30, "accuracy_macro": 0.62, "cost_rank": 0},
            {"name": "t2", "trace_macro": 0.31, "accuracy_macro": 0.59, "cost_rank": 2},
        ]
        ordered = queue.rank_arms(rows)
        self.assertEqual([row["name"] for row in ordered], ["t2", "t0", "t1", "t3"])


if __name__ == "__main__":
    unittest.main()
