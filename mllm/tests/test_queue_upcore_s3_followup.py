"""Test follow-up scheduling and screening without starting GPU work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import queue_upcore_s3_followup as queue


def report(accuracy, trace):
    return {"summary": {"accuracy": accuracy, "whole_trace_pass_rate": trace},
            "by_task": {name: {"accuracy": accuracy, "whole_trace_pass_rate": trace}
                        for name in ("existence", "position", "distance", "motion")}}


class FollowupQueueTests(unittest.TestCase):
    def test_structure_selection_and_geometry_gate(self):
        reports = {"s1": report(.76, .50), "s2": report(.78, .51), "s3": report(.77, .52)}
        chosen = queue.choose_structure(reports, report(.75, .49))
        self.assertEqual(chosen["selected"], "s3")
        self.assertTrue(chosen["geometry_screen_passed"])
        rejected = queue.choose_structure(reports, report(.79, .53))
        self.assertFalse(rejected["geometry_screen_passed"])
        tied = queue.choose_structure({name: report(.8, .6) for name in reports}, report(.75, .49))
        self.assertEqual(tied["selected"], "s1")

    def test_command_keeps_mixed_smoke_on_complete_regional_data(self):
        p = {key: key for key in ("dataset", "geometry", "baseline", "preparation", "recipe", "gpu_lock",
                                  "small_regional", "regional", "mixed_plan")}
        p["smoke_updates"] = 50
        structural = queue.train_command(p, "out", "s3", smoke=True)
        self.assertIn("small_regional", structural)
        self.assertIn("--engineering-region-subset", structural)
        mixed = queue.train_command(p, "out", "s3", "t2", smoke=True)
        self.assertIn("regional", mixed)
        self.assertIn("mixed_plan", mixed)
        self.assertNotIn("--engineering-region-subset", mixed)
        formal = queue.train_command(p, "out", "s3", "t2")
        self.assertNotIn("--engineering", formal)
        self.assertNotIn("--resume", formal)

    def test_wait_refuses_failed_and_missing_running_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text('{}', encoding="utf-8")
            eval_dir = root / "eval"
            eval_dir.mkdir()
            plan = {"evaluation_plan": str(plan_path), "evaluation_plan_sha256": queue.sha256(plan_path),
                    "evaluation_output": str(eval_dir)}
            status = {"plan_sha256": plan["evaluation_plan_sha256"], "status": "failed", "error": "bad"}
            (eval_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                queue.validate_upstream(plan)
            status.update(status="running", pid=12345)
            (eval_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with patch.object(queue, "upstream_alive", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "process missing"):
                    queue.validate_upstream(plan)
            with patch.object(queue, "upstream_alive", return_value=True):
                self.assertFalse(queue.validate_upstream(plan))

    def test_completed_upstream_requires_every_stage_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps({"models": [{"name": "s1"}]}), encoding="utf-8")
            eval_dir = root / "eval"
            eval_dir.mkdir()
            plan = {"evaluation_plan": str(plan_path), "evaluation_plan_sha256": queue.sha256(plan_path),
                    "evaluation_output": str(eval_dir)}
            status = {"plan_sha256": plan["evaluation_plan_sha256"], "status": "complete", "completed": []}
            (eval_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            stages = [{"name": "inference", "model": "s1"}]
            with patch.object(queue, "build_stages", return_value=stages), patch.object(queue, "validate_stage") as check:
                with self.assertRaisesRegex(ValueError, "stage list"):
                    queue.validate_upstream(plan)
                status["completed"] = ["inference"]
                (eval_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                self.assertTrue(queue.validate_upstream(plan))
                check.assert_called_once_with(stages[0], {"name": "s1"})
                check.side_effect = ValueError("bad receipt")
                with self.assertRaisesRegex(ValueError, "bad receipt"):
                    queue.validate_upstream(plan)

    def test_incomplete_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            output.write_text(json.dumps({"complete_prediction_set": False, "run": {}, "summary": {"n": 4}}))
            with patch.object(queue, "inference_receipt", return_value={"requests": 4, "run": {}}):
                with self.assertRaisesRegex(ValueError, "report/receipt"):
                    queue.verified_report(root, output, 4)

    def test_screening_rejects_changed_truth(self):
        row = {"request_id": "r", "scene_id": "s", "task": "existence", "truth": "Yes"}
        report_row = {"rows": [row], "summary": {"n": 1}, "run": {}}
        reports = {name: json.loads(json.dumps(report_row)) for name in ("s1", "s2", "s3")}
        historical = json.loads(json.dumps(report_row))
        queue.validate_screen_reports(reports, historical)
        historical["rows"][0]["truth"] = "No"
        with self.assertRaisesRegex(ValueError, "question or truth differs"):
            queue.validate_screen_reports(reports, historical)


if __name__ == "__main__":
    unittest.main()
