"""Contract checks for the frozen endpoint evaluation queue."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("endpoint_queue", ROOT / "scripts/run_upcore_evaluation_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_plan_covers_all_endpoints_and_defers_comparisons(tmp_path):
    plan = queue.read(ROOT / "configs/upcore_endpoint_evaluation_20261108.json")
    stages = queue.build_stages(plan, tmp_path)
    names = {stage["name"] for stage in stages}
    models = {"s1", "s2", "t0", "b3", "b3_t0"}
    for model in models:
        assert f"smoke-{model}" in names
        assert f"six-infer-{model}" in names
        assert f"six-score-{model}" in names
        for split in ("internal_dev", "test"):
            assert f"geometry-{model}-{split}-identity" in names
            assert f"score-{model}-{split}-identity" in names
    for model in ("s1", "s2", "t0"):
        for condition in ("zero", "repeat_first", "shuffle_features", "cross_scene"):
            assert f"control-{model}-{condition}" in names
            assert f"compare-control-{model}-{condition}" in names
    assert len(names) == len(stages)
    first_comparison = next(i for i, stage in enumerate(stages) if stage["kind"] == "comparison")
    assert all(stage["kind"] == "comparison" for stage in stages[first_comparison:])
    assert all(stage["kind"] != "comparison" for stage in stages[:first_comparison])


def test_endpoint_requires_matching_completion_and_hash(tmp_path):
    checkpoint = tmp_path / "step-3.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = queue.sha(checkpoint)
    (tmp_path / "completion.json").write_text(json.dumps({"status": "training_complete", "full_budget": True,
        "updates": 3, "checkpoint_sha256": digest}), encoding="utf-8")
    (tmp_path / "latest.json").write_text(json.dumps({"step": 3, "checkpoint": checkpoint.name,
        "sha256": digest}), encoding="utf-8")
    model = {"name": "test", "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "updates": 3}
    assert queue.check_endpoint(model)["checkpoint_sha256"] == digest
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint endpoint receipt mismatch"):
        queue.check_endpoint(model)


def test_prediction_receipt_rejects_changed_output(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{}\n', encoding="utf-8")
    (tmp_path / "run.json").write_text('{"id": 1}', encoding="utf-8")
    receipt = {"status": "reasoning_inference_complete", "run": {"id": 1},
               "predictions_sha256": queue.sha(predictions)}
    (tmp_path / "completion.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert queue.inference_receipt(tmp_path, receipt["status"]) == receipt
    predictions.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prediction completion mismatch"):
        queue.inference_receipt(tmp_path, receipt["status"])
