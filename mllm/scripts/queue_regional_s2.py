"""Wait for a verified S1 endpoint, smoke-test S2, then train matched S2.

Each run uses a new output directory. A nonzero stage or mismatched identity
stops the queue; it never resumes or replaces an existing experiment.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from b4dl_core.io import digest, read_json, sha256, write_json
from b4dl_core.reasoning.__main__ import recipe_for
from b4dl_core.reasoning.trainer import implementation_identity
from b4dl_core.regional_model import RegionalConfig


def values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from values(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from values(value)
    else:
        yield obj


def verify_training(output, expected_status, expected_steps):
    """Verify run, exact budget, contiguous logs, checkpoint and optimizer."""
    output = Path(output)
    completion = read_json(output / "completion.json")
    run = read_json(output / "run.json")
    latest = read_json(output / "latest.json")
    if (completion.get("status") != expected_status or completion.get("updates") != expected_steps
            or run.get("requested_updates") != expected_steps
            or latest.get("step") != expected_steps or latest.get("checkpoint") != f"step-{expected_steps}.pt"
            or completion.get("identity") != run.get("identity")):
        raise ValueError("training completion or identity mismatch")
    if expected_status == "training_complete" and (completion.get("full_budget") is not True
            or run.get("total_updates") != expected_steps
            or completion["identity"].get("complete_budget") is not True
            or completion["identity"]["recipe"].get("engineering") is not False):
        raise ValueError("formal training has not completed its full budget")
    checkpoint = output / latest["checkpoint"]
    actual_hash = sha256(checkpoint)
    if actual_hash != latest.get("sha256") or actual_hash != completion.get("checkpoint_sha256"):
        raise ValueError("training checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("schema") != "b4dl-reasoning-checkpoint-v1" or payload.get("stage") != "sft"
            or payload.get("step") != expected_steps or payload.get("identity") != completion["identity"]
            or payload.get("optimizer") is None):
        raise ValueError("checkpoint stage, identity, step or optimizer mismatch")
    tensors = [value for value in values((payload["parameters"], payload["optimizer"]))
               if isinstance(value, torch.Tensor)]
    if not tensors or not all(torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("nonfinite or missing checkpoint tensors")
    steps = []
    for state in payload["optimizer"]["state"].values():
        if "step" not in state:
            raise ValueError("optimizer state has no step")
        steps.append(int(state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"]))
    if not steps or any(step != expected_steps for step in steps):
        raise ValueError("optimizer state step mismatch")
    count = 0
    with (output / "training.jsonl").open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("step") != count or not all(math.isfinite(value) for value in values(row)
                                                    if isinstance(value, float)):
                raise ValueError("nonfinite or noncontiguous training log")
    if count != expected_steps:
        raise ValueError("training log count mismatch")
    return {"checkpoint_sha256": actual_hash, "updates": expected_steps, "finite_tensors": len(tensors),
            "identity": completion["identity"], "full_budget": completion["full_budget"]}


def validate_s1_binding(s1, config, recipe, baseline, preparation, regional_manifest, validation, source_id):
    identity, common = s1["identity"], s1["identity"]["common"]
    old_config = common["options"]["regional"]
    if (not s1["full_budget"] or identity["stage"] != "sft" or identity["recipe"] != recipe
            or common["implementation"] != source_id or common["baseline_sha256"] != sha256(baseline)
            or common["preparation_sha256"] != sha256(preparation)
            or common["regional_dataset_id"] != regional_manifest["dataset_id"]
            or validation.get("status") != "validated"
            or validation.get("regional_dataset_id") != regional_manifest["dataset_id"]
            or validation.get("geometry_dataset_id") != regional_manifest["geometry_dataset_id"]
            or not regional_manifest.get("complete")):
        raise ValueError("S1 source, data, recipe or full validation mismatch")
    if (not old_config or old_config["use_temporal"] or old_config["use_geometry"]
            or not config.use_temporal or config.use_geometry
            or config.as_dict() != {**old_config, "use_temporal": True}):
        raise ValueError("S2 must differ from S1 only by temporal fusion")


def process_alive(pid, expected):
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        return expected.encode() in cmdline and state != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def train_command(args, output, *, engineering=False):
    command = [sys.executable, "-u", "-m", "b4dl_core.reasoning", "sft",
        "--dataset", str(args.dataset), "--geometry-dataset", str(args.geometry),
        "--baseline", str(args.baseline), "--preparation", str(args.preparation),
        "--recipe", str(args.recipe), "--gpu-lock", str(args.gpu_lock),
        "--regional-features", str(args.small_regional if engineering else args.regional),
        "--regional-config", str(args.s2_config), "--output", str(output)]
    if engineering:
        command.extend(("--engineering", "--engineering-region-subset", "--max-updates", "50"))
    return command


def run(args):
    import fcntl
    root, queue = args.parent, args.output
    with (root / "s2-next.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.mkdir(parents=False, exist_ok=False)
        status = {"schema": "b4dl-regional-s2-queue-v1", "status": "running", "stage": "waiting_for_s1",
                  "s1": str(args.s1), "created_unix": time.time(), "pid": os.getpid(), "error": None}

        def stage(name):
            status["stage"] = name
            write_json(queue / "status.json", status)
            print(name, flush=True)

        stage("waiting_for_s1")
        try:
            pid = int((root / "s1-pipeline.pid").read_text().strip())
            while not (root / "s1-pipeline-exit").exists():
                if not process_alive(pid, "run-s1-pipeline.sh"):
                    raise RuntimeError("S1 pipeline stopped without an exit receipt")
                time.sleep(args.poll_seconds)
            if ((root / "s1-pipeline-exit").read_text().strip() != "0"
                    or (root / "s1-pipeline-stage").read_text().strip() != "s1_formal_completed"):
                raise RuntimeError("S1 pipeline failed or did not reach its completed stage")
            stage("verifying_s1")
            s1 = verify_training(args.s1, "training_complete", args.updates)
            manifest = read_json(args.regional / "manifest.json")
            if manifest.get("dataset_id") != digest({k: v for k, v in manifest.items() if k != "dataset_id"}):
                raise ValueError("regional manifest identity mismatch")
            validation = read_json(args.validation)
            if validation.get("regional_manifest_sha256") != sha256(args.regional / "manifest.json"):
                raise ValueError("regional validation receipt does not match current manifest")
            if (read_json(args.dataset / "manifest.json").get("dataset_id")
                    != s1["identity"]["common"]["reasoning_dataset_id"]
                    or read_json(args.geometry / "manifest.json").get("dataset_id") != manifest["geometry_dataset_id"]):
                raise ValueError("S1 reasoning or geometry dataset mismatch")
            raw_config = read_json(args.s2_config)
            if raw_config.pop("schema", None) != "b4dl-regional-config-v1":
                raise ValueError("invalid S2 config")
            config = RegionalConfig(**raw_config)
            recipe = recipe_for(args.recipe)
            validate_s1_binding(s1, config, recipe, args.baseline, args.preparation, manifest,
                                validation, implementation_identity())
            write_json(queue / "s1-verification.json", s1)
            env = dict(os.environ, PYTHONPATH=str(args.source / "mllm"), PYTHONNOUSERSITE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            stage("s2_gpu_smoke")
            with (queue / "s2-smoke.log").open("x", encoding="utf-8") as log:
                subprocess.run(train_command(args, queue / "s2-smoke", engineering=True), cwd=args.source,
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            stage("verifying_s2_smoke")
            smoke = verify_training(queue / "s2-smoke", "engineering_budget_complete", 50)
            small_manifest = read_json(args.small_regional / "manifest.json")
            smoke_common = smoke["identity"]["common"]
            if (smoke_common["implementation"] != common_implementation(s1)
                    or smoke_common["regional_dataset_id"] != small_manifest["dataset_id"]
                    or smoke_common["options"]["regional"] != config.as_dict()
                    or not smoke["identity"]["recipe"]["engineering"] or smoke["full_budget"]):
                raise ValueError("S2 smoke identity mismatch")
            write_json(queue / "s2-smoke-verification.json", smoke)
            stage("s2_formal_training")
            with (queue / "s2-formal.log").open("x", encoding="utf-8") as log:
                subprocess.run(train_command(args, queue / "s2-formal"), cwd=args.source,
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            stage("verifying_s2")
            result = verify_training(queue / "s2-formal", "training_complete", args.updates)
            other = result["identity"]["common"]
            original = s1["identity"]["common"]
            if (not result["full_budget"] or result["identity"]["recipe"] != recipe
                    or other["options"]["regional"] != config.as_dict()
                    or any(other.get(key) != original.get(key) for key in
                           ("baseline_sha256", "preparation_sha256", "implementation", "reasoning_dataset_id",
                            "regional_dataset_id", "reward_version"))):
                raise ValueError("completed S2 differs from matched S1 assets or budget")
            write_json(queue / "s2-verification.json", result)
            status.update(status="complete", stage="s2_complete", finished_unix=time.time())
            write_json(queue / "status.json", status)
        except Exception as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error), finished_unix=time.time())
            write_json(queue / "status.json", status)
            traceback.print_exc()
            raise


def common_implementation(result):
    return result["identity"]["common"]["implementation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "s1", "source", "dataset", "geometry", "baseline", "preparation",
                 "recipe", "regional", "small-regional", "validation", "s2-config", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=13851)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    for name in ("parent", "output", "s1", "source", "dataset", "geometry", "baseline", "preparation", "recipe",
                 "regional", "small_regional", "validation", "s2_config", "gpu_lock"):
        setattr(args, name, getattr(args, name).resolve())
    if args.updates < 1 or not 1 <= args.poll_seconds <= 300 or args.output.parent != args.parent:
        parser.error("invalid budget, poll interval or queue output location")
    run(args)


if __name__ == "__main__":
    main()
