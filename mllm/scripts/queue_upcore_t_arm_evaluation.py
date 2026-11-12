"""Evaluate the five upcore T arms on internal_dev and rank them."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCHEMA = "b4dl-t-arm-evaluation-plan-v1"
TASKS = ("existence", "position", "distance", "motion")
PREFLIGHT = r'''
import json
import sys

from b4dl_core.b3 import device_lease
from b4dl_core.reasoning.data import ReasoningDataset
from b4dl_core.reasoning.evaluation import load_model
from b4dl_core.reasoning.mixed import MixedPlan

spec = json.loads(sys.argv[1])
dataset = ReasoningDataset(spec["dataset"], spec["geometry"], "internal_dev")
if spec["regional"]:
    from b4dl_core.regional_data import RegionalDataset
    dataset = RegionalDataset(dataset, spec["regional"])
plan = MixedPlan(spec["mixed_plan"], dataset)
with device_lease("cuda:0", spec["gpu_lock"]):
    tokenizer, model, meta = load_model(dataset, spec["baseline"], spec["preparation"],
                                        spec["checkpoint"], "cuda:0", plan)
print("LOAD_OK step=%s requests=%s" % (meta.get("step"), len(dataset.requests)))
'''


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA, "unknown plan")
    require(plan.get("split") == "internal_dev" and plan.get("condition") == "identity", "fixed split changed")
    require(plan.get("max_new_tokens") == 512 and plan.get("seed") == 42, "fixed decoding changed")
    require(plan.get("updates") == 13851 and plan.get("requests") == 13602, "invalid budget or request count")
    names = [arm["name"] for arm in plan["arms"]]
    require(len(names) == 5 and len(set(names)) == 5, "five distinct arms required")
    for key in ("dataset", "geometry", "baseline", "preparation", "mixed_plan", "python", "source"):
        require(Path(plan[key]).exists(), "missing input: " + key)
    for arm in plan["arms"]:
        require(Path(arm["checkpoint"]).is_file(), "missing checkpoint: " + arm["name"])
        if arm.get("regional"):
            require(Path(arm["regional"]).is_dir(), "missing regional features: " + arm["name"])
    return plan


def validate_checkpoint(plan, arm):
    checkpoint = Path(arm["checkpoint"])
    receipt = read(checkpoint.parent / "completion.json")
    require(receipt.get("status") == "training_complete", arm["name"] + " training incomplete")
    require(receipt.get("updates") == plan["updates"] and receipt.get("full_budget") is True,
            arm["name"] + " budget mismatch")
    actual = sha256(checkpoint)
    require(actual == receipt.get("checkpoint_sha256"), arm["name"] + " checkpoint hash mismatch")
    if arm.get("checkpoint_sha256"):
        require(actual == arm["checkpoint_sha256"], arm["name"] + " frozen hash mismatch")
    return actual


def validate_inference(output, requests):
    receipt = read(Path(output) / "completion.json")
    run = read(Path(output) / "run.json")
    require(receipt.get("status") == "reasoning_inference_complete", "invalid inference receipt")
    require(receipt.get("requests") == requests, "inference request count mismatch")
    require(receipt.get("run") == run, "inference run mismatch")
    return run


def validate_report(report, requests):
    data = read(report)
    require(data.get("complete_prediction_set") is True, "incomplete report")
    require(data.get("summary", {}).get("n") == requests, "report count mismatch")
    require(set(data.get("by_task", {})) == set(TASKS), "missing geometry tasks")
    return data


def report_scores(report):
    trace = sum(report["by_task"][name]["whole_trace_pass_rate"] for name in TASKS) / len(TASKS)
    accuracy = sum(report["by_task"][name]["accuracy"] for name in TASKS) / len(TASKS)
    return trace, accuracy


def rank_arms(rows):
    return sorted(rows, key=lambda row: (-row["trace_macro"], -row["accuracy_macro"], row["cost_rank"]))


def infer_command(plan, arm, output):
    command = [plan["python"], "-u", "-m", "b4dl_core.reasoning", "infer",
               "--dataset", plan["dataset"], "--geometry-dataset", plan["geometry"],
               "--baseline", plan["baseline"], "--preparation", plan["preparation"],
               "--checkpoint", arm["checkpoint"]]
    if arm.get("regional"):
        command += ["--regional-features", arm["regional"]]
    command += ["--mixed-plan", plan["mixed_plan"], "--gpu-lock", plan["gpu_lock"],
                "--split", plan["split"], "--condition", plan["condition"],
                "--max-new-tokens", str(plan["max_new_tokens"]), "--seed", str(plan["seed"]),
                "--output", str(output)]
    return command


def evaluate_command(plan, output, report):
    return [plan["python"], "-u", "-m", "b4dl_core.reasoning", "evaluate",
            "--dataset", plan["dataset"], "--geometry-dataset", plan["geometry"],
            "--split", plan["split"], "--predictions", str(Path(output) / "predictions.jsonl"),
            "--output", str(report)]


def preflight_command(plan, arm):
    spec = {"dataset": plan["dataset"], "geometry": plan["geometry"], "baseline": plan["baseline"],
            "preparation": plan["preparation"], "checkpoint": arm["checkpoint"], "regional": arm.get("regional"),
            "mixed_plan": plan["mixed_plan"], "gpu_lock": plan["gpu_lock"]}
    return [plan["python"], "-c", PREFLIGHT, json.dumps(spec)]


def environment(plan):
    return dict(os.environ, PYTHONPATH=str(Path(plan["source"]) / "mllm"), PYTHONNOUSERSITE="1",
                PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    plan = validate_plan(read(args.plan))
    root = args.output.resolve()
    root.mkdir(parents=False, exist_ok=False)
    status = {"schema": "b4dl-t-arm-evaluation-status-v1", "status": "running", "stage": "preflight",
              "pid": os.getpid(), "stage_pid": None, "completed": [], "error": None,
              "plan_sha256": sha256(args.plan), "started_utc": now()}

    def save(**fields):
        status.update(fields)
        write(root / "status.json", status)

    def perform(name, command):
        if shutil.disk_usage(root).free < plan["minimum_free_gib"] * 1024 ** 3:
            raise RuntimeError("insufficient disk reserve for " + name)
        log = root / (name + ".log")
        save(stage=name, command=command, stage_pid=None, log=str(log), stage_started_utc=now())
        begin = time.monotonic()
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=plan["source"], env=environment(plan),
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            save(stage_pid=process.pid)
            code = process.wait()
        save(stage_pid=None)
        if code:
            raise RuntimeError(name + " exited with code " + str(code))
        status["completed"].append(name)
        save(last_completed=name, last_seconds=round(time.monotonic() - begin, 1))

    save()
    try:
        import fcntl
        with Path(plan["controller_lock"]).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            hashes = {arm["name"]: validate_checkpoint(plan, arm) for arm in plan["arms"]}
            if args.preflight_only:
                for arm in plan["arms"]:
                    perform("preflight-" + arm["name"], preflight_command(plan, arm))
                save(status="preflight_complete", stage="preflight_complete", checkpoint_sha256=hashes)
                return
            summary = []
            for arm in plan["arms"]:
                output, report = root / arm["name"], root / (arm["name"] + "-report.json")
                perform(arm["name"] + "-infer", infer_command(plan, arm, output))
                validate_inference(output, plan["requests"])
                perform(arm["name"] + "-score", evaluate_command(plan, output, report))
                data = validate_report(report, plan["requests"])
                trace, accuracy = report_scores(data)
                summary.append({"name": arm["name"], "trace_macro": trace, "accuracy_macro": accuracy,
                                "cost_rank": arm["cost_rank"], "checkpoint_sha256": hashes[arm["name"]],
                                "report": str(report)})
                write(root / "t-arm-summary.json", {"arms": summary})
            ordered = rank_arms(summary)
            write(root / "t-arm-ranking.json", {
                "order": [row["name"] for row in ordered], "arms": ordered, "selected": ordered[0]["name"],
                "note": "internal_dev screening only; test, six-task and intervention gates remain"})
            save(status="complete", stage="complete", finished_utc=now())
    except Exception as error:
        save(status="failed", error_type=type(error).__name__, error=str(error), stage_pid=None, finished_utc=now())
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
