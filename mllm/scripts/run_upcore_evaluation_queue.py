"""Run frozen endpoint evaluations serially, with receipts and safe resume."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def check_endpoint(model):
    if model.get("original_b3"):
        return {"kind": "original_b3", "baseline_sha256": sha(model["baseline"])}
    path = Path(model["checkpoint"])
    receipt = read(path.parent / "completion.json")
    latest = read(path.parent / "latest.json")
    actual = sha(path)
    if (receipt.get("status") != "training_complete" or not receipt.get("full_budget")
            or receipt.get("updates") != model["updates"] or latest.get("step") != model["updates"]
            or latest.get("checkpoint") != path.name or actual != model["checkpoint_sha256"]
            or receipt.get("checkpoint_sha256") != actual or latest.get("sha256") != actual):
        raise ValueError("checkpoint endpoint receipt mismatch: " + model["name"])
    return {"kind": "trained_endpoint", "checkpoint_sha256": actual, "updates": receipt["updates"]}


def inference_receipt(output, expected_status):
    output = Path(output)
    receipt = read(output / "completion.json")
    if (receipt.get("status") != expected_status or receipt.get("run") != read(output / "run.json")
            or receipt.get("predictions_sha256") != sha(output / "predictions.jsonl")):
        raise ValueError("prediction completion mismatch: " + str(output))
    return receipt


def validate_stage(stage, model):
    output = Path(stage["output"])
    kind = stage["kind"]
    if kind == "geometry-infer":
        inference_receipt(output, "reasoning_inference_complete")
    elif kind == "six-infer":
        inference_receipt(output, "six_task_inference_complete")
    elif kind == "six-score":
        receipt = read(output / "completion.json")
        if (receipt.get("status") != "six_task_metrics_complete" or receipt.get("metrics_sha256") != sha(output / "metrics.json")
                or receipt.get("predictions_sha256") != sha(stage["predictions"])):
            raise ValueError("six-task metric receipt mismatch")
    elif kind == "geometry-score":
        report = read(output)
        prediction_dir = Path(stage["predictions"]).parent
        receipt = inference_receipt(prediction_dir, "reasoning_inference_complete")
        if (not report.get("complete_prediction_set") or report.get("run") != receipt.get("run")
                or report.get("summary", {}).get("n") != receipt.get("requests")):
            raise ValueError("geometry metric output is incomplete or mismatched")
    elif kind == "smoke":
        result = read(output)
        artifact = model["baseline"] if model.get("original_b3") else model["checkpoint"]
        if result.get("status") != "evaluation_smoke_passed" or result.get("artifact_sha256") != sha(artifact):
            raise ValueError("evaluation smoke failed or artifact changed")
    elif kind == "comparison":
        if not isinstance(read(output), dict) or not read(output):
            raise ValueError("comparison result is empty")
    else:
        raise ValueError("unknown evaluation stage")


def stage_log(queue, name):
    logs = queue / "logs"
    logs.mkdir(exist_ok=True)
    index = 1
    while (logs / f"{name}.{index}.log").exists():
        index += 1
    return logs / f"{name}.{index}.log"


def model_args(model):
    result = ["--baseline", model["baseline"], "--preparation", model["preparation"]]
    if model.get("original_b3"):
        result.append("--original-b3")
    else:
        result += ["--checkpoint", model["checkpoint"]]
    if model.get("regional"):
        result += ["--regional-features", model["regional"]]
    if model.get("mixed_plan"):
        result += ["--mixed-plan", model["mixed_plan"]]
    return result


def build_stages(plan, queue):
    models = {row["name"]: row for row in plan["models"]}
    common = ["--dataset", plan["dataset"], "--geometry-dataset", plan["geometry"]]
    six_proto = plan["six_protocol"]
    stages = []
    for model in plan["models"]:
        output = queue / "smoke" / (model["name"] + ".json")
        smoke_common = ["--dataset", plan["dataset"], "--geometry", plan["geometry"], "--protocol", six_proto,
                        "--gpu-lock", plan["gpu_lock"], "--output", str(output)]
        stages.append({"name": "smoke-" + model["name"], "model": model["name"], "kind": "smoke",
            "output": str(output), "args": [plan["smoke_script"], *smoke_common, *model_args(model)]})
    for split in ("internal_dev", "test"):
        for model in plan["models"]:
            name = model["name"]
            output = queue / "geometry" / name / split / "identity"
            args = ["-m", "b4dl_core.reasoning", "infer", *common, *model_args(model), "--gpu-lock", plan["gpu_lock"],
                    "--split", split, "--condition", "identity", "--max-new-tokens", "512", "--seed", "42", "--output", str(output)]
            stages.append({"name": f"geometry-{name}-{split}-identity", "model": name, "kind": "geometry-infer",
                           "output": str(output), "args": args})
            report = queue / "geometry" / name / split / "identity-report.json"
            stages.append({"name": f"score-{name}-{split}-identity", "model": name, "kind": "geometry-score",
                "output": str(report), "predictions": str(output / "predictions.jsonl"),
                "args": ["-m", "b4dl_core.reasoning", "evaluate", *common, "--split", split,
                         "--predictions", str(output / "predictions.jsonl"), "--output", str(report)]})
        comparisons = (("s1", "s2", "--ablation"), ("b3", "s1", "--original-b3"),
                       ("b3", "s2", "--original-b3"), ("b3_t0", "t0", "--original-b3"))
        for left, right, mode in comparisons:
            report = queue / "comparisons" / f"geometry-{split}-{left}-vs-{right}.json"
            base = queue / "geometry" / left / split / "identity-report.json"
            candidate = queue / "geometry" / right / split / "identity-report.json"
            stages.append({"name": f"compare-geometry-{split}-{left}-{right}", "model": right, "kind": "comparison",
                "output": str(report), "args": ["-m", "b4dl_core.reasoning", "compare", mode,
                    "--base", str(base), "--candidate", str(candidate), "--output", str(report)]})
    for model in plan["models"]:
        name = model["name"]
        output = queue / "six-task" / name / "predictions"
        args = ["-m", "b4dl_core.reasoning", "six-task-infer", *common, *model_args(model),
                "--protocol", six_proto, "--gpu-lock", plan["gpu_lock"], "--output", str(output)]
        stages.append({"name": f"six-infer-{name}", "model": name, "kind": "six-infer", "output": str(output), "args": args})
        metrics = queue / "six-task" / name / "metrics"
        stages.append({"name": f"six-score-{name}", "model": name, "kind": "six-score", "output": str(metrics),
            "predictions": str(output / "predictions.jsonl"), "args": ["-m", "b4dl_core.reasoning", "six-task-metrics", *common,
                "--protocol", six_proto, "--predictions", str(output / "predictions.jsonl"), "--output", str(metrics)]})
    for left, right, ablation in (("s1", "s2", True), ("b3", "s1", False),
                                  ("b3", "s2", False), ("b3_t0", "t0", False)):
        output = queue / "comparisons" / f"six-{left}-vs-{right}.json"
        args = ["-m", "b4dl_core.reasoning", "six-task-compare", "--base", str(queue / "six-task" / left / "metrics"),
                "--candidate", str(queue / "six-task" / right / "metrics"), "--output", str(output)]
        if ablation:
            args.append("--ablation")
        stages.append({"name": f"compare-six-{left}-{right}", "model": right, "kind": "comparison", "output": str(output), "args": args})
    for name in ("s1", "s2", "t0"):
        model = models[name]
        for condition in ("zero", "repeat_first", "shuffle_features", "cross_scene"):
            output = queue / "geometry" / name / "internal_dev" / condition
            args = ["-m", "b4dl_core.reasoning", "infer", *common, *model_args(model), "--gpu-lock", plan["gpu_lock"],
                    "--split", "internal_dev", "--condition", condition, "--max-new-tokens", "512", "--seed", "42", "--output", str(output)]
            stages.append({"name": f"control-{name}-{condition}", "model": name, "kind": "geometry-infer", "output": str(output), "args": args})
            report = output.parent / f"{condition}-report.json"
            stages.append({"name": f"score-control-{name}-{condition}", "model": name, "kind": "geometry-score",
                "output": str(report), "predictions": str(output / "predictions.jsonl"),
                "args": ["-m", "b4dl_core.reasoning", "evaluate", *common, "--split", "internal_dev",
                         "--predictions", str(output / "predictions.jsonl"), "--output", str(report)]})
            comparison = queue / "comparisons" / f"control-{name}-{condition}.json"
            stages.append({"name": f"compare-control-{name}-{condition}", "model": name, "kind": "comparison", "output": str(comparison),
                "args": ["-m", "b4dl_core.reasoning", "compare", "--intervention", "--base", str(output.parent / "identity-report.json"),
                         "--candidate", str(report), "--output", str(comparison)]})
    # A comparison can legitimately fail its strict lineage gate. Finish every
    # individual model evaluation before any such failure can stop the queue.
    return [stage for stage in stages if stage["kind"] != "comparison"] + [
        stage for stage in stages if stage["kind"] == "comparison"
    ]


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan, queue = read(args.plan), args.output.resolve()
    if plan.get("schema") != "b4dl-endpoint-evaluation-plan-v1":
        raise ValueError("unknown evaluation plan")
    if {m["name"] for m in plan["models"]} != {"s1", "s2", "t0", "b3", "b3_t0"}:
        raise ValueError("evaluation plan must contain the frozen endpoints and matching B3 references")
    plan_hash = sha(args.plan)
    if args.resume:
        status = read(queue / "status.json")
        if status["plan_sha256"] != plan_hash or status["status"] == "complete":
            raise ValueError("resume plan changed or queue is already complete")
        if alive(status.get("pid", -1)) or status.get("stage_pid") and alive(status["stage_pid"]):
            raise RuntimeError("queue or its previous stage is still running")
        status.update(status="running", pid=os.getpid(), stage_pid=None, error=None)
    else:
        queue.mkdir(parents=True, exist_ok=False)
        status = {"schema": "b4dl-endpoint-evaluation-queue-v1", "status": "running", "plan_sha256": plan_hash,
                  "pid": os.getpid(), "stage": "preflight", "stage_pid": None, "completed": [], "error": None,
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(queue / "status.json", status)
    model_map = {model["name"]: model for model in plan["models"]}
    try:
        with (queue / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            endpoints = {model["name"]: check_endpoint(model) for model in plan["models"]}
            if not (queue / "endpoint-verification.json").exists():
                write(queue / "endpoint-verification.json", endpoints)
            stages = build_stages(plan, queue)
            for stage in stages:
                model = model_map[stage["model"]]
                if stage["name"] in status["completed"]:
                    validate_stage(stage, model)
                    continue
                output = Path(stage["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", *stage["args"]]
                if stage["kind"] in ("geometry-infer", "six-infer") and output.exists():
                    if not (output / "run.json").is_file():
                        raise ValueError("existing inference output has no run identity: " + str(output))
                    command.append("--resume")
                elif output.exists():
                    # A process may have completed just before its queue status was written.
                    validate_stage(stage, model)
                    status["completed"].append(stage["name"])
                    status.update(stage_pid=None, last_completed=stage["name"])
                    write(queue / "status.json", status)
                    continue
                env = dict(os.environ, PYTHONPATH=model["source"] + "/mllm:" + plan["metric_backend_root"],
                    PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                log_path = stage_log(queue, stage["name"])
                status.update(stage=stage["name"], stage_pid=None, command=command, log=str(log_path),
                              current_output=str(output), stage_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                write(queue / "status.json", status)
                with log_path.open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=model["source"], env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    status["stage_pid"] = process.pid
                    write(queue / "status.json", status)
                    try:
                        while process.poll() is None:
                            time.sleep(10)
                            if stage["kind"] == "six-infer" and (output / "predictions.jsonl").is_file():
                                try:
                                    with (output / "predictions.jsonl").open(encoding="utf-8") as stream:
                                        first = [json.loads(line) for _, line in zip(range(20), stream)]
                                except json.JSONDecodeError:
                                    continue  # A concurrent append may not yet contain its final newline.
                                if len(first) == 20 and all(row.get("status") == "error" for row in first):
                                    raise RuntimeError("first 20 six-task predictions failed; halted systemic error")
                    except BaseException:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        raise
                    if process.returncode:
                        raise RuntimeError(stage["name"] + " exited " + str(process.returncode))
                validate_stage(stage, model)
                status["completed"].append(stage["name"])
                status.update(stage_pid=None, last_completed=stage["name"])
                write(queue / "status.json", status)
            status.update(status="complete", stage="complete", stage_pid=None,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(queue / "status.json", status)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error), stage_pid=None,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(queue / "status.json", status)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
