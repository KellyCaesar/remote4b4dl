"""Wait for the frozen evaluation; run S3 and matched mixed-SFT controls."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from b4dl_core.io import digest, read_json, sha256, write_json
from b4dl_core.reasoning.__main__ import recipe_for
from b4dl_core.reasoning.evaluation import validate_ablation_lineage
from b4dl_core.reasoning.trainer import implementation_identity
from b4dl_core.regional_model import RegionalConfig
from queue_regional_s2 import process_alive, verify_training
from run_upcore_evaluation_queue import alive, build_stages, inference_receipt, validate_stage


def config(path):
    data = read_json(path)
    if data.pop("schema", None) != "b4dl-regional-config-v1":
        raise ValueError("invalid regional config")
    return RegionalConfig(**data).as_dict()


def train_command(plan, output, regional_config=None, objective=None, smoke=False):
    cmd = [sys.executable, "-u", "-m", "b4dl_core.reasoning", "sft", "--dataset", plan["dataset"],
           "--geometry-dataset", plan["geometry"], "--baseline", plan["baseline"],
           "--preparation", plan["preparation"], "--recipe", plan["recipe"],
           "--gpu-lock", plan["gpu_lock"], "--output", str(output)]
    if regional_config:
        regional = plan["small_regional"] if smoke and objective is None else plan["regional"]
        cmd += ["--regional-features", regional, "--regional-config", str(regional_config)]
    if objective:
        cmd += ["--mixed-plan", plan["mixed_plan"], "--supervision-config", str(objective)]
    if smoke:
        cmd += ["--engineering", "--max-updates", str(plan["smoke_updates"])]
        if regional_config and objective is None:
            cmd.append("--engineering-region-subset")
    return cmd


def upstream_alive(pid, plan_path, output):
    if not alive(pid) or not process_alive(pid, "run_upcore_evaluation_queue.py"):
        return False
    try:
        process = Path("/proc") / str(pid)
        args = process.joinpath("cmdline").read_bytes().decode().split("\0")
        cwd = process.joinpath("cwd").resolve(strict=True)
        actual_plan = (cwd / args[args.index("--plan") + 1]).resolve()
        actual_output = (cwd / args[args.index("--output") + 1]).resolve()
        return actual_plan == Path(plan_path).resolve() and actual_output == Path(output).resolve()
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return False


def truncation_expected(plan, upstream, stages):
    """Return the frozen stop point when an operator-approved truncation is recorded."""
    stop = plan.get("evaluation_stop_after")
    if not stop:
        return None
    names = [row["name"] for row in stages]
    if stop not in names:
        raise ValueError("unknown evaluation stop point")
    expected = names[:names.index(stop) + 1]
    path = Path(plan["evaluation_output"]) / "truncation.json"
    if not path.is_file():
        return None
    receipt = read_json(path)
    if (receipt.get("schema") != "b4dl-evaluation-truncation-v1"
            or receipt.get("reason") != "user_approved_scope_reduction"
            or receipt.get("stopped_after") != stop
            or receipt.get("stage_list") != expected
            or receipt.get("plan_sha256") != plan["evaluation_plan_sha256"]):
        raise ValueError("truncated evaluation receipt mismatch")
    if upstream.get("completed") != expected:
        raise ValueError("truncated evaluation stage list mismatch")
    return expected


def validate_upstream(plan):
    path = Path(plan["evaluation_plan"])
    output = Path(plan["evaluation_output"])
    if sha256(path) != plan["evaluation_plan_sha256"]:
        raise ValueError("evaluation plan changed")
    upstream = read_json(output / "status.json")
    if upstream["plan_sha256"] != plan["evaluation_plan_sha256"]:
        raise ValueError("evaluation status plan mismatch")
    truncated = plan.get("evaluation_stop_after") and (output / "truncation.json").is_file()
    if upstream["status"] == "running" and not truncated:
        pid = upstream.get("pid")
        if not upstream_alive(pid, path, plan["evaluation_output"]):
            raise RuntimeError("evaluation process missing while marked running")
        return False
    if upstream["status"] in ("failed", "stopped") and not plan.get("evaluation_stop_after"):
        raise RuntimeError("evaluation did not complete: " + str(upstream.get("error")))
    if upstream["status"] not in ("running", "complete", "failed", "stopped"):
        raise RuntimeError("evaluation did not complete: " + str(upstream.get("error")))
    evaluation = read_json(path)
    stages = build_stages(evaluation, output)
    if upstream["status"] == "complete":
        expected = [row["name"] for row in stages]
        if upstream["completed"] != expected:
            raise ValueError("evaluation stage list is incomplete or changed")
    else:
        expected = truncation_expected(plan, upstream, stages)
        if expected is None:
            raise RuntimeError("evaluation did not complete: " + str(upstream.get("error")))
        if upstream["status"] == "running" and (
                upstream_alive(upstream.get("pid"), path, plan["evaluation_output"])
                or alive(upstream.get("stage_pid"))):
            raise RuntimeError("truncated evaluation is still running")
    models = {row["name"]: row for row in evaluation["models"]}
    by_name = {row["name"]: row for row in stages}
    for name in expected:
        stage = by_name[name]
        validate_stage(stage, models[stage["model"]])
    return True


def report_score(report):
    tasks = report["by_task"]
    if set(tasks) != {"existence", "position", "distance", "motion"}:
        raise ValueError("missing geometry tasks")
    return (sum(tasks[k]["whole_trace_pass_rate"] for k in tasks) / 4,
            sum(tasks[k]["accuracy"] for k in tasks) / 4)


def choose_structure(reports, historical):
    # Frozen selection rule: complete-chain macro, then answer macro, then
    # lower structural cost in an exact tie (S1, S2, S3).
    order = ("s1", "s2", "s3")
    winner = max(order, key=lambda name: (*report_score(reports[name]), -order.index(name)))
    candidate, baseline = reports[winner]["summary"], historical["summary"]
    accuracy_gain = candidate["accuracy"] - baseline["accuracy"]
    trace_gain = candidate["whole_trace_pass_rate"] - baseline["whole_trace_pass_rate"]
    admitted = accuracy_gain >= 0 and trace_gain >= 0 and max(accuracy_gain, trace_gain) >= 0.01
    return {"selected": winner, "scores": {name: report_score(reports[name]) for name in order},
            "historical_accuracy_gain": accuracy_gain, "historical_trace_gain": trace_gain,
            "geometry_screen_passed": admitted,
            "note": "development screening only; historical source differs; final six-task and test gates remain"}


def verified_report(predictions, report, expected):
    receipt = inference_receipt(predictions, "reasoning_inference_complete")
    result = read_json(report)
    if (receipt["requests"] != expected or not result.get("complete_prediction_set")
            or result.get("run") != receipt["run"] or result["summary"]["n"] != expected):
        raise ValueError("geometry report/receipt mismatch")
    return result


def validate_screen_reports(reports, historical):
    keys = ("dataset_id", "split", "seed", "max_new_tokens", "decoding", "reward_version", "runtime", "device",
            "dtype", "numerics", "gpu_name")
    anchor = reports["s3"]
    identities = {row["request_id"]: (row["scene_id"], row["task"], row["truth"]) for row in anchor["rows"]}
    if len(identities) != anchor["summary"]["n"]:
        raise ValueError("duplicate or missing screening rows")
    for name, report in {**reports, "historical": historical}.items():
        actual = {row["request_id"]: (row["scene_id"], row["task"], row["truth"]) for row in report["rows"]}
        if len(actual) != report["summary"]["n"] or identities != actual:
            raise ValueError("screening question or truth differs: " + name)
        run_keys = keys if name != "historical" else keys[:5]
        if any(report["run"].get(key) != anchor["run"].get(key) for key in run_keys):
            raise ValueError("screening runtime or input identity differs: " + name)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="check inputs and upstream without GPU work")
    args = parser.parse_args()
    plan = read_json(args.plan)
    if plan.get("schema") != "b4dl-s3-followup-plan-v1" or plan["updates"] != 13851 or plan["smoke_updates"] != 50:
        raise ValueError("invalid follow-up plan")
    root = args.output.resolve()
    root.mkdir(parents=False, exist_ok=False)
    source_config_dir = Path(plan["source"]) / "mllm" / "configs"
    objective_names = ("core_mixed_t0_t1_v1.json", "core_mixed_t2_v1.json", "core_mixed_t3_v1.json",
                       "core_mixed_t3_no_replay_v1.json")
    inputs = [Path(plan[key]) for key in ("baseline", "preparation", "recipe", "s1_config", "s2_config",
                                          "s3_config", "regional_validation")]
    inputs += [Path(plan["regional"]) / "manifest.json", Path(plan["dataset"]) / "manifest.json",
               Path(plan["geometry"]) / "manifest.json", Path(plan["small_regional"]) / "manifest.json",
               *(Path(plan["mixed_plan"]) / name for name in ("manifest.json", "plan.jsonl", "replay.jsonl")),
               *(source_config_dir / name for name in objective_names)]
    bound_inputs = {str(path): sha256(path) for path in inputs}
    status = {"schema": "b4dl-s3-followup-status-v1", "status": "running", "stage": "preflight",
              "pid": os.getpid(), "stage_pid": None, "completed": [], "error": None,
              "plan_sha256": sha256(args.plan), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def save(**fields):
        status.update(fields)
        write_json(root / "status.json", status)

    def perform(name, cmd, env):
        if shutil.disk_usage(root).free < plan["minimum_free_gib"] * 1024**3:
            raise RuntimeError("insufficient disk reserve for next stage")
        if implementation_identity() != plan["source_id"] or sha256(args.plan) != status["plan_sha256"]:
            raise ValueError("source or plan changed during queue")
        if any(sha256(path) != expected for path, expected in bound_inputs.items()):
            raise ValueError("a bound training input changed during queue")
        log = root / (name + ".log")
        save(stage=name, command=cmd, stage_pid=None, log=str(log))
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(cmd, cwd=plan["source"], env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save(stage_pid=process.pid)
            code = process.wait()
        save(stage_pid=None)
        if code:
            raise RuntimeError(name + " exited with code " + str(code))
        status["completed"].append(name)
        save(last_completed=name)

    save()
    write_json(root / "bound-inputs.json", bound_inputs)
    try:
        with Path(plan["controller_lock"]).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source = Path(plan["source"])
            if implementation_identity() != plan["source_id"]:
                raise ValueError("wrong frozen source")
            s2 = verify_training(plan["s2"], "training_complete", plan["updates"])
            raw_s2 = config(plan["s2_config"])
            raw_s3 = config(plan["s3_config"])
            if (s2["identity"]["common"]["implementation"] != plan["source_id"]
                    or s2["identity"]["common"]["options"]["regional"] != raw_s2
                    or raw_s2["use_geometry"] or raw_s3 != {**raw_s2, "use_geometry": True}
                    or s2["identity"]["recipe"] != recipe_for(plan["recipe"])):
                raise ValueError("S3 must match S2 except geometry")
            manifest = read_json(Path(plan["regional"]) / "manifest.json")
            validation = read_json(plan["regional_validation"])
            common = s2["identity"]["common"]
            if (manifest.get("dataset_id") != digest({k: v for k, v in manifest.items() if k != "dataset_id"})
                    or not manifest.get("complete") or validation.get("status") != "validated"
                    or validation.get("regional_manifest_sha256") != sha256(Path(plan["regional"]) / "manifest.json")
                    or manifest["dataset_id"] != common["regional_dataset_id"]
                    or sha256(plan["baseline"]) != common["baseline_sha256"]
                    or sha256(plan["preparation"]) != common["preparation_sha256"]):
                raise ValueError("training assets do not match S2")
            if args.preflight_only:
                upstream_complete = validate_upstream(plan)
                save(status="preflight_complete", stage="preflight_complete", upstream_complete=upstream_complete)
                return
            save(stage="waiting_for_evaluation")
            while not validate_upstream(plan):
                time.sleep(plan["poll_seconds"])
            save(stage="evaluation_verified")
            env = dict(os.environ, PYTHONPATH=str(source / "mllm"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            for smoke in (True, False):
                name = "s3-smoke" if smoke else "s3-formal"
                perform(name, train_command(plan, root / name, plan["s3_config"], smoke=smoke), env)
                result = verify_training(root / name, "engineering_budget_complete" if smoke else "training_complete",
                                         plan["smoke_updates"] if smoke else plan["updates"])
                if (result["identity"]["common"]["implementation"] != plan["source_id"]
                        or result["identity"]["common"]["options"]["regional"] != raw_s3
                        or result["identity"]["common"]["regional_dataset_id"] !=
                           (read_json(Path(plan["small_regional"]) / "manifest.json")["dataset_id"] if smoke
                            else common["regional_dataset_id"])):
                    raise ValueError("S3 endpoint identity mismatch")
                write_json(root / (name + "-verification.json"), result)
            s3ckpt = root / "s3-formal" / ("step-" + str(plan["updates"]) + ".pt")
            pred, report = root / "s3-dev", root / "s3-dev-report.json"
            common_args = ["--dataset", plan["dataset"], "--geometry-dataset", plan["geometry"]]
            perform("s3-dev-infer", [sys.executable, "-u", "-m", "b4dl_core.reasoning", "infer", *common_args,
                "--baseline", plan["baseline"], "--preparation", plan["preparation"], "--checkpoint", str(s3ckpt),
                "--regional-features", plan["regional"], "--gpu-lock", plan["gpu_lock"], "--split", "internal_dev",
                "--condition", "identity", "--max-new-tokens", "512", "--seed", "42", "--output", str(pred)], env)
            perform("s3-dev-score", [sys.executable, "-u", "-m", "b4dl_core.reasoning", "evaluate", *common_args,
                "--split", "internal_dev", "--predictions", str(pred / "predictions.jsonl"), "--output", str(report)], env)
            s3_report = verified_report(pred, report, 13602)
            reports = {"s3": s3_report}
            for name in ("s1", "s2"):
                parent = Path(plan["evaluation_output"]) / "geometry" / name / "internal_dev"
                reports[name] = verified_report(parent / "identity", parent / "identity-report.json", 13602)
                validate_ablation_lineage(reports[name]["run"], s3_report["run"])
            historical = verified_report(plan["historical_sft_predictions"], plan["historical_sft_report"], 13602)
            validate_screen_reports(reports, historical)
            selection = choose_structure(reports, historical)
            write_json(root / "structure-selection.json", selection)
            if not selection["geometry_screen_passed"]:
                save(status="needs_review", stage="geometry_screen_not_passed", selection=selection)
                return
            chosen_config = plan[selection["selected"] + "_config"]
            save(stage="mixed_pipeline", selection=selection)
            configs = source / "mllm" / "configs"
            arms = (("t0-matched", None, "core_mixed_t0_t1_v1.json"),
                    ("t1", chosen_config, "core_mixed_t0_t1_v1.json"),
                    ("t2", chosen_config, "core_mixed_t2_v1.json"),
                    ("t3", chosen_config, "core_mixed_t3_v1.json"),
                    ("t3-no-replay", chosen_config, "core_mixed_t3_no_replay_v1.json"))
            for name, regional_config, objective_name in arms:
                objective = configs / objective_name
                for smoke in (True, False):
                    label = name + ("-smoke" if smoke else "-formal")
                    perform(label, train_command(plan, root / label, regional_config, objective, smoke), env)
                    result = verify_training(root / label, "engineering_budget_complete" if smoke else "training_complete",
                                             plan["smoke_updates"] if smoke else plan["updates"])
                    ident = result["identity"]["common"]
                    if (ident["implementation"] != plan["source_id"]
                            or ident.get("mixed_plan_id") != read_json(Path(plan["mixed_plan"]) / "manifest.json")["plan_id"]
                            or ident["options"].get("supervision") != read_json(objective)
                            or ident["baseline_sha256"] != common["baseline_sha256"]
                            or ident["preparation_sha256"] != common["preparation_sha256"]
                            or ident["reasoning_dataset_id"] != common["reasoning_dataset_id"]
                            or regional_config and ident.get("regional_dataset_id") != common["regional_dataset_id"]
                            or regional_config and ident["options"].get("regional") != config(regional_config)):
                        raise ValueError(label + " training identity mismatch")
                    write_json(root / (label + "-verification.json"), result)
            save(status="training_complete", stage="mixed_training_complete", stage_pid=None,
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception as error:
        save(status="failed", error_type=type(error).__name__, error=str(error), stage_pid=None)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
