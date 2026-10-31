"""Wait for matched SFT/B3 evaluation, then run fixed development diagnostics.

Uses only the standard library; model work runs through the frozen remote CLI.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


CONDITIONS = ("identity", "zero", "repeat_first", "shuffle_features", "cross_scene")
PRIMARY = {"existence": "accuracy", "binary_qa": "accuracy", "time_grounding": "miou",
           "description": "meteor", "temporal_understanding": "meteor", "comprehensive_reasoning": "meteor"}
STOP_REQUESTED = False


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment(plan):
    return dict(os.environ, PYTHONPATH=str(Path(plan["run_root"]) / "mllm"),
                PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")


def preflight(plan, check_implementation=True):
    require(plan["schema"] == "b4dl-post-eval-plan-v1", "unknown plan")
    require(plan["conditions"] == list(CONDITIONS), "fixed diagnostic conditions changed")
    require(plan["split"] == "internal_dev", "follow-up must use development data")
    require(plan["max_new_tokens"] == 512 and plan["seed"] == 42, "fixed decoding changed")
    for key in ("poll_seconds", "wait_timeout_hours", "stage_timeout_hours", "execution_timeout_hours"):
        require(type(plan[key]) in (int, float) and math.isfinite(plan[key]) and plan[key] > 0, "invalid " + key)
    root, out, evaluation = (Path(plan[k]).resolve() for k in ("run_root", "output", "evaluation"))
    require(out.parent == root and out != evaluation and out != root, "output must be a new direct run-root child")
    require(not out.is_symlink(), "output may not be a symlink")
    for key in ("dataset", "geometry_dataset", "checkpoint", "protocol", "baseline", "preparation", "python"):
        asset = Path(plan[key]).resolve()
        require(asset.exists() and out != asset and out not in asset.parents, "output contains or replaces an input: " + key)
    for key in ("checkpoint", "baseline", "preparation"):
        require(sha(plan[key]) == plan[key + "_sha256"], key + " hash changed")
    manifest = read(Path(plan["dataset"]) / "manifest.json")
    require(manifest["dataset_id"] == plan["dataset_id"], "dataset identity changed")
    for name, expected in manifest["files"].items():
        require(sha(Path(plan["dataset"]) / name) == expected, "dataset file changed: " + name)
    protocol = read(plan["protocol"])
    require(protocol["protocol_id"] == plan["protocol_id"] and protocol["requests"] == plan["six_requests"], "six-task protocol changed")
    require(protocol["input_protocol"]["oracle_assisted_metadata"] is False, "oracle input is not allowed")
    require(sha(Path(plan["protocol"]).parent / "requests.jsonl") == protocol["requests_sha256"], "six-task requests changed")
    training = read(Path(plan["checkpoint"]).parent / "completion.json")
    require(training["status"] == "training_complete" and training["full_budget"] is True
            and training["checkpoint_sha256"] == plan["checkpoint_sha256"]
            and training["updates"] == 13851, "SFT endpoint is not the completed fixed budget")
    if check_implementation:
        value = subprocess.check_output([plan["python"], "-c",
            "from b4dl_core.reasoning.trainer import implementation_identity; print(implementation_identity())"],
            cwd=root, env=environment(plan), text=True, timeout=180).strip().splitlines()[-1]
        require(value == plan["implementation"], "frozen model implementation changed")
    request_rows = list(rows(Path(plan["dataset"]) / "requests.jsonl"))
    result = {split: {r["request_id"]: r["input_fingerprint"] for r in request_rows if r["split"] == split}
              for split in ("test", "internal_dev")}
    for split, expected in (("test", plan["test_requests"]), ("internal_dev", plan["dev_requests"])):
        require(len(result[split]) == expected, "request count differs: " + split)
    result["six"] = {r["request_id"]: r["input_fingerprint"] for r in rows(Path(plan["protocol"]).parent / "requests.jsonl")}
    require(len(result["six"]) == plan["six_requests"], "six-task IDs differ")
    return result


def verify_predictions(directory, expected, plan, *, condition=None, split=None, arm=None):
    directory = Path(directory)
    receipt, run = read(directory / "completion.json"), read(directory / "run.json")
    kind = "six_task_inference_complete" if arm else "reasoning_inference_complete"
    require(receipt["status"] == kind and receipt["run"] == run and receipt["requests"] == len(expected), "invalid inference receipt")
    require(sha(directory / "predictions.jsonl") == receipt["predictions_sha256"], "prediction hash mismatch")
    seen = set()
    for row in rows(directory / "predictions.jsonl"):
        ident = row["request_id"]
        require(ident not in seen and ident in expected and row["input_fingerprint"] == expected[ident], "prediction IDs/fingerprints differ")
        seen.add(ident)
    require(seen == set(expected), "incomplete prediction denominator")
    if arm:
        require(run["protocol_id"] == plan["protocol_id"] and run["oracle_assisted_metadata"] is False, "six-task input identity changed")
        require(run["model_kind"] == ("reasoning" if arm == "sft" else "original_b3"), "incorrect model arm")
        artifact = plan["checkpoint_sha256"] if arm == "sft" else plan["baseline_sha256"]
        require(run["model_artifact_sha256"] == artifact, "incorrect model artifact")
        for key in ("baseline_sha256", "preparation_sha256", "implementation"):
            require(run[key] == plan[key], "six-task identity changed: " + key)
    else:
        require(run["condition"] == condition and run["split"] == split and run["dataset_id"] == plan["dataset_id"]
                and run["checkpoint_sha256"] == plan["checkpoint_sha256"]
                and run["max_new_tokens"] == plan["max_new_tokens"] and run["seed"] == plan["seed"]
                and run["decoding"] == "greedy", "geometry run changed")
    return receipt


def verify_report(path, expected, plan, condition, split):
    report = read(path)
    require(report["complete_prediction_set"] is True and report["summary"]["n"] == len(expected), "incomplete report")
    require(len(report["rows"]) == len(expected) and {r["request_id"] for r in report["rows"]} == set(expected), "report IDs differ")
    run = report["run"]
    require(run["checkpoint_sha256"] == plan["checkpoint_sha256"] and run["dataset_id"] == plan["dataset_id"]
            and run["condition"] == condition and run["split"] == split, "report identity differs")
    return report


def predecessor_ready(plan):
    evaluation = Path(plan["evaluation"])
    geo, six = read(evaluation / "status.json"), read(evaluation / "six-task/status.json")
    for state in (geo, six):
        require(state.get("status") not in ("failed", "evaluation_failed"), "predecessor evaluation failed: " + str(state.get("message")))
        require(state["checkpoint_sha256"] == plan["checkpoint_sha256"], "predecessor checkpoint changed")
    require(six["protocol_id"] == plan["protocol_id"], "predecessor protocol changed")
    if six["status"] == "six_task_comparison_complete":
        require(geo["status"] == "geometry_evaluation_complete", "six tasks completed before geometry")
        require({s["name"] for s in geo["completed_stages"] if s["returncode"] == 0}
                == {"identity-infer", "identity-evaluate", "zero-infer", "zero-evaluate", "intervention-compare"},
                "geometry stages incomplete")
        expected = {"sft-infer", "sft-metrics", "b3-infer", "b3-metrics", "compare"}
        require({s["stage"] for s in six["completed_stages"] if s["exit_code"] == 0} == expected, "six-task stages incomplete")
        return True
    proc = Path("/proc") / str(six["pid"])
    command = (proc / "cmdline").read_bytes().split(b"\0") if proc.exists() else []
    script = str(Path(plan["run_root"]) / "evaluate_sft_six_tasks.py").encode()
    require(script in command, "predecessor controller disappeared or PID was reused")
    state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
    require(state not in ("Z", "X"), "predecessor controller is dead")
    return False


def error_audit(report):
    counts, examples = defaultdict(Counter), defaultdict(list)
    for row in sorted(report["rows"], key=lambda r: r["request_id"]):
        if not row["valid"]:
            bucket = "invalid_structure_or_generation"
        elif not row["correct"]:
            bucket = "wrong_answer"
        elif not all(score > 0 for score in row["step_scores"]):
            bucket = "correct_answer_unsupported_trace"
        else:
            bucket = "grounded_correct"
        counts[row["task"]][bucket] += 1
        key = row["task"] + "/" + bucket
        if bucket != "grounded_correct" and len(examples[key]) < 8:
            examples[key].append({k: row[k] for k in ("request_id", "scene_id", "answer", "truth", "status", "step_scores", "absolute_errors")})
    return {"split": report["run"]["split"], "counts": dict(counts), "examples": dict(examples),
            "example_selection": "first 8 request IDs per task/error bucket; illustrative, not a representative sample"}


def recommendations(comparison, dev_report, controls):
    decreases = []
    for task, metric in PRIMARY.items():
        value = comparison["per_task_metrics"][task][metric]
        require(all(type(value[k]) in (int, float) and math.isfinite(value[k]) for k in ("base", "candidate", "delta")), "missing primary metric")
        if value["delta"] < 0:
            decreases.append({"task": task, "metric": metric, **value})
    weak = sorted(({"task": task, "whole_trace_pass_rate": v["whole_trace_pass_rate"], "n": v["n"]}
                   for task, v in dev_report["by_task"].items()), key=lambda v: v["whole_trace_pass_rate"])
    return {"priority": "six_task_retention" if decreases else "grounded_reasoning",
            "six_task_directional_decreases": decreases, "dev_trace_priority": weak,
            "control_evidence": controls,
            "next_candidate": "六任务原训练监督回放与几何监督混合；在开发集选择配比" if decreases
                              else "按开发集错误补强几何观测与过程监督，分别比较答案奖励和过程奖励",
            "grpo_gate": "blocked_unresolved_sampling_recompute_logprob_mismatch",
            "training_launched": False,
            "limits": ["六任务下降仅为描述性方向，不是显著性或预注册验收阈值",
                       "开发集可能被历史B3见过；本次新诊断使用开发集，不用测试答案构造训练样本",
                       "置零可能引起格式崩溃；结合有效率、可用配对及其他输入干预解释",
                       "保留自车位姿的置零不是完全无传感器输入；范围仍为CAM_BACK_RIGHT裁剪",
                       "GRPO须先定位概率不一致根因并完成独立数值验收"]}


@contextmanager
def queue_lock(path):
    import fcntl
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle.fileno()


class Queue:
    def __init__(self, plan, state, lock_fd):
        self.plan, self.state, self.lock_fd = plan, state, lock_fd
        self.out = Path(plan["output"])
        self.started = time.monotonic()
        self.execution_started = None

    def save(self, **changes):
        self.state.update(changes, updated_utc=now())
        write(self.out / "status.json", self.state)

    def check_stop(self):
        if STOP_REQUESTED or (self.out / "STOP").exists():
            raise InterruptedError("stop requested")
        if self.execution_started is not None and time.monotonic() - self.execution_started > self.plan["execution_timeout_hours"] * 3600:
            raise TimeoutError("follow-up execution deadline exceeded")

    def stage(self, name, args, outputs, inference=False):
        self.check_stop()
        previous = self.state["completed_stages"].get(name)
        if previous:
            require(previous["args"] == args, "completed stage command changed")
            require(all(Path(p).is_file() and sha(p) == h for p, h in previous["files"].items()), "completed stage outputs changed")
            return
        actual = list(args)
        if inference and Path(args[args.index("--output") + 1]).exists():
            require(self.state["resumed"], "inference output already exists")
            actual.append("--resume")
        command = [self.plan["python"], "-m", "b4dl_core.reasoning"] + actual
        log_path = self.out / (name + ".log")
        self.save(status="running", stage=name, command=command)
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=self.plan["run_root"], env=environment(self.plan),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(self.lock_fd,))
            self.save(stage_pid=process.pid, stage_started_utc=now())
            begin = time.monotonic()
            try:
                while process.poll() is None:
                    self.check_stop()
                    if time.monotonic() - begin > self.plan["stage_timeout_hours"] * 3600:
                        raise TimeoutError("stage timed out: " + name)
                    self.save()
                    time.sleep(min(self.plan["poll_seconds"], 5))
                require(process.returncode == 0, name + " failed; see " + str(log_path))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                self.state.pop("stage_pid", None)
        files = {str(p): sha(p) for p in outputs}
        self.state["completed_stages"][name] = {"args": args, "files": files, "finished_utc": now()}
        self.save()

    def run(self):
        self.save(status="waiting_for_evaluation", stage=None)
        while not predecessor_ready(self.plan):
            self.check_stop()
            if time.monotonic() - self.started > self.plan["wait_timeout_hours"] * 3600:
                raise TimeoutError("evaluation wait deadline exceeded")
            self.save()
            time.sleep(min(self.plan["poll_seconds"], 30))
        self.execution_started = time.monotonic()
        expected = preflight(self.plan)
        ev = Path(self.plan["evaluation"])
        evidence = {}
        for condition in ("identity", "zero"):
            verify_predictions(ev / ("test-" + condition), expected["test"], self.plan, condition=condition, split="test")
            report = verify_report(ev / ("test-" + condition + "-report.json"), expected["test"], self.plan, condition, "test")
            require(report["run"] == read(ev / ("test-" + condition) / "run.json"), "report and predictions differ")
            evidence[condition] = {k: v for k, v in report.items() if k != "rows"}
        intervention = read(ev / "test-zero-comparison.json")
        require(intervention["complete_prediction_sets"] is True and intervention["n"] == len(expected["test"])
                and intervention["base_run"] == evidence["identity"]["run"]
                and intervention["candidate_run"] == evidence["zero"]["run"], "test intervention comparison differs")
        evidence["zero_intervention"] = {k: v for k, v in intervention.items() if k != "rows"}
        for arm in ("sft", "b3"):
            directory = ev / "six-task" / (arm + "-predictions")
            pred = verify_predictions(directory, expected["six"], self.plan, arm=arm)
            metrics = ev / "six-task" / (arm + "-metrics")
            receipt = read(metrics / "completion.json")
            require(receipt["status"] == "six_task_metrics_complete" and receipt["requests"] == len(expected["six"])
                    and receipt["protocol_id"] == self.plan["protocol_id"]
                    and receipt["predictions_sha256"] == pred["predictions_sha256"]
                    and receipt["inference_run"] == pred["run"]
                    and receipt["metrics_sha256"] == sha(metrics / "metrics.json"), "metric receipt mismatch")
            evidence[arm + "_metrics"] = read(metrics / "metrics.json")
            evidence[arm + "_receipt"] = receipt
        evidence["artifact_hashes"] = {str(p): sha(p) for p in ev.glob("test-*-report.json")}
        write(self.out / "endpoint-summary.json", evidence)
        comparison_path = self.out / "six-task-comparison.json"
        self.stage("verify-six-comparison", ["six-task-compare", "--base", str(ev / "six-task/b3-metrics"),
            "--candidate", str(ev / "six-task/sft-metrics"), "--output", str(comparison_path)], [comparison_path])
        comparison = read(comparison_path)
        require(comparison == read(ev / "six-task/comparison.json"), "upstream comparison differs from verified metrics")
        dataset_args = ["--dataset", self.plan["dataset"], "--geometry-dataset", self.plan["geometry_dataset"]]
        reports, controls = {}, {}
        for condition in self.plan["conditions"]:
            directory = self.out / ("dev-" + condition)
            self.stage(condition + "-infer", ["infer"] + dataset_args + ["--baseline", self.plan["baseline"],
                "--preparation", self.plan["preparation"], "--checkpoint", self.plan["checkpoint"],
                "--split", "internal_dev", "--condition", condition, "--seed", str(self.plan["seed"]),
                "--max-new-tokens", str(self.plan["max_new_tokens"]), "--output", str(directory)],
                [directory / n for n in ("completion.json", "run.json", "predictions.jsonl")], inference=True)
            verify_predictions(directory, expected["internal_dev"], self.plan, condition=condition, split="internal_dev")
            report_path = self.out / ("dev-" + condition + "-report.json")
            self.stage(condition + "-evaluate", ["evaluate"] + dataset_args + ["--split", "internal_dev",
                "--predictions", str(directory / "predictions.jsonl"), "--output", str(report_path)], [report_path])
            reports[condition] = verify_report(report_path, expected["internal_dev"], self.plan, condition, "internal_dev")
            if condition != "identity":
                pair_path = self.out / ("dev-" + condition + "-comparison.json")
                self.stage(condition + "-compare", ["compare", "--base", str(self.out / "dev-identity-report.json"),
                    "--candidate", str(report_path), "--intervention", "--output", str(pair_path)], [pair_path])
                paired = read(pair_path)
                controls[condition] = {k: paired[k] for k in ("n", "accuracy_delta", "available_pairs", "by_task")}
                controls[condition]["valid_rate"] = reports[condition]["summary"]["valid_rate"]
            write(self.out / "dev-progress-summary.json", {k: {"summary": v["summary"], "by_task": v["by_task"]} for k, v in reports.items()})
        audit_path = self.out / "dev-error-audit.json"
        write(audit_path, error_audit(reports["identity"]))
        decision = recommendations(comparison, reports["identity"], controls)
        write(self.out / "next-step.json", decision)
        lines = ["# SFT 评测后诊断", "", "完成时间（UTC）：" + now(), "",
                 "全部开发集对照完成。诊断建议：" + decision["next_candidate"], "",
                 "## 六任务 SFT 相对 B3", "", "|任务|指标|B3|SFT|差值|", "|---|---|---:|---:|---:|"]
        for task, metric in PRIMARY.items():
            value = comparison["per_task_metrics"][task][metric]
            lines.append(f"|{task}|{metric}|{value['base']:.6f}|{value['candidate']:.6f}|{value['delta']:+.6f}|")
        lines += ["", "## 开发集输入对照", "", "|输入|题数|准确率|结构有效率|整链通过率|", "|---|---:|---:|---:|---:|"]
        for condition, report in reports.items():
            s = report["summary"]
            lines.append(f"|{condition}|{s['n']}|{s['accuracy']:.4%}|{s['valid_rate']:.4%}|{s['whole_trace_pass_rate']:.4%}|")
        lines += ["", "开发集错误分类及示例见 dev-error-audit.json；配对分母及场景区间见 dev-*-comparison.json。", "",
                  "## 结论边界", ""] + ["- " + item for item in decision["limits"]]
        (self.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(self.out / "completion.json", {"status": "development_diagnostics_complete", "plan": self.plan,
            "finished_utc": now(), "completed_stages": self.state["completed_stages"],
            "artifacts": {name: sha(self.out / name) for name in
                ("endpoint-summary.json", "six-task-comparison.json", "dev-error-audit.json", "next-step.json", "report.md")}})
        self.save(status="completed", stage=None, finished_utc=now(), training_launched=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    plan = read(args.plan)
    counts = preflight(plan)
    if args.check_only:
        print(json.dumps({"status": "preflight_passed", "counts": {k: len(v) for k, v in counts.items()},
                          "planned_inferences": plan["dev_requests"] * len(CONDITIONS), "training_launched": False}))
        return 0
    out = Path(plan["output"])
    if not args.resume:
        out.mkdir(exist_ok=False)
    require(out.is_dir(), "resume output is missing")
    binding = {"plan": plan, "script_sha256": sha(__file__)}
    with queue_lock(out / ".queue.lock") as fd:
        if args.resume:
            require(read(out / "plan.json") == binding, "resume plan/script mismatch")
            state = read(out / "status.json")
            require(state["status"] != "completed", "queue already completed")
        else:
            write(out / "plan.json", binding)
            state = {"created_utc": now(), "completed_stages": {}}
        state.update(pid=os.getpid(), resumed=args.resume, attempt_started_utc=now())
        queue = Queue(plan, state, fd)
        try:
            queue.run()
        except (Exception, KeyboardInterrupt) as error:
            queue.save(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                       error_type=type(error).__name__, message=str(error), finished_utc=now())
            raise
    return 0


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    raise SystemExit(main())
