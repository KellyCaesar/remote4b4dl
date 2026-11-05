"""Freeze train-only geometry pairs and original-task replay in a reproducible plan."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..evaluation import opposite_pairs
from ..io import digest, read_json, read_jsonl, sha256, unique_index, write_json, write_jsonl
from .supervision import pair_regularizer


SCHEMA = "b4dl-mixed-training-plan-v1"


def replay_prompt(text):
    if not isinstance(text, str):
        raise ValueError("replay prompt must be text")
    question = text.split("<meta>", 1)[0]
    for marker in ("<video>", "<4DLiDAR>"):
        question = question.replace(marker, "")
    question = question.strip()
    if not question:
        raise ValueError("empty replay question")
    return question


def _replay_rows(dataset, sources):
    scenes = dataset.geometry.manifest["scenes"]
    result, excluded = [], Counter()
    source_info = []
    for source in sources:
        path = Path(source)
        source_hash = sha256(path)
        source_info.append({"path": str(path.resolve()), "sha256": source_hash})
        items = read_json(path)
        if isinstance(items, dict):
            items = list(items.values())
        if not isinstance(items, list):
            raise ValueError("original-task source must be an array or object")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                excluded["invalid_record"] += 1
                continue
            scene = item.get("scene_id") or item.get("id")
            if scene not in scenes or scenes[scene]["split"] != "train":
                excluded["outside_train_scenes"] += 1
                continue
            conversation = item.get("conversations")
            if (not isinstance(conversation, list) or len(conversation) != 2
                    or not all(isinstance(turn, dict) for turn in conversation)
                    or conversation[0].get("from") != "human" or conversation[1].get("from") != "gpt"
                    or not isinstance(conversation[1].get("value"), str) or not conversation[1]["value"].strip()):
                excluded["unsupported_conversation"] += 1
                continue
            try:
                question = replay_prompt(conversation[0].get("value"))
            except ValueError:
                excluded["invalid_question"] += 1
                continue
            answer = conversation[1]["value"].strip()
            ident = digest([source_hash, index, scene, question, answer])[:24]
            result.append({"replay_id": ident, "scene_id": scene, "prompt": question, "answer": answer})
    unique_index(result, "replay_id")
    if not result:
        raise ValueError("no valid original-task examples in geometry train scenes")
    return result, source_info, dict(excluded)


def _geometry_groups(dataset):
    requests = dataset.requests
    if not requests or {row["split"] for row in requests} != {"train"}:
        raise ValueError("mixed plan requires only geometry train requests")
    indexed = unique_index(requests, "request_id")
    groups = defaultdict(lambda: defaultdict(list))
    for pair in opposite_pairs(requests, dataset.targets):
        predicate = indexed[pair["yes_id"]]["predicate"]
        groups[pair["task"]][predicate].append((pair["yes_id"], pair["no_id"], "opposite"))
    orbits = defaultdict(list)
    for row in requests:
        if row["task"] == "distance":
            orbits[row["semantic_id"]].append(row)
    for ident, rows in sorted(orbits.items()):
        if len(rows) != 4 or {r["option_rotation"] for r in rows} != set(range(4)):
            raise ValueError("incomplete distance rotation group: " + ident)
        rows = sorted(rows, key=lambda row: row["option_rotation"])
        for first in range(4):
            for second in range(first + 1, 4):
                groups["distance"][rows[first]["predicate"]].append(
                    (rows[first]["request_id"], rows[second]["request_id"], "rotation"))
    for predicates in groups.values():
        for values in predicates.values():
            for first, second, kind in values:
                r = [indexed[first], indexed[second]]
                t = [dataset.targets[first], dataset.targets[second]]
                dimension = 2 if kind == "opposite" else 4
                dummy = [(tuple(("Yes", "No") if dimension == 2 else row["options"]),
                          torch.zeros(dimension)) for row in r]
                pair_regularizer(r, t, dummy, kind)
    return {task: dict(predicates) for task, predicates in groups.items()}


def build_plan(dataset, replay_sources, output, *, seed=42, updates=13851):
    if type(seed) is not int or seed < 0 or type(updates) is not int or updates < 1:
        raise ValueError("invalid mixed plan seed or update count")
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new mixed-plan directory")
    if not replay_sources:
        raise ValueError("at least one original-task source is required")
    replay, sources, excluded = _replay_rows(dataset, replay_sources)
    groups = _geometry_groups(dataset)
    geometry_by_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    replay_by_scene = defaultdict(list)
    for request in dataset.requests:
        ident = request["request_id"]
        label = dataset.targets[ident]["answer"]
        geometry_by_task[request["task"]][request["predicate"]][label].append(ident)
    for row in replay:
        replay_by_scene[row["scene_id"]].append(row["replay_id"])
    tasks = sorted(geometry_by_task)
    scenes = sorted(replay_by_scene)
    rng = np.random.default_rng(seed)
    plan = []
    for step in range(updates):
        task = tasks[int(rng.integers(len(tasks)))]
        predicates = sorted(geometry_by_task[task])
        predicate = predicates[int(rng.integers(len(predicates)))]
        matches = groups.get(task, {}).get(predicate, [])
        if matches:
            first, second, kind = matches[int(rng.integers(len(matches)))]
            reason = None
        else:
            labels = sorted(geometry_by_task[task][predicate])
            selected = []
            for _ in range(2):
                label = labels[int(rng.integers(len(labels)))]
                choices = geometry_by_task[task][predicate][label]
                selected.append(choices[int(rng.integers(len(choices)))])
            first, second = selected
            kind = "unpaired"
            reason = "no_supported_pair_for_task_predicate"
        replay_ids = []
        for _ in range(2):
            scene = scenes[int(rng.integers(len(scenes)))]
            examples = replay_by_scene[scene]
            replay_ids.append(examples[int(rng.integers(len(examples)))])
        plan.append({"step": step + 1, "geometry": [first, second], "replay": replay_ids,
                     "task": task, "predicate": predicate, "pair_type": kind, "fallback_reason": reason})
    output.mkdir(parents=True)
    write_jsonl(output / "replay.jsonl", replay)
    write_jsonl(output / "plan.jsonl", plan)
    manifest = {"schema": SCHEMA, "reasoning_dataset_id": dataset.manifest["dataset_id"],
        "geometry_dataset_id": dataset.geometry.manifest["dataset_id"], "seed": seed, "updates": updates,
        "source_files": sources, "replay_rows": len(replay), "replay_scenes": len(scenes),
        "excluded": excluded, "pair_counts": dict(Counter(row["pair_type"] for row in plan)),
        "task_counts": dict(Counter(row["task"] for row in plan)),
        "fallback_counts": dict(Counter(row["fallback_reason"] for row in plan if row["fallback_reason"])),
        "files": {name: sha256(output / name) for name in ("replay.jsonl", "plan.jsonl")}}
    manifest["plan_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


class MixedPlan:
    def __init__(self, root, dataset):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        manifest = self.manifest
        if (manifest.get("schema") != SCHEMA or manifest.get("plan_id") != digest(
                {key: value for key, value in manifest.items() if key != "plan_id"})
                or manifest.get("reasoning_dataset_id") != dataset.manifest["dataset_id"]
                or manifest.get("geometry_dataset_id") != dataset.geometry.manifest["dataset_id"]):
            raise ValueError("mixed-plan identity mismatch")
        for name, expected in manifest["files"].items():
            if name not in ("replay.jsonl", "plan.jsonl") or sha256(self.root / name) != expected:
                raise ValueError("mixed-plan file hash mismatch")
        if set(manifest["files"]) != {"replay.jsonl", "plan.jsonl"}:
            raise ValueError("mixed-plan files incomplete")
        self.replay = unique_index(read_jsonl(self.root / "replay.jsonl"), "replay_id")
        self.plan = read_jsonl(self.root / "plan.jsonl")
        self.requests = unique_index(read_jsonl(dataset.root / "requests.jsonl"), "request_id")
        if (len(self.replay) != manifest["replay_rows"] or len(self.plan) != manifest["updates"]
                or dict(Counter(row["pair_type"] for row in self.plan)) != manifest["pair_counts"]
                or dict(Counter(row["task"] for row in self.plan)) != manifest["task_counts"]
                or dict(Counter(row["fallback_reason"] for row in self.plan if row["fallback_reason"])) != manifest["fallback_counts"]):
            raise ValueError("mixed-plan counts disagree")
        scenes = dataset.geometry.manifest["scenes"]
        for replay in self.replay.values():
            if (set(replay) != {"replay_id", "scene_id", "prompt", "answer"}
                    or replay["scene_id"] not in scenes or scenes[replay["scene_id"]]["split"] != "train"
                    or not isinstance(replay["answer"], str) or not replay["answer"].strip()
                    or replay_prompt(replay["prompt"]) != replay["prompt"]):
                raise ValueError("invalid original-task replay row")
        for index, row in enumerate(self.plan):
            if (set(row) != {"step", "geometry", "replay", "task", "predicate", "pair_type", "fallback_reason"}
                    or row["step"] != index + 1
                    or len(row["geometry"]) != 2 or len(row["replay"]) != 2
                    or (row["pair_type"] == "unpaired") != (row["fallback_reason"] == "no_supported_pair_for_task_predicate")
                    or row["pair_type"] != "unpaired" and row["fallback_reason"] is not None
                    or any(ident not in self.requests for ident in row["geometry"])
                    or any(ident not in self.replay for ident in row["replay"])):
                raise ValueError("invalid mixed-plan step")
            requests = [self.requests[ident] for ident in row["geometry"]]
            if any(r["split"] != "train" or r["task"] != row["task"] or r["predicate"] != row["predicate"] for r in requests):
                raise ValueError("mixed-plan geometry must come from train and match task")
            targets = [dataset.targets[r["request_id"]] for r in requests]
            dimension = 2 if requests[0]["kind"] == "binary" else 4
            dummy = [(tuple(("Yes", "No") if dimension == 2 else r["options"]), torch.zeros(dimension)) for r in requests]
            pair_regularizer(requests, targets, dummy, row["pair_type"])

    def replay_input(self, dataset, replay_id):
        row = self.replay[replay_id]
        scene = dataset.geometry.scene(row["scene_id"])
        item = {"prompt": row["prompt"], "features": scene["features"],
                "frame_times": scene["frame_times"], "frame_geometry": scene["frame_geometry"],
                "frame_valid_mask": np.ones(len(scene["features"]), dtype=bool)}
        for key in ("cell_features", "cell_xyz", "cell_indices", "cell_valid_mask", "ego_from_sst"):
            if key in scene:
                item[key] = scene[key]
        return item, row["answer"]
