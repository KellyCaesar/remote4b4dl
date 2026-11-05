"""Prepare and evaluate the fixed LiDAR-Hallu questions with B4DL checkpoints."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .evaluation import scene_interval
from .inputs import GeometryDataset
from .integration import FRAME_KEYS, check_context, language_model, padding_side
from .io import digest, iter_array, read_json, read_jsonl, sha256, unique_index, write_json, write_jsonl
from .resume import append_jsonl, bind_run, output_lease, recover_jsonl, validate_predictions, verify_completion


PAPER_COMMIT = "2ba623dabe2c59ea164e761320d537cdc09a6583"
SCENES_BLOB = "838d436829013438161a9d1b37236aa028689dfa"
METRICS_BLOB = "d4393a24eb3e95db8e0b8f35b635a4549a30c253"
LENGTHS_SHA256 = "87a7e8124f88f9130cb69338a79a3235d634d7622e22238c4b9b68195c860107"
PAPER_FILES = {
    "distance_depth": "3116ec4dc47f32e126dbbd2e412bb65570ca2342f6f25775543f4f76250ce5b1",
    "ego_relative_spatial": "468dc1f9621ce48733dcc12d61272159fe670f1f98bf994172caa6670f581d6d",
    "motion_action": "7cba052defc3547857b391d51a692d816f497f00e67a1a0b5fdfbb33d23aaae1",
    "object_existence": "b7f73f3d1a010a5a5bfbf56beee499d09fbd0b102ceaeb059fbb4e8d46afb239",
    "temporal_grounding": "42801e94d237f798c636fc327cb21aa267bde2922c7923fe2fd5064026a2dfa7",
}
PAPER_BASE_BLOBS = {
    "distance_depth": "41caab0b0aec98ebf9a3be8ef17bb91e45936e23",
    "ego_relative_spatial": "a0888e932feba4129167a8c04d3e969b88654d93",
    "motion_action": "e922ec54115be23d0be565b3fe98e4ad1b4c67f1",
    "object_existence": "18f3e80cf088ffc4e60fbf340a2f5d1e6d5525d5",
    "temporal_grounding": "c84253b21e461a22f15e27cdb3b26d240e2af46f",
}
SCHEMA = "b4dl-lidar-hallu-protocol-v1"


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _check_sample_data(geometry, table_path):
    matches = [value for path, value in geometry.manifest["sources"].items()
               if PurePosixPath(path.replace("\\", "/")).name == "sample_data.json"]
    if len(matches) != 1 or sha256(table_path) != matches[0]:
        raise ValueError("sample_data source differs from the geometry dataset")
    tokens = {token for scene in geometry.manifest["scenes"].values() for token in scene["frame_tokens"]}
    rows = unique_index(row for row in iter_array(table_path) if row["token"] in tokens)
    if set(rows) != tokens:
        raise ValueError("sample_data does not cover geometry frame tokens")
    return rows


def prepare(geometry, paper_root, table_path, output, *, per_category=2000, scene_count=150):
    paper_root, output = Path(paper_root), Path(output)
    scene_path = paper_root / "data/manifests/benchmark_scenes.json"
    if git_blob(scene_path) != SCENES_BLOB:
        raise ValueError("paper benchmark scene manifest differs from pinned source")
    scenes = unique_index(read_json(scene_path)["scenes"], "scene_id")
    if len(scenes) != scene_count:
        raise ValueError("paper benchmark scene count mismatch")
    lengths_path = paper_root / "data/b4dl_eval/hallucination_v2_temporal_shuffle_cd/object_existence_predictions.jsonl"
    if sha256(lengths_path) != LENGTHS_SHA256:
        raise ValueError("paper feature-length evidence differs from pinned source")
    length_rows = read_jsonl(lengths_path)
    if len(length_rows) != per_category:
        raise ValueError("paper feature-length evidence count mismatch")
    lengths = {}
    for index, item in enumerate(length_rows):
        if item.get("index") != index or item.get("scene_id") not in scenes:
            raise ValueError("paper feature-length record identity mismatch")
        try:
            permutation = [int(value) for value in item["shuffle_permutation"].split()]
        except (AttributeError, KeyError, ValueError):
            raise ValueError("invalid paper feature permutation") from None
        if sorted(permutation) != list(range(len(permutation))) or not permutation:
            raise ValueError("invalid paper feature permutation")
        scene_id = item["scene_id"]
        if scene_id in lengths and lengths[scene_id] != len(permutation):
            raise ValueError("paper feature lengths disagree within scene")
        lengths[scene_id] = len(permutation)
    if set(lengths) != set(scenes):
        raise ValueError("paper feature-length evidence does not cover all scenes")
    samples = _check_sample_data(geometry, table_path)
    for scene_id, reference in scenes.items():
        record = geometry.manifest["scenes"].get(scene_id)
        if (record is None or record["scene_token"] != reference["scene_token"] or record["split"] != "test"
                or record["frames"] != lengths[scene_id]):
            raise ValueError("paper and geometry scene identities disagree: " + scene_id)
        for index, sample_token in reference["reference_samples"].items():
            frame = int(index)
            if frame < 0 or frame >= record["frames"] or samples[record["frame_tokens"][frame]]["sample_token"] != sample_token:
                raise ValueError("paper and geometry frame mapping disagree: " + scene_id)
    rows, targets, counts, seen = [], [], {}, set()
    for category, expected_hash in PAPER_FILES.items():
        path = paper_root / "data/b4dl_dataset/hallucination_v2" / (category + ".json")
        if sha256(path) != expected_hash:
            raise ValueError("paper question file hash mismatch: " + category)
        items = read_json(path)
        if not isinstance(items, list) or len(items) != per_category:
            raise ValueError("paper category question count mismatch: " + category)
        counts[category] = len(items)
        for index, item in enumerate(items):
            scene_id, question = item.get("scene_id"), item.get("question")
            kind, answer = item.get("question_type"), item.get("answer")
            if category == "object_existence":
                evidence = length_rows[index]
                if (evidence.get("scene_id") != scene_id or evidence.get("question") != question
                        or evidence.get("reference_answer") != answer):
                    raise ValueError("paper feature-length record does not match question")
            if (scene_id not in scenes or not isinstance(question, str) or not question.strip()
                    or "<video>" in question or item.get("prompt") != "<video>\n" + question
                    or kind not in ("binary", "mcq")):
                raise ValueError("invalid paper question or prompt")
            if kind == "binary":
                if answer not in ("Yes.", "No."):
                    raise ValueError("invalid paper binary answer")
                options = {}
            else:
                options = item.get("options")
                if (not isinstance(options, dict) or set(options) != set("ABCD") or answer not in options
                        or item.get("correct_option") != answer
                        or any(not isinstance(value, str) or not value.strip() for value in options.values())
                        or len(set(options.values())) != 4):
                    raise ValueError("invalid paper multiple-choice options")
            conversations = item.get("conversations", [])
            if (len(conversations) != 2 or conversations[0].get("from") != "human"
                    or conversations[0].get("value") != item["prompt"] or conversations[1].get("from") != "gpt"
                    or conversations[1].get("value") != answer):
                raise ValueError("paper prompt and archived conversation disagree")
            record = geometry.manifest["scenes"][scene_id]
            for key in ("frame", "start_frame", "end_frame"):
                if item.get(key) is not None and (type(item[key]) is not int or not 0 <= item[key] < record["frames"]):
                    raise ValueError("paper question frame outside scene")
            unique = (category, scene_id, question)
            if unique in seen:
                raise ValueError("duplicate paper question within category and scene")
            seen.add(unique)
            request_id = digest([category, index, scene_id, question, options])[:24]
            public = {"request_id": request_id, "source_index": index, "scene_id": scene_id,
                      "category": category, "question_type": kind, "question": question, "options": options,
                      "feature_length": record["frames"], "predicate": item.get("action") or ""}
            public["input_fingerprint"] = digest({**public, "scene_bundle_sha256": record["sha256"]})
            rows.append(public)
            targets.append({"request_id": request_id, "answer": answer})
    if len({row["scene_id"] for row in rows}) != scene_count:
        raise ValueError("paper questions do not cover the expected scenes")
    unique_index(rows, "request_id")
    if output.exists():
        raise FileExistsError("choose a new paper protocol directory")
    output.mkdir(parents=True)
    write_jsonl(output / "requests.jsonl", rows)
    write_jsonl(output / "targets.jsonl", targets)
    protocol = {"schema": SCHEMA, "paper_commit": PAPER_COMMIT, "scene_manifest_git_blob": SCENES_BLOB,
        "paper_file_sha256": PAPER_FILES, "paper_lengths_sha256": LENGTHS_SHA256,
        "geometry_dataset_id": geometry.manifest["dataset_id"],
        "geometry_manifest_sha256": sha256(geometry.root / "manifest.json"), "requests": len(rows),
        "scene_count": len(scenes), "category_counts": counts,
        "files": {name: sha256(output / name) for name in ("requests.jsonl", "targets.jsonl")},
        "input_scope": geometry.manifest.get("input_scope"),
        "scope_note": "Original paper questions with the supplied geometry feature crop and frame mapping."}
    protocol["protocol_id"] = digest(protocol)
    write_json(output / "manifest.json", protocol)
    return protocol


def load_protocol(root, geometry):
    root = Path(root)
    protocol = read_json(root / "manifest.json")
    if (protocol.get("schema") != SCHEMA or protocol.get("protocol_id") != digest(
            {key: value for key, value in protocol.items() if key != "protocol_id"})
            or protocol.get("paper_commit") != PAPER_COMMIT or protocol.get("scene_manifest_git_blob") != SCENES_BLOB
            or protocol.get("paper_file_sha256") != PAPER_FILES
            or protocol.get("paper_lengths_sha256") != LENGTHS_SHA256
            or protocol.get("geometry_dataset_id") != geometry.manifest["dataset_id"]
            or protocol.get("geometry_manifest_sha256") != sha256(geometry.root / "manifest.json")
            or set(protocol.get("files", {})) != {"requests.jsonl", "targets.jsonl"}):
        raise ValueError("paper protocol identity mismatch")
    for name, expected in protocol["files"].items():
        if sha256(root / name) != expected:
            raise ValueError("paper protocol file hash mismatch")
    rows = read_jsonl(root / "requests.jsonl")
    targets = unique_index(read_jsonl(root / "targets.jsonl"), "request_id")
    unique_index(rows, "request_id")
    if (len(rows) != protocol["requests"] or set(targets) != {row["request_id"] for row in rows}
            or dict(Counter(row["category"] for row in rows)) != protocol["category_counts"]):
        raise ValueError("paper protocol denominator mismatch")
    for row in rows:
        record = geometry.manifest["scenes"].get(row["scene_id"])
        content = {key: value for key, value in row.items() if key != "input_fingerprint"}
        if (record is None or row["feature_length"] != record["frames"] or record["split"] != "test"
                or row["input_fingerprint"] != digest({**content, "scene_bundle_sha256": record["sha256"]})
                or set(targets[row["request_id"]]) != {"request_id", "answer"}):
            raise ValueError("paper request input or target identity mismatch")
    return protocol, rows, targets


def _item(geometry, row):
    scene = geometry.scene(row["scene_id"])
    item = {"prompt": row["question"], "features": scene["features"], "frame_times": scene["frame_times"],
            "frame_geometry": scene["frame_geometry"], "frame_valid_mask": np.ones(len(scene["features"]), bool)}
    for key in ("cell_features", "cell_xyz", "cell_indices", "cell_valid_mask", "ego_from_sst"):
        if key in scene:
            item[key] = scene[key]
    return item


def shuffle_item(item, permutation):
    frames = len(item["features"])
    if sorted(permutation) != list(range(frames)) or permutation == list(range(frames)):
        raise ValueError("temporal contrast requires a non-identity frame permutation")
    result = dict(item)
    for key in ("features", "cell_features", "cell_xyz", "cell_indices", "cell_valid_mask", "ego_from_sst"):
        if key in item:
            result[key] = np.asarray(item[key])[permutation].copy()
    return result


def _prefill(model, tokenizer, item, budget):
    from .reasoning.policy import sequence_batch
    batch = sequence_batch(model, tokenizer, [item])
    base = language_model(model)
    check_context(model, batch["input_ids"], batch["attention_mask"], batch["frame_valid_mask"], budget, batch)
    with padding_side(base, "left"):
        _, position, attention, _, embeds, _ = base.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, None, batch["images"],
            **{key: batch.get(key) for key in FRAME_KEYS})
    out = model(inputs_embeds=embeds, attention_mask=attention, position_ids=position, use_cache=True)
    if out.past_key_values is None:
        raise RuntimeError("paper decoding requires a KV cache; use evaluation mode with caching enabled")
    return out, attention


def _advance(model, token, output, attention):
    attention = torch.cat((attention, attention.new_ones(1, 1)), dim=1)
    position = attention.sum(dim=1, keepdim=True) - 1
    result = model(input_ids=token.reshape(1, 1), attention_mask=attention, position_ids=position,
                   past_key_values=output.past_key_values, use_cache=True)
    if result.past_key_values is None:
        raise RuntimeError("paper decoding lost its KV cache")
    return result, attention


def generate_answer(model, tokenizer, item, *, seed, temperature=.05, max_new_tokens=32, alpha=0.):
    """Use a shared sampled prefix and independent clean/shuffled KV caches."""
    from .reasoning.trainer import _precision
    if (type(seed) is not int or seed < 0 or type(max_new_tokens) is not int or max_new_tokens < 1
            or type(temperature) not in (int, float) or not 0 <= temperature < float("inf")
            or type(alpha) not in (int, float) or not 0 <= alpha < float("inf")):
        raise ValueError("invalid paper decoding settings")
    device = language_model(model).get_input_embeddings().weight.device
    permutation = None
    if alpha:
        n = len(item["features"])
        if n < 2:
            raise ValueError("TSCD requires at least two frames")
        permutation = np.random.default_rng(seed).permutation(n).tolist()
        if permutation == list(range(n)):
            permutation = permutation[1:] + permutation[:1]
        shuffled = shuffle_item(item, permutation)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    ids = []
    model.eval()
    with torch.random.fork_rng(devices=devices), torch.no_grad(), _precision(model):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        clean, clean_mask = _prefill(model, tokenizer, item, max_new_tokens)
        if alpha:
            contrast, contrast_mask = _prefill(model, tokenizer, shuffled, max_new_tokens)
        for _ in range(max_new_tokens):
            logits = clean.logits[0, -1].float()
            if alpha:
                logits = logits + alpha * (logits - contrast.logits[0, -1].float())
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite paper decoding logits")
            token = (torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
                     if temperature else logits.argmax().reshape(1))
            value = int(token.item())
            ids.append(value)
            if value == tokenizer.eos_token_id:
                break
            clean, clean_mask = _advance(model, token, clean, clean_mask)
            if alpha:
                contrast, contrast_mask = _advance(model, token, contrast, contrast_mask)
    status = "ok" if ids[-1] == tokenizer.eos_token_id else "truncated"
    forbidden = set(getattr(tokenizer, "all_special_ids", [])) - {tokenizer.eos_token_id}
    if any(value in forbidden for value in ids):
        status = "error"
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"text": text, "status": status, "generated_tokens": len(ids), "permutation": permutation}


def infer(protocol_root, dataset, model, tokenizer, artifact, baseline, output, *, model_kind,
          checkpoint_metadata=None, seed=42, temperature=.05, max_new_tokens=32, alpha=0., resume=False):
    from .reasoning.evaluation import checkpoint_lineage, checkpoint_training
    from .reasoning.trainer import implementation_identity, numerics_identity
    from .runtime import runtime_identity
    protocol, rows, _ = load_protocol(protocol_root, dataset.geometry)
    artifact_hash = sha256(artifact)
    if model_kind == "original_b3":
        if artifact_hash != sha256(baseline):
            raise ValueError("original B3 artifact must be the baseline lock")
        lineage, training = None, None
    elif model_kind == "reasoning":
        if checkpoint_metadata is None:
            raise ValueError("reasoning checkpoint metadata required")
        lineage = checkpoint_lineage(checkpoint_metadata, artifact_hash, dataset.manifest["dataset_id"])
        training = checkpoint_training(checkpoint_metadata)
    else:
        raise ValueError("unknown paper model kind")
    base = language_model(model)
    device = base.get_input_embeddings().weight.device
    run = {"schema": "b4dl-lidar-hallu-inference-v1", "protocol_id": protocol["protocol_id"],
        "model_kind": model_kind, "artifact_sha256": artifact_hash, "baseline_sha256": sha256(baseline),
        "lineage": lineage, "training": training, "implementation": implementation_identity(),
        "regional_dataset_id": getattr(dataset, "regional_manifest", {}).get("dataset_id"),
        "decoding": {"seed": seed, "temperature": temperature, "max_new_tokens": max_new_tokens, "alpha": alpha},
        "runtime": runtime_identity(base), "device": str(device),
        "dtype": str(base.get_input_embeddings().weight.dtype), "numerics": numerics_identity(model),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_scope": protocol["input_scope"]}
    output = Path(output)
    if output.exists() and not resume or not output.exists() and resume:
        raise ValueError("choose a new paper output or explicitly resume it")
    output.mkdir(parents=True, exist_ok=resume)
    path = output / "predictions.jsonl"
    completion = {"status": "paper_inference_complete", "requests": len(rows), "run": run}
    with output_lease(output):
        bind_run(output / "run.json", run, resume, [path, output / "completion.json"])
        if verify_completion(output, completion, {"predictions_sha256": path}):
            validate_predictions(rows, read_jsonl(path), complete=True)
            return read_json(output / "completion.json")
        existing = recover_jsonl(path, lambda values: validate_predictions(rows, values))
        for row in rows[len(existing):]:
            sample_seed = int(digest([seed, row["request_id"]])[:16], 16) % (2**63 - 1)
            prediction = generate_answer(model, tokenizer, _item(dataset.geometry, row), seed=sample_seed,
                temperature=temperature, max_new_tokens=max_new_tokens, alpha=alpha)
            prediction.update(request_id=row["request_id"], input_fingerprint=row["input_fingerprint"], seed=sample_seed)
            append_jsonl(path, [prediction])
        validate_predictions(rows, read_jsonl(path), complete=True)
        completion["predictions_sha256"] = sha256(path)
        write_json(output / "completion.json", completion)
    return completion


def import_archive(protocol_root, geometry, paper_root, output):
    """Bind the published no-metatoken B4DL predictions to the verified questions."""
    protocol, rows, targets = load_protocol(protocol_root, geometry)
    paper_root, output = Path(paper_root), Path(output)
    archived = {}
    for category, expected_blob in PAPER_BASE_BLOBS.items():
        path = paper_root / "data/b4dl_eval/hallucination_v2" / (category + "_predictions.jsonl")
        if git_blob(path) != expected_blob:
            raise ValueError("paper baseline prediction file differs from pinned source: " + category)
        predictions = read_jsonl(path)
        if len(predictions) != protocol["category_counts"][category]:
            raise ValueError("paper baseline prediction count mismatch: " + category)
        archived[category] = predictions
    predictions = []
    for row in rows:
        source = archived[row["category"]][row["source_index"]]
        if (source.get("index") != row["source_index"] or source.get("category") != row["category"]
                or source.get("scene_id") != row["scene_id"] or source.get("question") != row["question"]
                or source.get("prompt") != "<video>\n" + row["question"]
                or source.get("question_type") != row["question_type"]
                or (source.get("options") or {}) != row["options"]
                or source.get("reference_answer") != targets[row["request_id"]]["answer"]
                or not isinstance(source.get("prediction"), str) or not isinstance(source.get("error"), str)):
            raise ValueError("paper archived prediction does not match its question")
        predictions.append({"request_id": row["request_id"], "input_fingerprint": row["input_fingerprint"],
            "text": source["prediction"], "status": "error" if source["error"] else "ok",
            "generated_tokens": 0, "archive_error": source["error"]})
    run = {"schema": "b4dl-lidar-hallu-inference-v1", "protocol_id": protocol["protocol_id"],
        "model_kind": "paper_archive", "artifact_sha256": digest(PAPER_BASE_BLOBS), "baseline_sha256": None,
        "lineage": None, "training": None, "implementation": PAPER_COMMIT, "regional_dataset_id": None,
        "decoding": {"seed": None, "temperature": .05, "max_new_tokens": 32, "alpha": 0.},
        "runtime": {"source": "published predictions", "checkpoint_identity": "not provided"},
        "device": "paper", "dtype": "float16", "numerics": {}, "gpu_name": None,
        "input_scope": "published LiDAR-CLIP feature cache; preprocessing equivalence is not established",
        "source_git_blobs": PAPER_BASE_BLOBS}
    if output.exists():
        raise FileExistsError("choose a new paper archive output directory")
    output.mkdir(parents=True)
    path = output / "predictions.jsonl"
    write_jsonl(path, predictions)
    write_json(output / "run.json", run)
    completion = {"status": "paper_inference_complete", "requests": len(rows), "run": run,
                  "predictions_sha256": sha256(path)}
    write_json(output / "completion.json", completion)
    return completion


def load_paper_parser(paper_root):
    path = Path(paper_root) / "lidar_hallu/metrics.py"
    if git_blob(path) != METRICS_BLOB:
        raise ValueError("paper answer parser differs from pinned source")
    spec = importlib.util.spec_from_file_location("b4dl_pinned_lidar_hallu_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_answer, module.INVALID


def _summary(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "valid_rate": None}
    labels = Counter(row["truth"] for row in rows)
    predictions = Counter(row["answer"] for row in rows)
    result = {"n": len(rows), "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "valid_rate": sum(row["valid"] for row in rows) / len(rows), "label_counts": dict(labels),
        "answer_counts": dict(predictions), "status_counts": dict(Counter(row["status"] for row in rows)),
        "scene_interval": scene_interval(rows, repetitions=10000, seed=20260904)}
    if all(row["question_type"] == "binary" for row in rows):
        positives = [row for row in rows if row["truth"] == "Yes."]
        negatives = [row for row in rows if row["truth"] == "No."]
        result.update(tpr=sum(row["answer"] == "Yes." for row in positives) / len(positives) if positives else None,
                      fpr=sum(row["answer"] == "Yes." for row in negatives) / len(negatives) if negatives else None,
                      fixed_no_accuracy=len(negatives) / len(rows))
    if all(row["question_type"] == "mcq" for row in rows):
        result.update(fixed_a_accuracy=labels["A"] / len(rows), a_rate=predictions["A"] / len(rows))
    return result


def paper_pairs(requests, targets):
    """Reproduce the paper's deterministic scene/index-ordered binary matching."""
    groups = defaultdict(lambda: {"Yes.": [], "No.": []})
    for request in requests:
        if request["question_type"] == "binary":
            key = (request["category"], "<video>\n" + request["question"], request["feature_length"])
            groups[key][targets[request["request_id"]]["answer"]].append(request)
    pairs = []
    for key, group in sorted(groups.items()):
        positives = sorted(group["Yes."], key=lambda row: (row["scene_id"], row["source_index"]))
        negatives = sorted(group["No."], key=lambda row: (row["scene_id"], row["source_index"]))
        owners = {}

        def assign(positive, seen):
            for negative, row in enumerate(negatives):
                if negative in seen or row["scene_id"] == positives[positive]["scene_id"]:
                    continue
                seen.add(negative)
                if negative not in owners or assign(owners[negative], seen):
                    owners[negative] = positive
                    return True
            return False

        for positive in range(len(positives)):
            assign(positive, set())
        for negative, positive in sorted(owners.items()):
            pairs.append({"category": key[0], "yes_id": positives[positive]["request_id"],
                          "no_id": negatives[negative]["request_id"]})
    return pairs


def evaluate(protocol_root, geometry, prediction_path, paper_root, output):
    protocol, requests, targets = load_protocol(protocol_root, geometry)
    path = Path(prediction_path)
    receipt = read_json(path.parent / "completion.json")
    run = read_json(path.parent / "run.json")
    if (receipt.get("status") != "paper_inference_complete" or receipt.get("run") != run
            or receipt.get("requests") != len(requests) or run.get("protocol_id") != protocol["protocol_id"]
            or receipt.get("predictions_sha256") != sha256(path)):
        raise ValueError("paper prediction completion identity mismatch")
    predictions = read_jsonl(path)
    validate_predictions(requests, predictions, complete=True)
    parse_answer, invalid = load_paper_parser(paper_root)
    rows, groups = [], defaultdict(list)
    for request, prediction in zip(requests, predictions):
        answer = parse_answer(prediction["text"], request["question_type"], request["options"],
                              "inference failed" if prediction["status"] in ("error", "unavailable") else "")
        truth = targets[request["request_id"]]["answer"]
        row = {"request_id": request["request_id"], "scene_id": request["scene_id"],
               "category": request["category"], "question_type": request["question_type"],
               "predicate": request["predicate"], "truth": truth, "answer": answer,
               "valid": answer != invalid, "correct": answer == truth, "status": prediction["status"]}
        rows.append(row)
        groups[(row["category"], row["question_type"])].append(row)
    pairs = paper_pairs(requests, targets)
    index = {row["request_id"]: row for row in rows}
    by_motion = defaultdict(list)
    for row in rows:
        if row["category"] == "motion_action":
            by_motion[row["predicate"]].append(row)
    report = {"schema": "b4dl-lidar-hallu-evaluation-v1", "protocol_id": protocol["protocol_id"],
        "run": run, "parser_git_blob": METRICS_BLOB, "complete_prediction_set": True,
        "summary": _summary(rows), "by_category_type": {f"{category}/{kind}": _summary(group)
            for (category, kind), group in sorted(groups.items())},
        "motion_by_predicate": {key: _summary(group) for key, group in sorted(by_motion.items())},
        "opposite_pairs": {"n": len(pairs), "both_correct": sum(index[p["yes_id"]]["correct"]
            and index[p["no_id"]]["correct"] for p in pairs)}, "rows": rows}
    report["opposite_pairs"]["both_correct_rate"] = (report["opposite_pairs"]["both_correct"] / len(pairs)) if pairs else None
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, report)
    return report


def compare(first, second, *, literature=False):
    if (first.get("schema") != "b4dl-lidar-hallu-evaluation-v1" or second.get("schema") != first["schema"]
            or not first.get("complete_prediction_set") or not second.get("complete_prediction_set")
            or first["protocol_id"] != second["protocol_id"] or first["parser_git_blob"] != second["parser_git_blob"]):
        raise ValueError("paper comparison requires complete matched evaluations")
    a, b = first["run"], second["run"]
    if literature:
        if sum(run.get("model_kind") == "paper_archive" for run in (a, b)) != 1:
            raise ValueError("literature comparison requires one published archive and one local model")
        if a["decoding"]["alpha"] != 0 or b["decoding"]["alpha"] != 0:
            raise ValueError("published default baseline requires alpha=0 in both runs")
        if any(a["decoding"][key] != b["decoding"][key] for key in ("temperature", "max_new_tokens")):
            raise ValueError("literature comparison requires the paper short-answer settings")
    else:
        for key in ("protocol_id", "baseline_sha256", "implementation", "runtime", "device", "dtype", "numerics", "gpu_name"):
            if a.get(key) != b.get(key):
                raise ValueError("paper comparison runtime or baseline mismatch: " + key)
        for key in ("seed", "temperature", "max_new_tokens"):
            if a["decoding"][key] != b["decoding"][key]:
                raise ValueError("paper comparison decoding mismatch: " + key)
        if a["decoding"]["alpha"] != b["decoding"]["alpha"] and a["artifact_sha256"] != b["artifact_sha256"]:
            raise ValueError("different TSCD strengths require the same checkpoint")
    left, right = unique_index(first["rows"], "request_id"), unique_index(second["rows"], "request_id")
    if not left or set(left) != set(right):
        raise ValueError("paper comparison denominators differ")
    rows = []
    for ident, one in left.items():
        two = right[ident]
        if (one["scene_id"], one["truth"], one["category"]) != (two["scene_id"], two["truth"], two["category"]):
            raise ValueError("paper comparison truth mismatch")
        rows.append({"request_id": ident, "scene_id": one["scene_id"], "delta": int(two["correct"]) - int(one["correct"])})
    repaired = sum(row["delta"] == 1 for row in rows)
    regressed = sum(row["delta"] == -1 for row in rows)
    return {"schema": "b4dl-lidar-hallu-comparison-v1", "protocol_id": first["protocol_id"],
        "comparison_scope": "same questions; published and local input/weights/seed may differ" if literature
                            else "same questions, baseline, runtime and decoding settings",
        "base_run": a, "candidate_run": b, "n": len(rows), "repaired": repaired, "regressed": regressed,
        "accuracy_delta": (repaired - regressed) / len(rows),
        "scene_interval": scene_interval(rows, "delta", repetitions=10000, seed=20260904)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("geometry-dataset", "paper-root", "sample-data", "output"):
        prepare_parser.add_argument("--" + name, required=True)
    infer_parser = sub.add_parser("infer")
    for name in ("dataset", "geometry-dataset", "baseline", "preparation", "protocol", "output"):
        infer_parser.add_argument("--" + name, required=True)
    source = infer_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--original-b3", action="store_true")
    source.add_argument("--checkpoint")
    infer_parser.add_argument("--regional-features")
    infer_parser.add_argument("--mixed-plan")
    infer_parser.add_argument("--device", default="cuda:0")
    infer_parser.add_argument("--gpu-lock", required=True)
    infer_parser.add_argument("--seed", type=int, default=42)
    infer_parser.add_argument("--temperature", type=float, default=.05)
    infer_parser.add_argument("--max-new-tokens", type=int, default=32)
    infer_parser.add_argument("--alpha", type=float, default=0.)
    infer_parser.add_argument("--resume", action="store_true")
    archive_parser = sub.add_parser("import-archive")
    for name in ("geometry-dataset", "paper-root", "protocol", "output"):
        archive_parser.add_argument("--" + name, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("geometry-dataset", "paper-root", "protocol", "predictions", "output"):
        evaluate_parser.add_argument("--" + name, required=True)
    compare_parser = sub.add_parser("compare")
    for name in ("base", "candidate", "output"):
        compare_parser.add_argument("--" + name, required=True)
    compare_parser.add_argument("--literature", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(GeometryDataset(args.geometry_dataset), args.paper_root, args.sample_data, args.output)
    elif args.command == "import-archive":
        result = import_archive(args.protocol, GeometryDataset(args.geometry_dataset), args.paper_root, args.output)
    elif args.command == "infer":
        from .b3 import device_lease
        from .reasoning.data import ReasoningDataset
        from .reasoning.evaluation import load_model, load_original_model
        dataset = ReasoningDataset(args.dataset, args.geometry_dataset)
        if args.regional_features:
            if args.original_b3:
                raise ValueError("original B3 uses historical features")
            from .regional_data import RegionalDataset
            dataset = RegionalDataset(dataset, args.regional_features)
        plan = None
        if args.mixed_plan:
            if args.original_b3:
                raise ValueError("original B3 has no mixed training plan")
            from .reasoning.mixed import MixedPlan
            plan = MixedPlan(args.mixed_plan, dataset)
        with device_lease(args.device, args.gpu_lock):
            if args.original_b3:
                tokenizer, model = load_original_model(dataset, args.baseline, args.preparation, args.device)
                metadata, artifact, kind = None, args.baseline, "original_b3"
            else:
                tokenizer, model, metadata = load_model(dataset, args.baseline, args.preparation, args.checkpoint, args.device, plan)
                artifact, kind = args.checkpoint, "reasoning"
            result = infer(args.protocol, dataset, model, tokenizer, artifact, args.baseline, args.output,
                model_kind=kind, checkpoint_metadata=metadata, seed=args.seed, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, alpha=args.alpha, resume=args.resume)
    elif args.command == "evaluate":
        result = evaluate(args.protocol, GeometryDataset(args.geometry_dataset), args.predictions, args.paper_root, args.output)
    else:
        result = compare(read_json(args.base), read_json(args.candidate), literature=args.literature)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    print({key: value for key, value in result.items() if key != "rows"}, flush=True)


if __name__ == "__main__":
    main()
