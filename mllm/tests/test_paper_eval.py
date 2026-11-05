"""Paper question identity, archived scores and TSCD decoding checks."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from b4dl_core import paper_eval
from b4dl_core.io import digest, read_json, read_jsonl, sha256, write_json, write_jsonl
from test_reasoning_training import CharacterTokenizer, item, model
from test_regional_pipeline import regional_item, regional_model


def fake_geometry(root, references, lengths):
    geometry_root = root / "geometry"
    geometry_root.mkdir()
    table = root / "sample_data.json"
    scenes, samples = {}, []
    for reference in references:
        scene = reference["scene_id"]
        tokens = []
        for index in range(lengths[scene]):
            token = f"{scene}-{index}"
            tokens.append(token)
            samples.append({"token": token, "sample_token": reference["reference_samples"].get(
                str(index), f"{scene}-synthetic-{index}")})
        scenes[scene] = {"scene_token": reference["scene_token"], "split": "test", "frames": len(tokens),
                         "frame_tokens": tokens, "sha256": digest([scene])}
    write_json(table, samples)
    manifest = {"dataset_id": "fixture", "scenes": scenes, "sources": {str(table): sha256(table)},
                "input_scope": {"description": "synthetic fixture"}}
    write_json(geometry_root / "manifest.json", manifest)
    return SimpleNamespace(root=geometry_root, manifest=manifest), table


def small_paper(root):
    paper = root / "paper"
    (paper / "data/manifests").mkdir(parents=True)
    (paper / "data/b4dl_dataset/hallucination_v2").mkdir(parents=True)
    (paper / "data/b4dl_eval/hallucination_v2").mkdir(parents=True)
    (paper / "data/b4dl_eval/hallucination_v2_temporal_shuffle_cd").mkdir(parents=True)
    (paper / "lidar_hallu").mkdir()
    references = [{"scene_id": "s1", "scene_token": "token1", "reference_samples": {"0": "sample1"}},
                  {"scene_id": "s2", "scene_token": "token2", "reference_samples": {"0": "sample2"}}]
    write_json(paper / "data/manifests/benchmark_scenes.json", {"scenes": references})
    files, blobs, length_rows = {}, {}, []
    for category in paper_eval.PAPER_FILES:
        rows, predictions = [], []
        for index, scene in enumerate(("s1", "s2")):
            kind = "mcq" if category in ("distance_depth", "temporal_grounding") else "binary"
            question = "Which option? A. car B. bus C. cone D. truck" if kind == "mcq" else "Is there a car?"
            answer = ("A" if index == 0 else "B") if kind == "mcq" else ("Yes." if index == 0 else "No.")
            options = {"A": "car", "B": "bus", "C": "cone", "D": "truck"} if kind == "mcq" else None
            prompt = "<video>\n" + question
            row = {"scene_id": scene, "question": question, "question_type": kind, "answer": answer,
                   "prompt": prompt, "conversations": [{"from": "human", "value": prompt},
                                                          {"from": "gpt", "value": answer}]}
            if options:
                row.update(options=options, correct_option=answer)
            rows.append(row)
            prediction = {"index": index, "category": category, "scene_id": scene, "question": question,
                          "question_type": kind, "prompt": prompt, "reference_answer": answer,
                          "options": options, "prediction": answer, "error": ""}
            predictions.append(prediction)
            if category == "object_existence":
                length_rows.append({**prediction, "shuffle_permutation": "0 2 1"})
        path = paper / "data/b4dl_dataset/hallucination_v2" / (category + ".json")
        write_json(path, rows)
        files[category] = sha256(path)
        path = paper / "data/b4dl_eval/hallucination_v2" / (category + "_predictions.jsonl")
        write_jsonl(path, predictions)
        blobs[category] = paper_eval.git_blob(path)
    length_path = paper / "data/b4dl_eval/hallucination_v2_temporal_shuffle_cd/object_existence_predictions.jsonl"
    write_jsonl(length_path, length_rows)
    metrics = paper / "lidar_hallu/metrics.py"
    metrics.write_text("INVALID='<invalid>'\ndef parse_answer(text, kind, options=None, error=''):\n"
                       "    return INVALID if error else text\n", encoding="utf-8")
    return paper, references, {"PAPER_FILES": files, "PAPER_BASE_BLOBS": blobs,
        "SCENES_BLOB": paper_eval.git_blob(paper / "data/manifests/benchmark_scenes.json"),
        "LENGTHS_SHA256": sha256(length_path), "METRICS_BLOB": paper_eval.git_blob(metrics)}


class PaperEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_tiny_prepare_archive_metrics_and_literature_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper, refs, overrides = small_paper(root)
            geometry, table = fake_geometry(root, refs, {"s1": 3, "s2": 3})
            with patch.multiple(paper_eval, **overrides):
                protocol = paper_eval.prepare(geometry, paper, table, root / "protocol", per_category=2, scene_count=2)
                _, rows, targets = paper_eval.load_protocol(root / "protocol", geometry)
                self.assertEqual(len(rows), 10)
                self.assertTrue(all("answer" not in row and "conversations" not in row for row in rows))
                paper_eval.import_archive(root / "protocol", geometry, paper, root / "archive")
                report = paper_eval.evaluate(root / "protocol", geometry, root / "archive/predictions.jsonl",
                                             paper, root / "metrics.json")
                self.assertEqual(report["summary"]["accuracy"], 1.)
                self.assertEqual(report["opposite_pairs"]["n"], 3)
                candidate = copy.deepcopy(report)
                candidate["run"]["model_kind"] = "reasoning"
                candidate["run"]["decoding"]["seed"] = 42
                candidate["rows"][0]["correct"] = False
                result = paper_eval.compare(report, candidate, literature=True)
                self.assertEqual((result["n"], result["regressed"]), (10, 1))
                self.assertEqual(result["protocol_id"], protocol["protocol_id"])
                def scene(_ident):
                    return {"features": np.ones((3, 768), np.float32),
                            "frame_times": np.arange(3, dtype=np.float64) * .5,
                            "frame_geometry": np.tile(np.eye(4), (3, 1, 1))}
                geometry.scene = scene
                dataset = SimpleNamespace(geometry=geometry, manifest={"dataset_id": "reasoning-fixture"})
                baseline = root / "baseline.json"
                write_json(baseline, {"fixture": True})
                net, tokenizer = model(), CharacterTokenizer()
                args = (root / "protocol", dataset, net, tokenizer, baseline, baseline, root / "model-predictions")
                first = paper_eval.infer(*args, model_kind="original_b3", max_new_tokens=2)
                resumed = paper_eval.infer(*args, model_kind="original_b3", max_new_tokens=2, resume=True)
                self.assertEqual(first, resumed)
                scored = paper_eval.evaluate(root / "protocol", geometry, root / "model-predictions/predictions.jsonl",
                                             paper, root / "model-metrics.json")
                self.assertEqual(scored["summary"]["n"], 10)
                question_path = paper / "data/b4dl_dataset/hallucination_v2/object_existence.json"
                question_path.write_text(question_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "question file hash mismatch"):
                    paper_eval.prepare(geometry, paper, table, root / "bad", per_category=2, scene_count=2)

    def test_shared_prefix_tscd_and_reproducible_sampling(self):
        tokenizer = CharacterTokenizer()
        for build, row in ((model, item("Question")), (lambda: regional_model(5)[0], regional_item())):
            with self.subTest(regional="cell_features" in row):
                net = build()
                net.eval()
                state = torch.get_rng_state().clone()
                clean_cache, _ = paper_eval._prefill(net, tokenizer, row, 3)
                other_cache, _ = paper_eval._prefill(net, tokenizer, row, 3)
                self.assertNotEqual(clean_cache.past_key_values[0][0].data_ptr(),
                                    other_cache.past_key_values[0][0].data_ptr())
                plain = paper_eval.generate_answer(net, tokenizer, row, seed=42, max_new_tokens=3)
                contrast = paper_eval.generate_answer(net, tokenizer, row, seed=42, max_new_tokens=3, alpha=1.)
                self.assertEqual((plain["text"], plain["status"]), (contrast["text"], contrast["status"]))
                self.assertNotEqual(contrast["permutation"], [0, 1, 2])
                torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("B4DL_PAPER_ROOT"), "official paper files not supplied")
    def test_official_questions_and_published_predictions(self):
        paper = Path(os.environ["B4DL_PAPER_ROOT"])
        refs = read_json(paper / "data/manifests/benchmark_scenes.json")["scenes"]
        length_rows = read_jsonl(paper / "data/b4dl_eval/hallucination_v2_temporal_shuffle_cd/object_existence_predictions.jsonl")
        lengths = {row["scene_id"]: len(row["shuffle_permutation"].split()) for row in length_rows}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry, table = fake_geometry(root, refs, lengths)
            paper_eval.prepare(geometry, paper, table, root / "protocol")
            paper_eval.import_archive(root / "protocol", geometry, paper, root / "archive")
            report = paper_eval.evaluate(root / "protocol", geometry, root / "archive/predictions.jsonl",
                                         paper, root / "metrics.json")
            self.assertEqual(report["summary"]["n"], 10000)
            self.assertEqual(report["summary"]["accuracy"], .3665)
            self.assertEqual(report["opposite_pairs"]["n"], 348)
            self.assertEqual(report["opposite_pairs"]["both_correct"], 43)


if __name__ == "__main__":
    unittest.main()
