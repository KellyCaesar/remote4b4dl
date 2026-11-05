"""Check mixed training labels, evidence losses and exact resume on tiny models."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from b4dl_core.data import build_dataset
from b4dl_core.io import read_json, read_jsonl, write_json
from b4dl_core.regional_data import RegionalDataset
from b4dl_core.reasoning.data import ReasoningDataset, build_data, public_request
from b4dl_core.reasoning.evaluation import compare_reports, evaluate, infer, load_model, validate_ablation_lineage
from b4dl_core.reasoning.mixed import MixedPlan, build_plan
from b4dl_core.reasoning.protocol import encode_trace, make_trace
from b4dl_core.reasoning.supervision import (ObservationHead, observation_loss, pair_regularizer,
                                             segmented_trace_loss, short_answer_scores)
from b4dl_core.reasoning.trainer import train, use_fp32_output_head
from b4dl_core.integration import attach, language_model
from test_reasoning_data import raw_fixture
from test_reasoning_protocol import example
from test_reasoning_training import CharacterTokenizer
from test_regional_pipeline import regional_item, regional_model, write_regional_fixture
from vtimellm.model.vtimellm_llama import VTimeLLMConfig, VTimeLLMLlamaForCausalLM


def objective():
    return {"schema": "b4dl-mixed-objective-v1", "segmented": True,
            "numeric_weight": .2, "pair_weight": .1, "geometry_weight": .5,
            "replay_weight": .5, "trace_weight": .5, "short_weight": .5, "pair_margin": .5}


def make_dataset(root):
    tables, features, support = raw_fixture(root)
    build_dataset(tables, root / "metadata.json", features, root / "splits.json", root / "geometry", input_support=support)
    build_data(root / "geometry", tables, support, root / "reasoning")
    dataset = ReasoningDataset(root / "reasoning", root / "geometry", "train")
    write_regional_fixture(dataset.geometry, root / "regional")
    return RegionalDataset(dataset, root / "regional")


def make_replay_source(root, dataset):
    scenes = dataset.geometry.manifest["scenes"]
    train_scene = next(scene for scene, record in scenes.items() if record["split"] == "train")
    other_scene = next(scene for scene, record in scenes.items() if record["split"] != "train")
    source = [
        {"scene_id": train_scene, "conversations": [{"from": "human", "value": "<video>\nWhat is visible?<meta>secret"},
                                                      {"from": "gpt", "value": "A car."}]},
        {"scene_id": other_scene, "conversations": [{"from": "human", "value": "<video>\nHeld-out?"},
                                                      {"from": "gpt", "value": "Hidden."}]},
        {"scene_id": train_scene, "conversations": [{"from": "human", "value": "Bad."}]},
    ]
    path = root / "original.json"
    write_json(path, source)
    return path


class RegionalSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tokenizer = CharacterTokenizer()

    def test_numeric_head_uses_public_question_and_backpropagates(self):
        model, _ = regional_model()
        core = language_model(model).get_model()
        core.numeric_head = ObservationHead(768, 32, 8)
        request, target = example("position")
        request = public_request(request)
        row = regional_item()
        row["prompt"] = request["prompt"]
        loss, predicted = observation_loss(model, self.tokenizer, row, request, target)
        changed = copy.deepcopy(target)
        changed["trace"]["steps"][0]["value"]["x"] += 7.
        other_loss, other_predicted = observation_loss(model, self.tokenizer, row, request, changed)
        self.assertEqual(predicted, other_predicted)
        self.assertNotEqual(float(loss.detach()), float(other_loss.detach()))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.numeric_head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.regional_adapter.local_pool.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_segmented_trace_and_short_answer_are_differentiable(self):
        model, _ = regional_model()
        request, target = example("existence")
        request = public_request(request)
        row = regional_item()
        row["prompt"] = request["prompt"]
        trace, parts = segmented_trace_loss(model, self.tokenizer, row, request, target, 512)
        options, scores = short_answer_scores(model, self.tokenizer, row, request, 512)
        self.assertEqual(options, ("Yes", "No"))
        self.assertEqual(set(parts), {"observe", "derive", "answer"})
        (trace - torch.log_softmax(scores, dim=0)[0]).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_pair_labels_and_option_permutation(self):
        request, target = example("existence")
        first = public_request(request)
        second, negative = copy.deepcopy(first), copy.deepcopy(target)
        second.update(scene_id="other", request_id="other")
        negative.update(request_id="other", answer="No")
        negative["trace"] = make_trace(second, {**negative, "evidence": {"count": 0}})
        scores = [(('Yes', 'No'), torch.tensor([2., -1.], requires_grad=True)),
                  (('Yes', 'No'), torch.tensor([-1., 2.], requires_grad=True))]
        self.assertEqual(float(pair_regularizer([first, second], [target, negative], scores, "opposite").detach()), 0.)
        bad = [(scores[0][0], -scores[0][1]), (scores[1][0], -scores[1][1])]
        loss = pair_regularizer([first, second], [target, negative], bad, "opposite")
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertTrue(all(torch.isfinite(row[1].grad).all() for row in scores))
        request, target = example("distance")
        first = public_request({**request, "semantic_id": "event", "option_rotation": 0})
        second = copy.deepcopy(first)
        second["option_rotation"] = 1
        second["options"] = {"A": "truck", "B": "bus", "C": "bicycle", "D": "car"}
        rotated = {**target, "answer": "D"}
        first_scores = torch.tensor([4., 3., 2., 1.])
        second_scores = torch.tensor([3., 2., 1., 4.])
        pair = [(tuple("ABCD"), first_scores), (tuple("ABCD"), second_scores)]
        self.assertAlmostEqual(float(pair_regularizer([first, second], [target, rotated], pair, "rotation")), 0., places=7)
        second["scene_id"] = "another"
        with self.assertRaisesRegex(ValueError, "rotation pair"):
            pair_regularizer([first, second], [target, rotated], pair, "rotation")

    def test_mixed_plan_reproducible_train_only_and_tamper_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            source = make_replay_source(root, dataset)
            first = build_plan(dataset, [source], root / "plan1", seed=42, updates=3)
            second = build_plan(dataset, [source], root / "plan2", seed=42, updates=3)
            self.assertEqual(first["plan_id"], second["plan_id"])
            self.assertEqual(first["excluded"]["outside_train_scenes"], 1)
            self.assertEqual(first["excluded"]["unsupported_conversation"], 1)
            plan = MixedPlan(root / "plan1", dataset)
            self.assertEqual(len(plan.plan), 3)
            held_out = RegionalDataset(ReasoningDataset(root / "reasoning", root / "geometry", "test"), root / "regional")
            self.assertEqual(MixedPlan(root / "plan1", held_out).manifest["plan_id"], plan.manifest["plan_id"])
            self.assertEqual(next(iter(plan.replay.values()))["prompt"], "What is visible?")
            replay_input, target = plan.replay_input(dataset, plan.plan[0]["replay"][0])
            self.assertEqual(target, "A car.")
            self.assertNotIn(target, replay_input["prompt"])
            self.assertIn("cell_features", replay_input)
            path = root / "plan1" / "plan.jsonl"
            path.write_text(path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                MixedPlan(root / "plan1", dataset)

    def test_original_b3_structured_inference_and_matched_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root).base
            dataset.requests = dataset.requests[:1]
            config = VTimeLLMConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=2)
            config._attn_implementation = "sdpa"
            raw = VTimeLLMLlamaForCausalLM(config)
            raw.get_model().mm_projector = torch.nn.Linear(768, 32)
            model, _ = attach(raw, "A0")
            write_json(root / "baseline.json", {"fixture": True})
            write_json(root / "preparation.json", {"fixture": True})
            args = (dataset, model, self.tokenizer, root / "baseline.json", root / "b3")
            kwargs = {"max_new_tokens": 2, "model_kind": "original_b3", "baseline": root / "baseline.json",
                      "preparation": root / "preparation.json"}
            receipt = infer(*args, **kwargs)
            self.assertEqual(receipt, infer(*args, resume=True, **kwargs))
            run = receipt["run"]
            self.assertEqual(run["model_kind"], "original_b3")
            self.assertIsNone(run["training"])
            report = evaluate(dataset, read_jsonl(root / "b3/predictions.jsonl"), run, repetitions=10)
            candidate = copy.deepcopy(report)
            candidate["run"]["model_kind"] = "reasoning"
            candidate["run"]["checkpoint_sha256"] = "candidate"
            candidate["run"]["training"] = {"stage": "sft", "recipe": {}, "step": 1,
                "execution": {"device": "cpu"}, "planned_full_budget": False}
            candidate["run"]["lineage"]["sft_origin_sha256"] = "candidate"
            self.assertEqual(compare_reports(report, candidate, repetitions=10, mode="original_b3")["n"], 1)
            candidate["run"]["lineage"]["common"]["baseline_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "matched baseline"):
                compare_reports(report, candidate, repetitions=10, mode="original_b3")

    def test_observation_diagnostics_and_ablation_identity(self):
        request, target = example("position")
        dataset = SimpleNamespace(requests=[request], targets={"r": target}, manifest={"dataset_id": "fixture"})
        run = {"dataset_id": "fixture", "split": "train", "observation_head": True}
        prediction = {"request_id": "r", "input_fingerprint": "parent", "status": "ok",
            "text": encode_trace(target["trace"]), "generated_tokens": 10,
            "observation_head": {"values": {"x": 10., "y": 0.}, "rule_answer": "Yes"}}
        report = evaluate(dataset, [prediction], run, repetitions=10)
        self.assertEqual(report["observation_head"]["rule_answer_accuracy"], 1.)
        self.assertEqual(report["observation_head"]["generated_answer_agreement"], 1.)
        self.assertEqual(report["observation_head"]["numeric"]["x"]["mae"], 0.)
        self.assertEqual(report["rows"][0]["head_generated_absolute_difference"]["x"], 0.)
        prediction["observation_head"]["rule_answer"] = "No"
        with self.assertRaisesRegex(ValueError, "observation-head"):
            evaluate(dataset, [prediction], run, repetitions=10)
        common = {"baseline_sha256": "base", "preparation_sha256": "prep", "reasoning_dataset_id": "data",
                  "implementation": "impl", "reward_version": "reward", "mixed_plan_id": "plan",
                  "options": {"arm": "A3"}}
        training = {"stage": "sft", "recipe": {"seed": 42}, "step": 3,
                    "execution": {"device": "cpu"}, "planned_full_budget": True}
        first = {"lineage": {"common": common}, "training": training}
        second = copy.deepcopy(first)
        second["lineage"]["common"]["options"] = {"arm": "regional"}
        second["lineage"]["common"]["regional_dataset_id"] = "region"
        self.assertEqual(validate_ablation_lineage(first, second)["candidate_regional_dataset_id"], "region")
        second["lineage"]["common"]["mixed_plan_id"] = "another"
        with self.assertRaisesRegex(ValueError, "same mixed"):
            validate_ablation_lineage(first, second)

    def test_mixed_training_checkpoint_resume_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            source = make_replay_source(root, dataset)
            build_plan(dataset, [source], root / "plan", seed=42, updates=2)
            plan = MixedPlan(root / "plan", dataset)
            write_json(root / "baseline.json", {"fixture": True})
            write_json(root / "preparation.json", {"fixture": True})
            recipe = {"seed": 42, "lora_rank": 4, "sft_lr": .0001, "grpo_lr": .001,
                      "group_size": 2, "grad_accum": 4, "max_new_tokens": 512, "clip": .2,
                      "beta": .04, "process_weight": .5, "mode": "process", "save_every": 1,
                      "engineering": True, "max_updates": 2}
            config = regional_model()[1]["regional"]
            from b4dl_core.regional_model import RegionalConfig
            regional_config = RegionalConfig(**config)
            def prepare(*args):
                model, options = regional_model()
                model.get_base_model().get_model().numeric_head = ObservationHead(768, 32, 8)
                options["supervision"] = objective()
                use_fp32_output_head(model)
                return CharacterTokenizer(), model, options
            with patch("b4dl_core.reasoning.trainer.prepare_model", prepare):
                train(dataset, root / "baseline.json", root / "preparation.json", root / "first", "sft", recipe,
                      device="cpu", regional_config=regional_config, mixed_plan=plan, objective=objective())
                train(dataset, root / "baseline.json", root / "preparation.json", root / "resumed", "sft", recipe,
                      device="cpu", regional_config=regional_config, mixed_plan=plan, objective=objective(),
                      resume=root / "first" / "step-1.pt")
                _, restored, meta = load_model(dataset, root / "baseline.json", root / "preparation.json",
                                                root / "first" / "step-2.pt", "cpu", plan)
            first = torch.load(root / "first" / "step-2.pt", map_location="cpu", weights_only=True)
            resumed = torch.load(root / "resumed" / "step-2.pt", map_location="cpu", weights_only=True)
            self.assertEqual(first["identity"]["common"]["mixed_plan_id"], plan.manifest["plan_id"])
            self.assertIn("regional", meta["identity"]["common"]["options"])
            self.assertTrue(hasattr(language_model(restored).get_model(), "numeric_head"))
            for name, tensor in first["parameters"].items():
                torch.testing.assert_close(tensor, resumed["parameters"][name], atol=0, rtol=0)
            self.assertEqual(read_json(root / "first" / "completion.json")["updates"], 2)


if __name__ == "__main__":
    unittest.main()
