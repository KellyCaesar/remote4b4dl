import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from b4dl_core.integration import attach, language_model
from b4dl_core.reasoning.policy import (add_reference_adapter, decode_completion,
    expanded_log_probs, extension_state, grpo_sequence_batch, load_checkpoint, reference_policy, save_checkpoint,
    sequence_batch, step_token_ends, teacher_completions)
from b4dl_core.reasoning.protocol import encode_trace, parse_trace
from b4dl_core.reasoning.rewards import score_trace
from b4dl_core.reasoning.trainer import (FP32OutputHead, _optimizer, advantages, clipped_loss, grpo_update,
                                          reset_gpu_memory, rollout_group, sample_grpo, sampling_difference, sft_update,
                                          use_fp32_output_head)
from b4dl_core.temporal import TemporalConfig
from vtimellm.model.vtimellm_llama import VTimeLLMConfig, VTimeLLMLlamaForCausalLM
from test_reasoning_protocol import example


class CharacterTokenizer:
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c)+3 for c in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(int(i)-3) for i in ids if int(i) >= 3)


def model():
    torch.manual_seed(7)
    config = VTimeLLMConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=2,
        attention_dropout=0.)
    config._attn_implementation = "sdpa"
    raw = VTimeLLMLlamaForCausalLM(config)
    raw.get_model().mm_projector = torch.nn.Linear(768, 32)
    wrapped, _ = attach(raw, "A3", TemporalConfig(layers=1, ff_width=32), lora_rank=4)
    wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return wrapped


def item(prompt="Q"):
    return {"prompt": prompt, "features": np.ones((3, 768), np.float32),
            "frame_times": np.arange(3, dtype=np.float64)*.5,
            "frame_geometry": np.tile(np.eye(4), (3, 1, 1))}


class ReasoningTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tokenizer = CharacterTokenizer()

    def test_cuda_peak_reset_initializes_runtime_before_reset(self):
        calls = Mock()
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.init", calls.init), \
             patch("torch.cuda.reset_peak_memory_stats", calls.reset):
            reset_gpu_memory("cuda:0")
        self.assertEqual([call[0] for call in calls.mock_calls], ["init", "reset"])
        calls.reset.assert_called_once_with(torch.device("cuda:0"))
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                reset_gpu_memory("cuda:0")

    def test_frozen_output_head_stays_fp32_under_autocast(self):
        head = torch.nn.Linear(4, 3).requires_grad_(False)
        stable = FP32OutputHead(head)
        inputs = torch.randn(2, 4, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = stable(inputs)
        expected = torch.nn.functional.linear(inputs, head.weight, head.bias)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertIsNone(stable.weight.grad)
        net = model()
        use_fp32_output_head(net)
        self.assertIsInstance(language_model(net).get_output_embeddings(), FP32OutputHead)

    def test_group_advantage_and_scalar_control(self):
        inputs = ([1., 0.], [[1., 0.], [0., 1.]], [[2, 3], [3, 4]], [4, 5], [True, True])
        process = advantages(*inputs, "process", .5)
        scalar = advantages(*inputs, "scalar", .5)
        torch.testing.assert_close(process[0], torch.tensor([1., 1., .75, 1.]))
        torch.testing.assert_close(scalar[0], torch.ones(4))
        equal = advantages([1., 1.], [[.5, .5], [.5, .5]], [[2, 3], [2, 3]], [4, 4], [True, True])
        self.assertTrue(all(v.eq(0).all() for v in equal))
        invalid = advantages([-1., 1.], [[0., 0.], [1., 1.]], [[], [2, 3]], [4, 4], [False, True])
        self.assertTrue(invalid[0].eq(-1).all())

    def test_malformed_rollouts_do_not_change_valid_process_baseline(self):
        outcomes = [0., 0., -1.]
        scores = [[1., .75], [1., .75], [0., 0.]]
        ends = [[2, 3], [2, 3], []]
        valid = [True, True, False]
        outcome = advantages(outcomes, scores, ends, [4, 4, 4], valid, "outcome")
        for mode in ("process", "scalar"):
            actual = advantages(outcomes, scores, ends, [4, 4, 4], valid, mode)
            for expected, vector in zip(outcome, actual):
                torch.testing.assert_close(vector, expected)

        scores[1] = [0., 0.]
        process = advantages(outcomes, scores, ends, [4, 4, 4], valid, "process")
        self.assertGreater(float(process[0][0]), float(outcome[0][0]))
        self.assertLess(float(process[1][0]), float(outcome[1][0]))
        torch.testing.assert_close(process[2], outcome[2])

    def test_clipping_and_kl_gradients(self):
        logp = torch.tensor([np.log(1.5)], dtype=torch.float32, requires_grad=True)
        zero = torch.zeros(1)
        loss, _ = clipped_loss(logp, zero, logp.detach(), torch.ones(1), clip=.2, beta=0.)
        self.assertAlmostEqual(float(loss.detach()), -1.2, places=5)
        loss.backward()
        self.assertEqual(float(logp.grad), 0.)
        logp = torch.tensor([np.log(1.5)], dtype=torch.float32, requires_grad=True)
        loss, _ = clipped_loss(logp, zero, logp.detach(), -torch.ones(1), clip=.2, beta=0.)
        loss.backward()
        self.assertGreater(float(logp.grad), 0.)

    def test_sampling_alignment_guard_uses_actual_clip_bounds(self):
        old = torch.zeros(2)
        within = torch.tensor([np.log(.81), np.log(1.19)])
        self.assertGreater(sampling_difference(old, within, .2, cuda=True), .1)
        for ratio in (.79, 1.21, np.exp(-.2754228), np.exp(.2754228)):
            with self.assertRaises(ValueError):
                sampling_difference(torch.zeros(1), torch.tensor([np.log(ratio)]), .2, cuda=True)
        with self.assertRaises(ValueError):
            sampling_difference(torch.zeros(1), torch.tensor([.001]), .2, cuda=False)

    def test_rollout_preserves_log_probs_of_the_sampled_distribution(self):
        net = model(); add_reference_adapter(net); net.eval()
        request, target = example("existence")
        text = encode_trace(target["trace"])
        ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        batch = grpo_sequence_batch(net, self.tokenizer, item(), ids, 512)
        with torch.no_grad():
            recomputed = expanded_log_probs(net, batch)[0].cpu()
        sampled = recomputed + 0.00003
        response = {"ids": ids, "text": text, "status": "ok", "sampled_log_probs": sampled.clone()}
        with patch("b4dl_core.reasoning.trainer.sample_grpo", return_value=response):
            group = rollout_group(net, self.tokenizer, item(), request, target, group_size=1, max_new_tokens=512)
        torch.testing.assert_close(group[0]["old"], sampled, rtol=0, atol=0)
        self.assertGreater(group[0]["logprob_max_difference"], 0)

    def test_fixed_shape_sampler_matches_training_forward_and_keeps_eos(self):
        net = model(); add_reference_adapter(net)
        drawn, shapes = [], []

        def choose(probabilities, count):
            token = [ord("a") + 3, ord("b") + 3, self.tokenizer.eos_token_id][len(drawn)]
            drawn.append((probabilities.detach().clone(), token))
            return torch.tensor([token], device=probabilities.device)

        def capture(module, args, kwargs):
            shapes.append((kwargs["inputs_embeds"].shape[1], kwargs["use_cache"]))

        hook = net.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with patch("torch.multinomial", side_effect=choose):
                sample = sample_grpo(net, self.tokenizer, item(), 8)
            self.assertEqual(sample["ids"], [ord("a") + 3, ord("b") + 3, 2])
            self.assertEqual(sample["status"], "ok")
            for i, (probabilities, token) in enumerate(drawn):
                torch.testing.assert_close(sample["sampled_log_probs"][i], probabilities[token].log(),
                                           atol=1e-6, rtol=0)
            batch = grpo_sequence_batch(net, self.tokenizer, item(), sample["ids"], 8)
            net.train()
            recomputed = expanded_log_probs(net, batch)[0]
            self.assertEqual(len(recomputed), len(sample["ids"]))
            sampling_difference(sample["sampled_log_probs"], recomputed.detach().cpu(), .2, cuda=False)
            (-recomputed.mean()).backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                and p.grad.abs().sum() > 0 for n, p in net.named_parameters()
                                if ".lora_" in n and ".default." in n))
            self.assertTrue(shapes)
            self.assertEqual(len({shape for shape, _ in shapes}), 1)
            self.assertTrue(all(cache is False for _, cache in shapes))
        finally:
            hook.remove()

    def test_fixed_shape_sampler_truncation_and_causal_padding(self):
        net = model(); add_reference_adapter(net); net.eval()
        tokens = [ord("a") + 3, ord("b") + 3, ord("c") + 3]
        with patch("torch.multinomial", side_effect=[torch.tensor([token]) for token in tokens]):
            sample = sample_grpo(net, self.tokenizer, item(), len(tokens))
        self.assertEqual(sample["ids"], tokens)
        self.assertEqual(sample["status"], "truncated")
        with torch.no_grad():
            score = expanded_log_probs(net, grpo_sequence_batch(net, self.tokenizer, item(), tokens, 3))[0]
            longer = grpo_sequence_batch(net, self.tokenizer, item(), tokens, 8)
            reference = expanded_log_probs(net, longer)[0]
            longer["input_ids"][0, -5:] = ord("z") + 3
            changed_suffix = expanded_log_probs(net, longer)[0]
        sampling_difference(sample["sampled_log_probs"], score.cpu(), .2, cuda=False)
        torch.testing.assert_close(reference, changed_suffix, atol=1e-6, rtol=0)
        with self.assertRaises(ValueError):
            grpo_sequence_batch(net, self.tokenizer, item(), tokens, 2)
        with self.assertRaises(ValueError):
            grpo_sequence_batch(net, self.tokenizer, item(), [], 8)

    def test_fixed_shape_sampler_handles_immediate_eos_and_forbidden_token(self):
        net = model(); add_reference_adapter(net)
        with patch("torch.multinomial", return_value=torch.tensor([2])):
            sample = sample_grpo(net, self.tokenizer, item(), 8)
        self.assertEqual((sample["ids"], sample["status"]), ([2], "ok"))
        with torch.no_grad():
            logp = expanded_log_probs(net, grpo_sequence_batch(net, self.tokenizer, item(), [2], 8))[0]
        sampling_difference(sample["sampled_log_probs"], logp.cpu(), .2, cuda=False)
        with patch("torch.multinomial", side_effect=[torch.tensor([1]), torch.tensor([2])]):
            sample = sample_grpo(net, self.tokenizer, item(), 8)
        self.assertEqual((sample["ids"], sample["status"]), ([1, 2], "error"))
        with self.assertRaises(ValueError):
            sample_grpo(net, self.tokenizer, item(), 0)

    def test_fixed_shape_bfloat16_cpu_alignment(self):
        net = model(); use_fp32_output_head(net); add_reference_adapter(net)
        tokens = [ord("a") + 3, ord("b") + 3, self.tokenizer.eos_token_id]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), \
             patch("torch.multinomial", side_effect=[torch.tensor([token]) for token in tokens]):
            sample = sample_grpo(net, self.tokenizer, item(), 8)
            net.train()
            recomputed = expanded_log_probs(net, grpo_sequence_batch(net, self.tokenizer, item(), tokens, 8))[0]
        sampling_difference(sample["sampled_log_probs"], recomputed.detach().cpu(), .2, cuda=True)
        torch.testing.assert_close(sample["sampled_log_probs"], recomputed.detach().cpu(), atol=1e-4, rtol=0)

    def test_grpo_guard_rejects_mismatch_before_optimizer_step(self):
        net = model(); add_reference_adapter(net); net.eval()
        model_input = item()
        ids = self.tokenizer.encode("abc") + [self.tokenizer.eos_token_id]
        with torch.no_grad():
            batch = grpo_sequence_batch(net, self.tokenizer, model_input, ids, 8)
            old = expanded_log_probs(net, batch)[0].cpu()
            with reference_policy(net):
                ref = expanded_log_probs(net, batch)[0].cpu()
        samples = [{"ids": ids, "old": old.clone(), "ref": ref, "ends": [],
                    "score": {"answer_reward": reward, "step_scores": [0.], "valid": False},
                    "logprob_max_difference": 0.} for reward in (0., 1.)]
        samples[1]["old"] += .01
        optimizer = _optimizer(net, .001)
        before = extension_state(net)
        recipe = {"group_size": 2, "max_new_tokens": 8, "mode": "process",
                  "process_weight": .5, "clip": .2, "beta": .04}
        with patch("b4dl_core.reasoning.trainer.rollout_group", return_value=samples), \
             patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(ValueError, "sampled and recomputed"):
                grpo_update(net, self.tokenizer, [(model_input, {}, {})], optimizer, recipe)
            step.assert_not_called()
        for name, value in extension_state(net).items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_online_grpo_fixed_shape_reaches_finite_gradients_without_update(self):
        net = model(); add_reference_adapter(net)
        optimizer = _optimizer(net, .001)
        before, captured = extension_state(net), []
        tokens = [ord("a") + 3, ord("b") + 3, 2, ord("c") + 3, ord("d") + 3, 2]
        scores = [{"answer_reward": reward, "step_scores": [0.], "valid": False}
                  for reward in (0., 1.)]
        recipe = {"group_size": 2, "max_new_tokens": 8, "mode": "process",
                  "process_weight": .5, "clip": .2, "beta": .04}

        def inspect_gradients():
            captured.extend(p.grad.detach().clone() for p in net.parameters() if p.grad is not None)

        with patch("torch.multinomial", side_effect=[torch.tensor([token]) for token in tokens]), \
             patch("b4dl_core.reasoning.trainer.score_trace", side_effect=scores), \
             patch.object(optimizer, "step", side_effect=inspect_gradients) as step:
            result = grpo_update(net, self.tokenizer, [(item(), {}, {})], optimizer, recipe)
            step.assert_called_once()
        self.assertEqual(result["completions"], 2)
        self.assertTrue(captured and all(torch.isfinite(grad).all() for grad in captured))
        self.assertGreater(sum(float(grad.abs().sum()) for grad in captured), 0.)
        for name, value in extension_state(net).items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_token_boundaries_use_actual_completion_ids(self):
        request, target = example("motion")
        text = encode_trace(target["trace"])
        ids = self.tokenizer.encode(text) + [2]
        char_ends = parse_trace(text, request)["step_char_ends"]
        self.assertEqual(step_token_ends(self.tokenizer, ids, char_ends), char_ends)
        self.assertEqual(decode_completion(self.tokenizer, ids), text)

    def test_slow_sentencepiece_token_boundaries(self):
        import sentencepiece as spm
        from transformers import LlamaTokenizer
        request, target = example("motion")
        text = encode_trace(target["trace"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root/"corpus.txt"
            corpus.write_text((text + "\n" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[],:._- \n")*20,
                              encoding="utf-8")
            spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root/"tiny"), vocab_size=128,
                model_type="bpe", character_coverage=1.0, hard_vocab_limit=False, minloglevel=2)
            tokenizer = LlamaTokenizer(vocab_file=str(root/"tiny.model"))
            ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            decoded = decode_completion(tokenizer, ids)
            ends = parse_trace(decoded, request)["step_char_ends"]
            positions = step_token_ends(tokenizer, ids, ends)
            self.assertEqual(len(positions), len(ends))
            for end, position in zip(ends, positions):
                self.assertGreaterEqual(len(decode_completion(tokenizer, ids[:position])), end)

    def test_multimodal_expanded_labels_and_sft_gradients(self):
        net = model()
        request, target = example("existence")
        ids = teacher_completions(self.tokenizer, [target], 512)[0]
        batch = sequence_batch(net, self.tokenizer, [item()], [ids])
        logp = expanded_log_probs(net, batch)[0]
        self.assertEqual(logp.numel(), len(ids))
        (-logp.mean()).backward()
        core = language_model(net).get_model()
        self.assertGreater(float(core.mm_projector.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(core.frame_adapter.output.weight.grad.abs().sum()), 0.)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in net.named_parameters() if ".lora_" in n))
        self.assertTrue(all(p.grad is None for p in net.parameters() if not p.requires_grad))

    def test_reference_is_sft_copy_and_checkpoint_restores_all_state(self):
        net = model()
        _, target = example("existence")
        sft_opt = _optimizer(net, .001)
        sft_update(net, self.tokenizer, [(item(), target)], sft_opt, 512)
        before = extension_state(net)
        add_reference_adapter(net)
        ids = self.tokenizer.encode("abcdefgh") + [2]
        batch = sequence_batch(net, self.tokenizer, [item()], [ids])
        net.eval()
        with torch.no_grad():
            policy_before = expanded_log_probs(net, batch)[0].clone()
            with reference_policy(net): reference_before = expanded_log_probs(net, batch)[0].clone()
        torch.testing.assert_close(policy_before, reference_before, atol=0, rtol=0)
        optimizer = _optimizer(net, .001)
        net.train()
        loss = -expanded_log_probs(net, batch)[0].mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        net.eval()
        with torch.no_grad():
            with reference_policy(net): reference_after = expanded_log_probs(net, batch)[0].clone()
        torch.testing.assert_close(reference_before, reference_after, atol=0, rtol=0)
        after = extension_state(net)
        for key, value in before.items():
            if ".frame_adapter." in key or ".mm_projector." in key:
                torch.testing.assert_close(after[key], value, atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if ".lora_" in k))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"checkpoint.pt"
            save_checkpoint(path, net, {"identity": "fixture"}, "grpo", {"seed": 42}, 1, optimizer)
            restored = model(); add_reference_adapter(restored)
            opt2 = _optimizer(restored, .001)
            load_checkpoint(path, restored, {"identity": "fixture"}, "grpo", opt2)
            for key, value in extension_state(restored).items():
                torch.testing.assert_close(value, after[key], atol=0, rtol=0)
            with self.assertRaises(ValueError): load_checkpoint(path, restored, {"identity": "different"})
            def next_update(current, opt):
                current.train()
                data = sequence_batch(current, self.tokenizer, [item()], [ids])
                next_loss = -expanded_log_probs(current, data)[0].mean()
                next_loss.backward(); opt.step(); opt.zero_grad()
            next_update(net, optimizer)
            next_update(restored, opt2)
            expected = extension_state(net)
            for key, value in extension_state(restored).items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_generated_and_recomputed_log_probs_match(self):
        net = model(); add_reference_adapter(net); net.eval()
        request, target = example("existence")
        group = rollout_group(net, self.tokenizer, item(), request, target, group_size=2, max_new_tokens=8)
        self.assertEqual(len(group), 2)
        self.assertTrue(all(row["logprob_max_difference"] < 1e-4 for row in group))
        self.assertTrue(all(len(row["ids"]) == len(row["old"]) == len(row["ref"]) for row in group))

    def test_online_grpo_with_zero_variance_does_not_update_weights(self):
        net = model(); add_reference_adapter(net)
        request, target = example("existence")
        before = extension_state(net)
        optimizer = _optimizer(net, .001)
        recipe = {"group_size": 2, "max_new_tokens": 8, "mode": "process",
                  "process_weight": .5, "clip": .2, "beta": .04}
        result = grpo_update(net, self.tokenizer, [(item(), request, target)], optimizer, recipe)
        self.assertEqual(result["completions"], 2)
        self.assertEqual(result["valid_rate"], 0.)
        self.assertEqual(result["zero_outcome_variance_rate"], 1.)
        for key, value in extension_state(net).items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_grpo_update_uses_grounded_step_advantages_and_freezes_reference(self):
        net = model(); add_reference_adapter(net)
        request, target = example("existence")
        before = extension_state(net)
        optimizer = _optimizer(net, .001)
        recipe = {"group_size": 2, "max_new_tokens": 512, "mode": "process",
                  "process_weight": .5, "clip": .2, "beta": .04}
        def verified_samples(current, tokenizer, model_input, req, truth, group_size, max_new_tokens, clip):
            wrong = copy.deepcopy(truth["trace"])
            wrong["steps"][0]["value"]["count"] = 0
            wrong["steps"][1]["value"] = False
            wrong["answer"] = "No"
            result = []
            current.eval()
            for trace in (truth["trace"], wrong):
                text = encode_trace(trace)
                ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
                batch = grpo_sequence_batch(current, tokenizer, model_input, ids, max_new_tokens)
                with torch.no_grad():
                    old = expanded_log_probs(current, batch)[0].cpu()
                    with reference_policy(current): ref = expanded_log_probs(current, batch)[0].cpu()
                score = score_trace(req, truth, text)
                result.append({"ids": ids, "text": text, "status": "ok", "old": old, "ref": ref,
                               "score": score, "ends": step_token_ends(tokenizer, ids, score["step_char_ends"]),
                               "logprob_max_difference": 0.})
            return result
        with patch("b4dl_core.reasoning.trainer.rollout_group", verified_samples):
            result = grpo_update(net, self.tokenizer, [(item(), request, target)], optimizer, recipe)
        self.assertEqual(result["completions"], 2)
        self.assertGreater(result["gradient_norm"], 0.)
        after = extension_state(net)
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if ".lora_" in k and ".default." in k))
        for key in before:
            if ".reference." in key or ".frame_adapter." in key or ".mm_projector." in key:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
