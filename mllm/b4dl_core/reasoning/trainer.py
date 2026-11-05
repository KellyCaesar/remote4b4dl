"""SFT and token-level process GRPO using the B4DL multimodal model."""
from contextlib import nullcontext
from pathlib import Path
import math
import random
import time

import numpy as np
import torch

from ..b3 import chosen_config, load_original, verify_sst_binding
from ..integration import FRAME_KEYS, attach, language_model, padding_side
from ..io import digest, sha256, write_json
from ..resume import append_jsonl
from ..runtime import runtime_identity
from . import REWARD_VERSION
from .policy import (add_reference_adapter, decode_completion, expanded_log_probs, grpo_sequence_batch,
                     load_checkpoint, reference_policy, save_checkpoint, sequence_batch,
                     step_token_ends, teacher_completions)
from .rewards import score_trace


def standardize(values, dim=0, threshold=1e-4):
    values = torch.as_tensor(values, dtype=torch.float32)
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, unbiased=False, keepdim=True)
    return torch.where(std >= threshold, (values-mean)/std.clamp_min(threshold), torch.zeros_like(values))


def advantages(outcomes, step_scores, ends, lengths, valid, mode="process", weight=.5):
    """Group-relative outcome and corresponding-step process advantages."""
    if mode not in ("outcome", "scalar", "process") or not 0 <= weight < float("inf"):
        raise ValueError("invalid GRPO advantage configuration")
    if not lengths or any(n < 1 for n in lengths) or not (len(outcomes) == len(step_scores) == len(ends) == len(valid) == len(lengths)):
        raise ValueError("incompatible rollout group")
    rewards = torch.as_tensor(step_scores, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] < 1 or not torch.isfinite(rewards).all():
        raise ValueError("invalid process rewards")
    terminal = standardize(outcomes)
    # A malformed completion has no template steps. Including its synthetic
    # zeroes in the process baseline would also reward formatting in every
    # valid completion, even when their process scores are identical.
    steps = torch.zeros_like(rewards)
    valid_indices = [i for i, is_valid in enumerate(valid) if is_valid]
    if valid_indices:
        steps[valid_indices] = standardize(rewards[valid_indices], dim=0)
    result = []
    for i, length in enumerate(lengths):
        vector = torch.full((length,), float(terminal[i]), dtype=torch.float32)
        if mode != "outcome" and valid[i]:
            if len(ends[i]) != rewards.shape[1] or any(not 1 <= x <= length for x in ends[i]) or ends[i] != sorted(ends[i]):
                raise ValueError("invalid token step boundaries")
            if mode == "scalar":
                vector += weight * steps[i].sum()/rewards.shape[1]
            else:
                for j, end in enumerate(ends[i]):
                    vector[:end] += weight * steps[i, j]/rewards.shape[1]
        result.append(vector)
    return result


def clipped_loss(log_probs, old_log_probs, ref_log_probs, advantage, clip=.2, beta=.04):
    if not (log_probs.shape == old_log_probs.shape == ref_log_probs.shape == advantage.shape) or not log_probs.numel():
        raise ValueError("incompatible token loss tensors")
    ratio = torch.exp(log_probs-old_log_probs)
    ref_delta = ref_log_probs-log_probs
    kl = torch.exp(ref_delta)-ref_delta-1
    objective = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    loss = -objective.mean() + beta*kl.mean()
    if not torch.isfinite(loss) or not torch.isfinite(ratio).all() or not torch.isfinite(kl).all():
        raise FloatingPointError("nonfinite GRPO objective")
    return loss, {"kl": float(kl.detach().mean()), "ratio_mean": float(ratio.detach().mean()),
                  "clip_fraction": float(((ratio.detach()-1).abs() > clip).float().mean())}


def _precision(model):
    device = language_model(model).get_input_embeddings().weight.device
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class FP32OutputHead(torch.nn.Module):
    """Frozen output projection in FP32 while the transformer remains BF16."""
    def __init__(self, head):
        super().__init__()
        if (not isinstance(head, torch.nn.Linear) or head.weight.requires_grad
                or head.bias is not None and head.bias.requires_grad):
            raise ValueError("expected a frozen linear language output head")
        self.in_features, self.out_features = head.in_features, head.out_features
        self.weight = torch.nn.Parameter(head.weight.detach().float(), requires_grad=False)
        self.bias = (torch.nn.Parameter(head.bias.detach().float(), requires_grad=False)
                     if head.bias is not None else None)

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            return torch.nn.functional.linear(values.float(), self.weight, self.bias)


def configure_precision(device):
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def use_fp32_output_head(model):
    base = language_model(model)
    base.set_output_embeddings(FP32OutputHead(base.get_output_embeddings()))


def numerics_identity(model):
    base = language_model(model)
    device = base.get_input_embeddings().weight.device
    return {"output_head_dtype": str(base.get_output_embeddings().weight.dtype),
            "bf16_reduced_precision_reduction":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction if device.type == "cuda" else None}


def gpu_memory(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return {"peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1048576,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/1048576}


def reset_gpu_memory(device):
    """Initialize CUDA before resetting peaks, including model loading in the measurement."""
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)


def sample_one(model, tokenizer, item, max_new_tokens, do_sample=True, record_scores=False):
    batch = sequence_batch(model, tokenizer, [item])
    model.eval()
    started = time.perf_counter()
    with torch.no_grad(), _precision(model), padding_side(language_model(model), "left"):
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=1.0, top_p=1.0, top_k=0, num_beams=1, repetition_penalty=1.0,
            no_repeat_ngram_size=0, min_new_tokens=0, use_cache=True, pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=record_scores)
    ids = generated.sequences[0].tolist()
    if not ids:
        raise ValueError("generation returned no completion tokens")
    eos = tokenizer.eos_token_id
    if eos in ids:
        ids = ids[:ids.index(eos)+1]
        status = "ok"
    else:
        status = "truncated"
    forbidden = set(getattr(tokenizer, "all_special_ids", [])) - {eos}
    if any(token in forbidden for token in ids):
        status = "error"
    sampled = None
    if record_scores:
        if len(generated.sequences[0]) != len(generated.scores):
            raise ValueError("generation returned a prompt prefix; completion identity is ambiguous")
        sampled = torch.stack([torch.log_softmax(generated.scores[t][0].float(), -1)[token]
                               for t, token in enumerate(ids)]).detach().cpu()
    return {"ids": ids, "text": decode_completion(tokenizer, ids), "status": status,
            "sampled_log_probs": sampled, "generation_seconds": time.perf_counter()-started}


def sample_grpo(model, tokenizer, item, max_new_tokens):
    """Sample from the same fixed-shape, causal forward used by GRPO scoring."""
    from vtimellm.constants import IGNORE_INDEX
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("positive GRPO generation budget required")
    eos = tokenizer.eos_token_id
    batch = sequence_batch(model, tokenizer, [item], [[eos] * max_new_tokens])
    base = language_model(model)
    model.eval()
    started = time.perf_counter()
    ids, sampled = [], []
    with torch.no_grad(), _precision(model), padding_side(base, "right"):
        _, positions, attention, _, embeds, labels = base.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, batch["labels"],
            batch["images"], **{key: batch.get(key) for key in FRAME_KEYS})
        target_positions = labels[0].ne(IGNORE_INDEX).nonzero().flatten()
        if (len(target_positions) != max_new_tokens or int(target_positions[0]) < 1
                or not torch.equal(target_positions, torch.arange(
                    target_positions[0], target_positions[0] + max_new_tokens, device=target_positions.device))):
            raise ValueError("fixed-shape completion alignment failed")
        for position in target_positions:
            # Future EOS placeholders cannot affect this position through a causal mask.
            logits = model(inputs_embeds=embeds, attention_mask=attention, position_ids=positions,
                           use_cache=False).logits[0, position - 1].float().clone()
            token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            token_id = int(token.item())
            sampled.append(torch.log_softmax(logits, dim=-1)[token_id].clone())
            ids.append(token_id)
            if ids[-1] == eos:
                break
            embeds[0, position] = base.get_input_embeddings()(token)[0]
    status = "ok" if ids[-1] == eos else "truncated"
    forbidden = set(getattr(tokenizer, "all_special_ids", [])) - {eos}
    if any(token in forbidden for token in ids):
        status = "error"
    return {"ids": ids, "text": decode_completion(tokenizer, ids), "status": status,
            "sampled_log_probs": torch.stack(sampled).detach().cpu(),
            "generation_seconds": time.perf_counter()-started}


def sampling_difference(sampled, recomputed, clip, *, cuda):
    """Validate the actual sampling distribution against the full forward pass.

    BF16 forwards can still differ numerically. The unchanged policy must
    remain strictly inside both PPO clip limits; CPU keeps the tighter alignment
    check used by the tiny-model tests.
    """
    if (sampled.shape != recomputed.shape or not sampled.numel() or not 0 < clip < 1
            or not torch.isfinite(sampled).all() or not torch.isfinite(recomputed).all()):
        raise ValueError("invalid sampled/recomputed log-probabilities")
    delta = recomputed - sampled
    difference = float(delta.abs().max())
    if cuda:
        if (delta <= math.log1p(-clip)).any() or (delta >= math.log1p(clip)).any():
            raise ValueError("sampled and recomputed policy probabilities exceed PPO clip limits: " + str(difference))
    elif difference > 1e-4:
        raise ValueError("sampled and recomputed policy log-probs disagree: " + str(difference))
    return difference


def rollout_group(model, tokenizer, item, request, target, group_size=4, max_new_tokens=512, clip=.2):
    samples = []
    for _ in range(group_size):
        sample = sample_grpo(model, tokenizer, item, max_new_tokens)
        batch = grpo_sequence_batch(model, tokenizer, item, sample["ids"], max_new_tokens)
        with torch.no_grad(), _precision(model):
            recomputed = expanded_log_probs(model, batch)[0].detach().cpu()
            with reference_policy(model):
                ref = expanded_log_probs(model, batch)[0].detach().cpu()
        old = sample.pop("sampled_log_probs")
        difference = sampling_difference(old, recomputed, clip,
            cuda=language_model(model).get_input_embeddings().weight.device.type == "cuda")
        score = score_trace(request, target, sample["text"], sample["status"])
        ends = step_token_ends(tokenizer, sample["ids"], score["step_char_ends"]) if score["valid"] else []
        sample.update(score=score, ends=ends, old=old, ref=ref, logprob_max_difference=difference)
        samples.append(sample)
    return samples


def _optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype != torch.float32 for p in params):
        raise ValueError("optimizer requires FP32 trainable parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.)


def _finish_update(model, optimizer):
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("missing or nonfinite gradients; update not applied")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def sft_update(model, tokenizer, rows, optimizer, max_new_tokens):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    model.train()  # required for the model's gradient-checkpointing path
    for item, target in rows:
        completion = teacher_completions(tokenizer, [target], max_new_tokens)
        batch = sequence_batch(model, tokenizer, [item], completion)
        with _precision(model), padding_side(language_model(model), "right"):
            result = model(**batch, use_cache=False)
            loss = result.loss
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite SFT loss; update not applied")
        (loss/len(rows)).backward()
        losses.append(float(loss.detach()))
    return {"loss": sum(losses)/len(losses), "gradient_norm": _finish_update(model, optimizer)}


def mixed_sft_update(model, tokenizer, geometry_rows, replay_rows, optimizer, recipe, objective, pair_type):
    from torch.nn import functional as F
    from .supervision import observation_loss, pair_regularizer, segmented_trace_loss, short_answer_scores
    if len(geometry_rows) != 2 or len(replay_rows) != 2:
        raise ValueError("mixed updates require two geometry and two replay rows")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    geometry_logs, observations, score_rows = [], [], []
    for item, request, target in geometry_rows:
        with _precision(model):
            if objective["segmented"]:
                trace, parts = segmented_trace_loss(model, tokenizer, item, request, target, recipe["max_new_tokens"])
            else:
                completion = teacher_completions(tokenizer, [target], recipe["max_new_tokens"])
                batch = sequence_batch(model, tokenizer, [item], completion)
                trace = model(**batch, use_cache=False).loss
                parts = {}
        if not torch.isfinite(trace):
            raise FloatingPointError("nonfinite geometry trace loss")
        (objective["geometry_weight"] * objective["trace_weight"] * trace / len(geometry_rows)).backward()
        geometry_logs.append({"trace": float(trace.detach()), **parts})
        if objective["numeric_weight"]:
            numeric, prediction = observation_loss(model, tokenizer, item, request, target)
            if not torch.isfinite(numeric):
                raise FloatingPointError("nonfinite numeric observation loss")
            (objective["geometry_weight"] * objective["numeric_weight"] * numeric / len(geometry_rows)).backward()
            observations.append({"loss": float(numeric.detach()), "prediction": prediction})
        with _precision(model):
            score_rows.append(short_answer_scores(model, tokenizer, item, request, recipe["max_new_tokens"]))
    short_losses = []
    for (_, _, target), (options, scores) in zip(geometry_rows, score_rows):
        if target["answer"] not in options:
            raise ValueError("short answer is missing from model options")
        label = torch.tensor([options.index(target["answer"])], dtype=torch.long, device=scores.device)
        short_losses.append(F.cross_entropy(scores.unsqueeze(0).float(), label))
    short = torch.stack(short_losses).mean()
    pair = pair_regularizer([row[1] for row in geometry_rows], [row[2] for row in geometry_rows],
                            score_rows, pair_type, objective["pair_margin"])
    combined = objective["geometry_weight"] * objective["short_weight"] * short + objective["pair_weight"] * pair
    if not torch.isfinite(combined):
        raise FloatingPointError("nonfinite short-answer or pair loss")
    combined.backward()
    replay_losses = []
    if objective["replay_weight"]:
        for item, answer in replay_rows:
            completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(completion) > recipe["max_new_tokens"]:
                raise ValueError("replay answer exceeds generation budget")
            batch = sequence_batch(model, tokenizer, [item], [completion])
            with _precision(model):
                loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite replay loss")
            (objective["replay_weight"] * loss / len(replay_rows)).backward()
            replay_losses.append(float(loss.detach()))
    return {"trace": geometry_logs, "observations": observations, "short_loss": float(short.detach()),
            "pair_loss": float(pair.detach()), "pair_type": pair_type,
            "short_answer_forwards": sum(len(options) for options, _ in score_rows),
            "replay_loss": sum(replay_losses) / len(replay_losses) if replay_losses else None,
            "gradient_norm": _finish_update(model, optimizer)}


def grpo_update(model, tokenizer, rows, optimizer, recipe):
    """Collect all groups before the update, then backpropagate serially."""
    optimizer.zero_grad(set_to_none=True)
    groups = []
    for item, request, target in rows:
        group = rollout_group(model, tokenizer, item, request, target, recipe["group_size"], recipe["max_new_tokens"], recipe["clip"])
        group_adv = advantages([x["score"]["answer_reward"] for x in group],
            [x["score"]["step_scores"] for x in group], [x["ends"] for x in group],
            [len(x["ids"]) for x in group], [x["score"]["valid"] for x in group],
            recipe["mode"], recipe["process_weight"])
        groups.append((item, group, group_adv))
    count = sum(len(group) for _, group, _ in groups)
    logs, outcomes, valid, step_means, lengths, max_difference = [], [], [], [], [], 0.
    zero_outcome, zero_process = 0, 0
    for _, group, _ in groups:
        outcome = torch.tensor([sample["score"]["answer_reward"] for sample in group])
        valid_steps = [sample["score"]["step_scores"] for sample in group if sample["score"]["valid"]]
        zero_outcome += int(outcome.std(unbiased=False) < 1e-4)
        zero_process += int(len(valid_steps) < 2 or
            (torch.tensor(valid_steps).std(dim=0, unbiased=False) < 1e-4).all())
    model.train()
    for item, group, group_adv in groups:
        for sample, advantage in zip(group, group_adv):
            batch = grpo_sequence_batch(model, tokenizer, item, sample["ids"], recipe["max_new_tokens"])
            with _precision(model):
                logp = expanded_log_probs(model, batch)[0]
                sampling_difference(sample["old"], logp.detach().cpu(), recipe["clip"], cuda=logp.device.type == "cuda")
                loss, metrics = clipped_loss(logp, sample["old"].to(logp.device), sample["ref"].to(logp.device),
                                             advantage.to(logp.device), recipe["clip"], recipe["beta"])
            (loss/count).backward()
            logs.append({"loss": float(loss.detach()), **metrics})
            outcomes.append(sample["score"]["answer_reward"])
            valid.append(sample["score"]["valid"])
            step_means.append(sum(sample["score"]["step_scores"])/len(sample["score"]["step_scores"]))
            lengths.append(len(sample["ids"]))
            max_difference = max(max_difference, sample["logprob_max_difference"])
    norm = _finish_update(model, optimizer)
    return {key: sum(row[key] for row in logs)/len(logs) for key in logs[0]} | {
        "gradient_norm": norm, "answer_reward": sum(outcomes)/len(outcomes),
        "valid_rate": sum(valid)/len(valid), "mean_step_reward": sum(step_means)/len(step_means),
        "mean_generated_tokens": sum(lengths)/len(lengths), "sampled_logprob_max_difference": max_difference,
        "zero_outcome_variance_rate": zero_outcome/len(groups), "zero_process_variance_rate": zero_process/len(groups),
        "groups": len(groups), "completions": count}


def implementation_identity():
    from ..b3 import implementation_id
    root = Path(__file__).parent
    return digest({"core": implementation_id(), "reasoning": {p.name: sha256(p) for p in sorted(root.glob("*.py"))}})


def prepare_model(dataset, baseline_path, preparation_path, device, rank=16, regional_config=None, objective=None):
    from ..io import read_json
    lock, preparation = read_json(baseline_path), read_json(preparation_path)
    if lock["schema"] != "b4dl-core-baseline-v1" or preparation["baseline_checkpoint"] != lock["components"]["stage2"]["path"]:
        raise ValueError("preparation/baseline mismatch")
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required by this recipe")
    configure_precision(device)
    verify_sst_binding(lock, dataset.geometry)
    tokenizer, original = load_original(lock, device)
    if device.startswith("cuda"):
        original.to(dtype=torch.bfloat16)
    if regional_config is None:
        model, options = attach(original, "A3", chosen_config(preparation, "A3"), rank)
    else:
        from ..integration import attach_regional
        model, options = attach_regional(original, regional_config, rank)
    if objective is not None:
        from .supervision import ObservationHead, validate_objective
        objective = validate_objective(objective)
        if objective["numeric_weight"]:
            if regional_config is None:
                raise ValueError("numeric observation supervision requires regional features")
            core = language_model(model).get_model()
            core.numeric_head = ObservationHead(regional_config.width,
                language_model(model).get_input_embeddings().weight.shape[-1], regional_config.heads).to(
                    device=core.mm_projector.weight.device, dtype=torch.float32)
        options["supervision"] = objective
    use_fp32_output_head(model)
    options["numeric_policy"] = "bf16-strict-reduction-fp32-head-v1"
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("reasoning training requires dropout-free sampling and scoring")
    if float(getattr(language_model(model).config, "attention_dropout", 0.)):
        raise ValueError("reasoning training requires zero attention dropout")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tokenizer, model, options


def preflight_context(model, tokenizer, dataset, recipe):
    from ..integration import prompt_tokens
    config = language_model(model).config
    limits = [getattr(config, key, None) for key in ("max_position_embeddings", "tokenizer_model_max_length")]
    limits = [int(v) for v in limits if v is not None]
    if not limits:
        raise ValueError("model context limit is not declared")
    maximum, visual_counts = 0, {}
    adapter = getattr(language_model(model).get_model(), "regional_adapter", None)
    for request in dataset.requests:
        target = dataset.targets[request["request_id"]]
        answer = teacher_completions(tokenizer, [target], recipe["max_new_tokens"])[0]
        visual = request["feature_length"]
        if adapter is not None and adapter.config.tokens_per_frame == 5:
            scene_id = request["scene_id"]
            if scene_id not in visual_counts:
                from ..regional_model import visual_token_mask
                scene = dataset.geometry.scene(scene_id)
                valid = torch.ones(1, len(scene["features"]), dtype=torch.bool)
                visual_counts[scene_id] = int(visual_token_mask(valid,
                    torch.as_tensor(scene["cell_indices"])[None], torch.as_tensor(scene["cell_valid_mask"])[None], 5).sum())
            visual = visual_counts[scene_id]
        prompt = len(prompt_tokens(tokenizer, request["prompt"])) - 1 + visual
        maximum = max(maximum, prompt + max(len(answer), recipe["max_new_tokens"]))
        if maximum > min(limits):
            raise ValueError("reasoning prompt, visual tokens and generation budget exceed context")
    return {"max_expanded_with_generation": maximum, "context_limit": min(limits)}


def preflight_replay_context(model, tokenizer, dataset, mixed_plan, recipe):
    from ..integration import prompt_tokens
    from ..regional_model import visual_token_mask
    config = language_model(model).config
    limits = [int(value) for value in (getattr(config, "max_position_embeddings", None),
              getattr(config, "tokenizer_model_max_length", None)) if value is not None]
    if not limits:
        raise ValueError("model context limit is not declared")
    adapter = getattr(language_model(model).get_model(), "regional_adapter", None)
    used = {ident for step in mixed_plan.plan for ident in step["replay"]}
    visual_counts, maximum = {}, 0
    for ident in sorted(used, key=lambda key: mixed_plan.replay[key]["scene_id"]):
        row = mixed_plan.replay[ident]
        scene = row["scene_id"]
        if scene not in visual_counts:
            visual = dataset.geometry.manifest["scenes"][scene]["frames"]
            if adapter is not None and adapter.config.tokens_per_frame == 5:
                bundle = dataset.geometry.scene(scene)
                valid = torch.ones(1, visual, dtype=torch.bool)
                visual = int(visual_token_mask(valid, torch.as_tensor(bundle["cell_indices"])[None],
                             torch.as_tensor(bundle["cell_valid_mask"])[None], 5).sum())
            visual_counts[scene] = visual
        completion = tokenizer.encode(row["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(completion) > recipe["max_new_tokens"]:
            raise ValueError("replay answer exceeds generation budget: " + ident)
        length = len(prompt_tokens(tokenizer, row["prompt"])) - 1 + visual_counts[scene] + len(completion)
        maximum = max(maximum, length)
        if maximum > min(limits):
            raise ValueError("replay prompt and visual tokens exceed context: " + ident)
    return {"replay_unique_rows": len(used), "max_replay_expanded": maximum, "context_limit": min(limits)}


def train(dataset, baseline_path, preparation_path, output, stage, recipe, *, device="cuda:0",
          sft_checkpoint=None, resume=None, regional_config=None, mixed_plan=None, objective=None):
    """Run one frozen-recipe pass, or an explicitly bounded engineering run."""
    if stage not in ("sft", "grpo") or {r["split"] for r in dataset.requests} != {"train"}:
        raise ValueError("training requires a nonempty train split")
    if stage == "grpo" and not sft_checkpoint:
        raise ValueError("GRPO requires a completed CoT SFT checkpoint")
    if stage == "sft" and sft_checkpoint:
        raise ValueError("SFT starts from the verified B3 baseline")
    regional_manifest = getattr(dataset, "regional_manifest", None)
    if (regional_config is None) != (regional_manifest is None):
        raise ValueError("regional configuration and regional feature dataset must be supplied together")
    if regional_config is not None and stage != "sft":
        raise ValueError("regional training currently supports SFT")
    if (mixed_plan is None) != (objective is None) or mixed_plan is not None and stage != "sft":
        raise ValueError("mixed plan and objective require SFT and must be supplied together")
    if mixed_plan is not None:
        from .supervision import validate_objective
        objective = validate_objective(objective)
        if recipe["grad_accum"] != 4 or mixed_plan.manifest["seed"] != recipe["seed"]:
            raise ValueError("mixed training requires effective batch four and matched seed")
        if not recipe.get("engineering") and mixed_plan.manifest["updates"] != math.ceil(len(dataset.requests) / 4):
            raise ValueError("formal mixed plan must match the geometry update budget")
    if regional_manifest is not None and not recipe.get("engineering") and not regional_manifest["complete"]:
        raise ValueError("formal regional training requires complete regional features")
    if Path(output).exists():
        raise FileExistsError("choose a new training output directory")
    if not recipe.get("engineering") and (dataset.geometry.manifest.get("usage") == "mini_engineering_validation"
            or not dataset.geometry.manifest.get("encoder_checkpoint_verified")):
        raise ValueError("formal training requires verified full-scale features")
    random.seed(recipe["seed"]); np.random.seed(recipe["seed"]); torch.manual_seed(recipe["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(recipe["seed"])
    reset_gpu_memory(device)
    tokenizer, model, options = prepare_model(dataset, baseline_path, preparation_path, device,
                                               recipe["lora_rank"], regional_config, objective)
    context = preflight_context(model, tokenizer, dataset, recipe)
    if mixed_plan is not None:
        context["replay"] = preflight_replay_context(model, tokenizer, dataset, mixed_plan, recipe)
    common = {"baseline_sha256": sha256(baseline_path), "preparation_sha256": sha256(preparation_path),
              "reasoning_dataset_id": dataset.manifest["dataset_id"], "implementation": implementation_identity(),
              "reward_version": REWARD_VERSION, "options": options}
    if regional_manifest is not None:
        common["regional_dataset_id"] = regional_manifest["dataset_id"]
    if mixed_plan is not None:
        common["mixed_plan_id"] = mixed_plan.manifest["plan_id"]
    if stage == "grpo":
        sft = load_checkpoint(sft_checkpoint, model, expected_stage="sft")
        receipt_path = Path(sft_checkpoint).parent/"completion.json"
        from ..io import read_json
        receipt = read_json(receipt_path)
        completed = (receipt.get("checkpoint_sha256") == sha256(sft_checkpoint)
                     and receipt.get("identity") == sft["identity"] and receipt.get("updates") == sft["step"])
        formal = (not sft["recipe"]["engineering"] and sft["identity"]["complete_budget"] and receipt.get("status") == "training_complete"
                  and sft["step"] == math.ceil(len(dataset.requests)/sft["recipe"]["grad_accum"]))
        engineering = recipe.get("engineering") and receipt.get("status") == "engineering_budget_complete"
        if sft["identity"]["common"] != common or not completed or not (formal or engineering):
            raise ValueError("GRPO requires the matched completed SFT endpoint")
        add_reference_adapter(model)
    optimizer = _optimizer(model, recipe["sft_lr"] if stage == "sft" else recipe["grpo_lr"])
    total = mixed_plan.manifest["updates"] if mixed_plan is not None else math.ceil(len(dataset.requests)/recipe["grad_accum"])
    max_updates = recipe.get("max_updates")
    stop = min(total, max_updates) if max_updates else total
    execution = {"runtime": runtime_identity(language_model(model)), "device": device,
                 "dtype": str(language_model(model).get_input_embeddings().weight.dtype),
                 "numerics": numerics_identity(model),
                 "gpu_name": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None}
    identity = {"common": common, "stage": stage, "recipe": recipe, "complete_budget": stop == total,
                "execution": execution,
                "sft_checkpoint_sha256": sha256(sft_checkpoint) if sft_checkpoint else None}
    start = 0
    if resume:
        payload = load_checkpoint(resume, model, identity, stage, optimizer)
        start = payload["step"]
        if start >= stop:
            raise ValueError("resume checkpoint already reached requested budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"run.json", {"identity": identity, "context": context, "total_updates": total,
        "requested_updates": stop, **execution,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    order = np.random.default_rng(recipe["seed"]).permutation(len(dataset.requests)).tolist()
    step = start
    try:
        for step in range(start, stop):
            started = time.perf_counter()
            if mixed_plan is not None:
                planned = mixed_plan.plan[step]
                requests = [mixed_plan.requests[ident] for ident in planned["geometry"]]
                batch = [(dataset.model_input(r), r, dataset.targets[r["request_id"]]) for r in requests]
                replay = [mixed_plan.replay_input(dataset, ident) for ident in planned["replay"]]
                result = mixed_sft_update(model, tokenizer, batch, replay, optimizer, recipe, objective, planned["pair_type"])
                result["replay_ids"] = planned["replay"]
            elif stage == "sft":
                requests = [dataset.requests[i] for i in order[step*recipe["grad_accum"]:(step+1)*recipe["grad_accum"]]]
                batch = [(dataset.model_input(r), dataset.targets[r["request_id"]]) for r in requests]
                result = sft_update(model, tokenizer, batch, optimizer, recipe["max_new_tokens"])
            else:
                requests = [dataset.requests[i] for i in order[step*recipe["grad_accum"]:(step+1)*recipe["grad_accum"]]]
                batch = [(dataset.model_input(r), r, dataset.targets[r["request_id"]]) for r in requests]
                result = grpo_update(model, tokenizer, batch, optimizer, recipe)
            result.update(step=step+1, requests=[r["request_id"] for r in requests], seconds=time.perf_counter()-started)
            result["gpu_memory"] = gpu_memory(device)
            append_jsonl(output/"training.jsonl", [result])
            if step == start or (step+1) % recipe["save_every"] == 0 or step+1 == stop:
                path = output/("step-"+str(step+1)+".pt")
                save_checkpoint(path, model, identity, stage, recipe, step+1, optimizer)
                write_json(output/"latest.json", {"step": step+1, "checkpoint": path.name, "sha256": sha256(path)})
    except Exception as error:
        write_json(output/"failure.json", {"status": "training_failed", "attempted_step": step+1,
            "error_type": type(error).__name__, "message": str(error), "identity": identity,
            "optimizer_update_applied": "unknown_if_failure_during_update"})
        raise
    completion = {"status": "engineering_budget_complete" if recipe["engineering"] else "training_complete",
                  "updates": stop, "full_budget": stop == total, "quality_evaluation_completed": False,
                  "gpu_memory": gpu_memory(device),
                  "checkpoint_sha256": sha256(output/("step-"+str(stop)+".pt")), "identity": identity}
    write_json(output/"completion.json", completion)
    return completion
