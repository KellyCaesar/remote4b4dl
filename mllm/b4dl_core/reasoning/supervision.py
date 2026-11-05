"""Evidence supervision from public questions and private numeric targets."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..inputs import collate_frames
from ..integration import language_model
from .policy import decode_completion, expanded_log_probs, sequence_batch, step_token_ends, teacher_completions
from .protocol import encode_trace, parse_trace


FIELDS = ("count", "x", "y", "d0", "d1", "y0", "y1", "distance")


def validate_objective(config):
    expected = {"schema", "segmented", "numeric_weight", "pair_weight", "geometry_weight",
                "replay_weight", "trace_weight", "short_weight", "pair_margin"}
    if set(config) != expected or config.get("schema") != "b4dl-mixed-objective-v1" or type(config["segmented"]) is not bool:
        raise ValueError("invalid mixed objective schema")
    for key in expected - {"schema", "segmented"}:
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError("invalid mixed objective coefficient: " + key)
    if config["geometry_weight"] == 0 or config["trace_weight"] + config["short_weight"] == 0:
        raise ValueError("geometry and text supervision require positive weight")
    return dict(config)


class ObservationHead(nn.Module):
    """Read all requested observations before any teacher answer is supplied."""
    def __init__(self, visual_width, language_width, heads):
        super().__init__()
        self.query = nn.Linear(language_width, visual_width)
        self.field = nn.Embedding(len(FIELDS), visual_width)
        self.attention = nn.MultiheadAttention(visual_width, heads, dropout=0, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(visual_width), nn.Linear(visual_width, visual_width),
                                    nn.GELU(), nn.Linear(visual_width, 1))

    def forward(self, queries, fields, memory, memory_valid):
        vectors = self.query(queries.float()) + self.field(fields)
        attended, _ = self.attention(vectors.unsqueeze(0), memory.unsqueeze(0), memory.unsqueeze(0),
                                     key_padding_mask=~memory_valid.unsqueeze(0), need_weights=False)
        values = self.output(vectors + attended[0]).squeeze(-1)
        nonnegative = torch.tensor([FIELDS[index] in ("count", "d0", "d1", "distance")
                                    for index in fields.tolist()], device=values.device)
        return torch.where(nonnegative, F.softplus(values), values)


def observation_fields(request):
    task = request["task"]
    if task == "existence":
        names = [("count", None, "count")]
    elif task == "position":
        names = [(key, None, key) for key in ("x", "y")]
    elif task == "motion":
        names = [(key, None, key) for key in ("d0", "d1", "y0", "y1")]
    elif task == "distance":
        names = [("distance", category, "distance_" + category)
                 for category in request["options"].values()]
    else:
        raise ValueError("unsupported observation task")
    if len({key for _, _, key in names}) != len(names):
        raise ValueError("observation fields must be unique")
    return names


def observation_spec(request, target):
    names = observation_fields(request)
    observed = target["trace"]["steps"][0]["value"]
    if set(observed) != {key for _, _, key in names}:
        raise ValueError("observation fields do not match the task")
    result = []
    for kind, category, key in names:
        value = observed[key]
        if type(value) not in (int, float) or not math.isfinite(value) or kind == "count" and value < 0:
            raise ValueError("invalid observation truth")
        result.append((kind, category, float(value)))
    return result


def _observation_forward(model, tokenizer, item, request):
    """Compute observations using only the public question and sensor input."""
    base = language_model(model)
    core = base.get_model()
    adapter, head = getattr(core, "regional_adapter", None), getattr(core, "numeric_head", None)
    if adapter is None or head is None:
        raise ValueError("observation supervision requires regional adapter and numeric head")
    if not isinstance(request.get("question"), str) or not request["question"].strip():
        raise ValueError("public question text required")
    device = base.get_input_embeddings().weight.device
    tensors = {key: value.to(device) for key, value in collate_frames([item]).items()}
    frames = request["frames"]
    if (not frames or any(type(frame) is not int or not 0 <= frame < tensors["features"].shape[1] for frame in frames)
            or frames != sorted(set(frames))):
        raise ValueError("public question frame indices are invalid")
    fields = observation_fields(request)
    embedding = base.get_input_embeddings()
    queries = []
    with torch.no_grad():
        for _, category, _ in fields:
            text = request["question"] + ("\nCandidate class: " + category.replace("_", " ") if category else "")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError("public observation query tokenized to empty")
            queries.append(embedding(torch.tensor(ids, device=device)).float().mean(dim=0))
    with torch.autocast(device_type=device.type, enabled=False):
        global_tokens, regions, occupied = adapter.encode(tensors["features"].float(), tensors["frame_times"],
            tensors["frame_geometry"], tensors["frame_valid_mask"], tensors["cell_features"].float(),
            tensors["cell_xyz"].float(), tensors["cell_indices"], tensors["cell_valid_mask"], tensors["ego_from_sst"])
        global_selected = global_tokens[0, frames].unsqueeze(1)
        regions_selected = regions[0, frames]
        memory = torch.cat((global_selected, regions_selected), dim=1).flatten(0, 1)
        valid = torch.cat((torch.ones(len(frames), 1, dtype=torch.bool, device=device), occupied[0, frames]), dim=1).flatten()
        field_ids = torch.tensor([FIELDS.index(kind) for kind, _, _ in fields], dtype=torch.long, device=device)
        predicted = head(torch.stack(queries), field_ids, memory, valid)
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("nonfinite observation predictions")
    return fields, field_ids, predicted


def observation_values(fields, predicted):
    result = {}
    for (kind, _, key), value in zip(fields, predicted):
        number = float(value.detach())
        try:
            number = math.expm1(number) if kind == "count" else number * 10.
        except OverflowError:
            raise FloatingPointError("observation prediction exceeds numeric range") from None
        if not math.isfinite(number):
            raise FloatingPointError("nonfinite observation prediction")
        result[key] = number
    return result


def observation_loss(model, tokenizer, item, request, target):
    """Query with question/option text only; targets enter the final loss."""
    fields, field_ids, predicted = _observation_forward(model, tokenizer, item, request)
    spec = observation_spec(request, target)
    device = predicted.device
    with torch.autocast(device_type=device.type, enabled=False):
        truth = torch.tensor([math.log1p(value) if kind == "count" else value / 10.
                              for kind, _, value in spec], dtype=torch.float32, device=device)
        continuous = field_ids.ne(FIELDS.index("count"))
        parts = torch.empty_like(predicted)
        if continuous.any():
            parts[continuous] = F.smooth_l1_loss(predicted[continuous], truth[continuous], beta=.1, reduction="none")
        if (~continuous).any():
            parts[~continuous] = F.smooth_l1_loss(predicted[~continuous], truth[~continuous], beta=1., reduction="none")
    return parts.mean(), observation_values(fields, predicted)


def observation_rule_answer(request, values):
    from ..geometry import motion_truth, sector
    expected = {key for _, _, key in observation_fields(request)}
    if (set(values) != expected or any(type(value) not in (int, float) or not math.isfinite(value)
                                       for value in values.values())):
        raise ValueError("invalid observation prediction")
    if any(values[key] < 0 for key in expected if key == "count" or key in ("d0", "d1") or key.startswith("distance_")):
        raise ValueError("count and distance predictions must be nonnegative")
    if request["task"] == "existence":
        return "Yes" if values["count"] >= .5 else "No"
    if request["task"] == "position":
        try:
            result = sector(values["x"], values["y"]) == request["predicate"]
        except ValueError:
            return None
        return "Yes" if result else "No"
    if request["task"] == "motion":
        motion = {"delta_d": values["d1"] - values["d0"], "delta_y": values["y1"] - values["y0"]}
        return "Yes" if motion_truth(motion)[request["predicate"]] else "No"
    return min(sorted(request["options"]), key=lambda letter: values["distance_" + request["options"][letter]])


def predict_observations(model, tokenizer, item, request):
    fields, _, predicted = _observation_forward(model, tokenizer, item, request)
    values = observation_values(fields, predicted)
    return {"values": values, "rule_answer": observation_rule_answer(request, values)}


def segmented_trace_loss(model, tokenizer, item, request, target, max_new_tokens):
    completion = teacher_completions(tokenizer, [target], max_new_tokens)[0]
    text = decode_completion(tokenizer, completion)
    if text != encode_trace(target["trace"]):
        raise ValueError("teacher tokenization cannot reproduce the exact trace")
    parsed = parse_trace(text, request)
    ends = step_token_ends(tokenizer, completion, parsed["step_char_ends"])
    if len(ends) < 2 or not 0 < ends[0] < ends[-1] < len(completion):
        raise ValueError("trace token segments are not separable")
    batch = sequence_batch(model, tokenizer, [item], [completion])
    log_probs = expanded_log_probs(model, batch)[0]
    observe = -log_probs[:ends[0]].mean()
    derive = -log_probs[ends[0]:ends[-1]].mean()
    answer = -log_probs[ends[-1]:].mean()
    loss = (observe + .5 * derive + answer) / 2.5
    return loss, {"observe": float(observe.detach()), "derive": float(derive.detach()),
                  "answer": float(answer.detach())}


def short_question(request):
    question = request["question"]
    if request["kind"] == "binary":
        return question + "\nAnswer only Yes or No."
    options = "\n".join(f"{letter}. {value.replace('_', ' ')}" for letter, value in request["options"].items())
    return question + "\n" + options + "\nAnswer only with the option letter."


def short_answer_scores(model, tokenizer, item, request, max_new_tokens):
    """Score complete candidate answers using the same answer-only prompt."""
    options = ("Yes", "No") if request["kind"] == "binary" else tuple(request["options"])
    if request["kind"] == "mcq" and set(options) != set("ABCD"):
        raise ValueError("four distinct short-answer options required")
    question_item = {**item, "prompt": short_question(request)}
    scores = []
    for answer in options:
        ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        if not ids or len(ids) > max_new_tokens:
            raise ValueError("short answer exceeds generation budget")
        batch = sequence_batch(model, tokenizer, [question_item], [ids])
        scores.append(expanded_log_probs(model, batch)[0].sum())
    return options, torch.stack(scores)


def pair_regularizer(requests, targets, score_rows, pair_type, margin=.5):
    if len(requests) != 2 or len(targets) != 2 or len(score_rows) != 2:
        raise ValueError("two labeled geometry examples are required")
    first, second = requests
    if pair_type == "opposite":
        if (first["kind"] != "binary" or second["kind"] != "binary" or first["prompt"] != second["prompt"]
                or first["feature_length"] != second["feature_length"] or first["scene_id"] == second["scene_id"]
                or {targets[0]["answer"], targets[1]["answer"]} != {"Yes", "No"}):
            raise ValueError("opposite pair identity or labels are invalid")
        losses = []
        for target, (options, scores) in zip(targets, score_rows):
            correct = options.index(target["answer"])
            losses.append(F.relu(scores.new_tensor(margin) - scores[correct] + scores[1 - correct]))
        return torch.stack(losses).mean()
    if pair_type == "rotation":
        if (first["task"] != "distance" or second["task"] != "distance"
                or first["semantic_id"] != second["semantic_id"] or first["scene_id"] != second["scene_id"]
                or first["frames"] != second["frames"] or first["question"] != second["question"]
                or first["option_rotation"] == second["option_rotation"]
                or set(first["options"].values()) != set(second["options"].values())
                or first["options"][targets[0]["answer"]] != second["options"][targets[1]["answer"]]):
            raise ValueError("distance rotation pair identity or labels are invalid")
        classes = sorted(first["options"].values())
        probabilities = []
        for request, (options, scores) in zip(requests, score_rows):
            distribution = torch.softmax(scores.float(), dim=0)
            by_class = {request["options"][letter]: distribution[i] for i, letter in enumerate(options)}
            probabilities.append(torch.stack([by_class[name] for name in classes]))
        left, right = probabilities
        middle = (left + right) / 2
        return .5 * ((left * (left.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum()
                     + (right * (right.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum())
    if pair_type == "unpaired":
        return score_rows[0][1].new_zeros(())
    raise ValueError("unknown pair type")
