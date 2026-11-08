"""Exercise real geometry and six-task generation before full evaluation."""
import argparse
from pathlib import Path

import torch

from b4dl_core.b3 import device_lease
from b4dl_core.io import sha256, write_json
from b4dl_core.reasoning.data import ReasoningDataset
from b4dl_core.reasoning.evaluation import load_free_six_task, load_model, load_original_model
from b4dl_core.reasoning.trainer import sample_one
from b4dl_core.six_task import query_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "geometry", "baseline", "preparation", "protocol", "gpu-lock", "output"):
        parser.add_argument("--" + name, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--original-b3", action="store_true")
    model.add_argument("--checkpoint")
    parser.add_argument("--regional-features")
    parser.add_argument("--mixed-plan")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("choose a new smoke output")
    dataset = ReasoningDataset(args.dataset, args.geometry, "internal_dev")
    if args.regional_features:
        from b4dl_core.regional_data import RegionalDataset
        dataset = RegionalDataset(dataset, args.regional_features)
    plan = None
    if args.mixed_plan:
        from b4dl_core.reasoning.mixed import MixedPlan
        plan = MixedPlan(args.mixed_plan, dataset)
    protocol, six = load_free_six_task(args.protocol, dataset.geometry)
    with device_lease(args.device, args.gpu_lock):
        if args.original_b3:
            tokenizer, model = load_original_model(dataset, args.baseline, args.preparation, args.device)
            artifact = args.baseline
        else:
            tokenizer, model, _ = load_model(dataset, args.baseline, args.preparation, args.checkpoint, args.device, plan)
            artifact = args.checkpoint
        geometry, six_tasks = {}, {}
        for request in dataset.requests:
            if request["task"] in geometry:
                continue
            result = sample_one(model, tokenizer, dataset.model_input(request), 32, do_sample=False)
            if result["status"] not in ("ok", "truncated") or not result["ids"]:
                raise RuntimeError("geometry smoke generation failed for " + request["task"])
            geometry[request["task"]] = {"request_id": request["request_id"], "status": result["status"],
                                         "generated_tokens": len(result["ids"]), "seconds": result["generation_seconds"]}
            if len(geometry) == 4:
                break
        if set(geometry) != {"existence", "position", "distance", "motion"}:
            raise ValueError("geometry smoke does not cover the four tasks")
        for row in six:
            if row["task"] in six_tasks:
                continue
            with torch.no_grad():
                generated = model.generate(**query_batch(model, tokenizer, dataset.geometry, row),
                    max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            if generated.shape[0] != 1 or generated.shape[1] < 1:
                raise ValueError("empty six-task smoke generation")
            six_tasks[row["task"]] = {"request_id": row["request_id"], "generated_tokens": int(generated.shape[1])}
            if len(six_tasks) == 6:
                break
        if len(six_tasks) != 6:
            raise ValueError("six-task smoke does not cover all tasks")
        write_json(output, {"status": "evaluation_smoke_passed", "artifact_sha256": sha256(artifact),
            "reasoning_dataset_id": dataset.manifest["dataset_id"], "protocol_id": protocol["protocol_id"],
            "regional_dataset_id": getattr(dataset, "regional_manifest", {}).get("dataset_id"),
            "geometry": geometry, "six_tasks": six_tasks,
            "note": "bounded engineering generation; not model performance"})
    print("evaluation smoke passed", flush=True)


if __name__ == "__main__":
    main()
