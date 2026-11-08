"""Check existing regional assets and follow-up configurations on CPU only."""
import argparse
import json
from pathlib import Path

from b4dl_core.io import read_json, sha256
from b4dl_core.reasoning.data import ReasoningDataset
from b4dl_core.reasoning.evaluation import validate_ablation_lineage
from b4dl_core.reasoning.mixed import MixedPlan
from b4dl_core.reasoning.supervision import validate_objective
from b4dl_core.reasoning.trainer import implementation_identity
from b4dl_core.regional_data import RegionalDataset
from b4dl_core.regional_model import RegionalConfig


def config(path):
    value = read_json(path)
    if value.pop("schema", None) != "b4dl-regional-config-v1":
        raise ValueError("invalid regional configuration schema")
    return RegionalConfig(**value).as_dict()


def verify_configs(s2, s3, s4):
    if {k: v for k, v in s3.items() if k != "use_geometry"} != {
        k: v for k, v in s2.items() if k != "use_geometry"
    } or s2["use_geometry"] is not False or s3["use_geometry"] is not True:
        raise ValueError("S3 must differ from S2 only by geometry")
    if {k: v for k, v in s4.items() if k != "tokens_per_frame"} != {
        k: v for k, v in s3.items() if k != "tokens_per_frame"
    } or s3["tokens_per_frame"] != 1 or s4["tokens_per_frame"] != 5:
        raise ValueError("S4 must differ from S3 only by token count")
    if any(row["centroid_policy"] != "observed_assignment" for row in (s2, s3, s4)):
        raise ValueError("v2 assets require observed_assignment")


def endpoint(root):
    root = Path(root)
    run, completion, latest = (read_json(root / name) for name in ("run.json", "completion.json", "latest.json"))
    if (completion.get("status") != "training_complete" or completion.get("full_budget") is not True
            or completion.get("updates") != 13851 or run.get("identity") != completion.get("identity")
            or latest.get("step") != 13851 or latest.get("sha256") != completion.get("checkpoint_sha256")
            or sha256(root / latest["checkpoint"]) != latest["sha256"]):
        raise ValueError("training endpoint identity or hash mismatch: " + str(root))
    identity = run["identity"]
    return {"lineage": {"common": identity["common"]},
            "training": {"stage": identity["stage"], "recipe": identity["recipe"],
                         "step": completion["updates"], "execution": identity["execution"],
                         "planned_full_budget": identity["complete_budget"]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "geometry", "regional", "mixed-plan", "s1", "s2", "t0", "configs", "source-configs"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    if __import__("os").environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run this preflight with CUDA_VISIBLE_DEVICES empty")
    source_id = implementation_identity()
    s1, s2, t0 = (endpoint(path) for path in (args.s1, args.s2, args.t0))
    s2common, t0common = s2["lineage"]["common"], t0["lineage"]["common"]
    if s2common["implementation"] != source_id:
        raise ValueError("frozen source does not match S2 implementation")
    s2config = config(args.source_configs / "core_regional_s2_v2.json")
    s3config = config(args.configs / "core_regional_s3_v2.json")
    s4config = config(args.configs / "core_regional_s4_v2.json")
    if s2config != s2common["options"]["regional"]:
        raise ValueError("S2 configuration does not match training identity")
    verify_configs(s2config, s3config, s4config)
    validate_ablation_lineage(s1, s2)
    data = RegionalDataset(ReasoningDataset(args.dataset, args.geometry, "train"), args.regional)
    manifest = data.regional_manifest
    if (manifest.get("schema") != "b4dl-regional-features-v2" or not manifest.get("complete")
            or manifest["dataset_id"] != s2common["regional_dataset_id"]
            or data.manifest["dataset_id"] != s2common["reasoning_dataset_id"]):
        raise ValueError("regional or reasoning dataset does not match S2")
    mixed = MixedPlan(args.mixed_plan, data)
    if mixed.manifest["plan_id"] != t0common["mixed_plan_id"] or mixed.manifest["updates"] != 13851:
        raise ValueError("mixed plan does not match T0")
    objectives = {}
    for name in ("core_mixed_t0_t1_v1.json", "core_mixed_t2_v1.json", "core_mixed_t3_v1.json",
                 "core_mixed_t3_no_replay_v1.json"):
        objectives[name] = validate_objective(read_json(args.source_configs / name))
    same_source = t0common["implementation"] == source_id
    print(json.dumps({"status": "cpu_preflight_passed", "source_implementation": source_id,
        "t0_implementation": t0common["implementation"], "t0_same_implementation": same_source,
        "t0_strict_cross_source_comparison_ready": same_source,
        "dataset_id": data.manifest["dataset_id"], "regional_dataset_id": manifest["dataset_id"],
        "regional_scenes": len(manifest["scenes"]), "train_requests": len(data.requests),
        "mixed_plan_id": mixed.manifest["plan_id"], "mixed_updates": len(mixed.plan),
        "s1_s2_ablation_identity": "matched", "s3_config": s3config, "s4_config": s4config,
        "objectives": objectives, "note": "CPU identity and configuration check only; no model loaded or GPU training"},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
