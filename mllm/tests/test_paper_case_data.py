"""Dependency-free checks for evaluation/image identity and evidence selection."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


VTIMELLM_ROOT = Path(__file__).resolve().parents[1] / "vtimellm"
if str(VTIMELLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VTIMELLM_ROOT))

from model_effects import EvaluationSample  # noqa: E402
from paper_case_data import resolve_paper_sample, select_evidence_frames  # noqa: E402


class _SceneRepository:
    def __init__(self, frame_count=40):
        self.scene = SimpleNamespace(
            scene_token="token-a", sample_tokens=tuple(range(frame_count))
        )

    def get_scene_by_id(self, scene_id):
        if scene_id != "scene-a":
            raise KeyError(scene_id)
        return self.scene


def _sample(**changes):
    sample = EvaluationSample(
        sample_id="time_grounding:000007",
        task="time_grounding",
        source_index=7,
        question="When does the car turn?",
        ground_truth="from frame 10 to frame 14",
        prediction="from frame 11 to frame 14",
        scene_id="scene-a",
        scene_token="token-a",
        feat_indices=(10, 12, 14, 16, 18, 20, 22, 24, 26),
    )
    return replace(sample, **changes)


class EvidenceFrameTests(unittest.TestCase):
    def test_representatives_stay_inside_saved_model_input(self):
        self.assertEqual(
            select_evidence_frames(40, (10, 12, 14, 16, 18, 20, 22, 24, 26)),
            (10, 14, 18, 22, 26),
        )
        self.assertEqual(select_evidence_frames(40, (24, 10)), (10, 24))

    def test_all_indices_are_validated_before_subsampling(self):
        for indices in ((10, 12, 14, 16, 40, 20, 22, 24, 26), (-1, 10, 20)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "超出"):
                select_evidence_frames(40, indices)
        for indices in ((1.2, 3), (True, 3), ("1", 3)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "整数"):
                select_evidence_frames(40, indices)

    def test_empty_or_single_frame_cannot_expand_to_scene_montage(self):
        for indices in ((), (10,), (10, 10)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "不同输入帧"):
                select_evidence_frames(40, indices)
        self.assertEqual(select_evidence_frames(40, (10, 10, 20, 20)), (10, 20))

    def test_count_and_scene_length_enforce_montage_bounds(self):
        self.assertEqual(select_evidence_frames(9), (0, 2, 4, 6, 8))
        self.assertEqual(select_evidence_frames(2), (0, 1))
        self.assertEqual(len(select_evidence_frames(40, count=8)), 8)
        for count in (1, 9, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_evidence_frames(40, count=count)
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            select_evidence_frames(1)


class PaperSampleResolutionTests(unittest.TestCase):
    def test_resolves_scene_and_records_sample_source_and_input_role(self):
        resolved = resolve_paper_sample(
            _SceneRepository(), _sample(), evaluation_source="b3-eval-20261001"
        )
        self.assertEqual(resolved.scene_token, "token-a")
        self.assertEqual(resolved.frame_indices, (10, 14, 18, 22, 26))
        for value in (
            "time_grounding:000007", "time_grounding", "b3-eval-20261001",
            "saved feat_indices", "Camera/GT auxiliary", "model input: LiDAR",
        ):
            self.assertIn(value, resolved.source_note)

    def test_missing_unmapped_and_conflicting_scene_are_rejected(self):
        for changes, message in (
            ({"scene_id": None}, "缺少 scene_id"),
            ({"scene_id": "other-scene"}, "未映射"),
            ({"scene_token": "other-token"}, "映射冲突"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                resolve_paper_sample(_SceneRepository(), _sample(**changes))

    def test_absent_frame_metadata_uses_explicitly_labeled_scene_evidence(self):
        resolved = resolve_paper_sample(_SceneRepository(), _sample(feat_indices=None))
        self.assertEqual(resolved.frame_indices, (0, 10, 20, 29, 39))
        self.assertIn("Scene evidence; input frames not saved", resolved.source_note)
        self.assertNotIn("saved feat_indices", resolved.source_note)

    def test_inclusive_range_evidence_is_kept_distinct_from_exact_input(self):
        resolved = resolve_paper_sample(
            _SceneRepository(), _sample(feat_indices=None, feat_range=(10, 18))
        )
        self.assertEqual(resolved.frame_indices, (10, 12, 14, 16, 18))
        self.assertIn("Recorded range; exact input frames not saved", resolved.source_note)
        self.assertNotIn("saved feat_indices", resolved.source_note)

    def test_indices_take_priority_and_bad_saved_indices_do_not_use_range(self):
        resolved = resolve_paper_sample(
            _SceneRepository(), _sample(feat_range=(0, 39))
        )
        self.assertEqual(resolved.frame_indices, (10, 14, 18, 22, 26))
        with self.assertRaisesRegex(ValueError, "超出"):
            resolve_paper_sample(
                _SceneRepository(), _sample(feat_indices=(10, 40), feat_range=(0, 39))
            )

    def test_invalid_or_single_frame_ranges_are_rejected(self):
        for saved_range in ((10,), (10, 12, 14), (-1, 10), (10, 40), (20, 10), (10, 10)):
            with self.subTest(saved_range=saved_range), self.assertRaises(ValueError):
                resolve_paper_sample(
                    _SceneRepository(), _sample(feat_indices=None, feat_range=saved_range)
                )

    def test_footer_is_one_line_when_metadata_contains_whitespace(self):
        resolved = resolve_paper_sample(
            _SceneRepository(), _sample(sample_id="id\n7"), evaluation_source="eval\nrun"
        )
        self.assertNotIn("\n", resolved.source_note)
        self.assertIn("id 7", resolved.source_note)
        self.assertIn("eval run", resolved.source_note)


if __name__ == "__main__":
    unittest.main()
