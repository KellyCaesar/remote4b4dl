"""Exercise the paper-case UI callbacks without nuScenes data or a model."""

import argparse
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


VTIMELLM_ROOT = Path(__file__).resolve().parents[1] / "vtimellm"
if str(VTIMELLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VTIMELLM_ROOT))

from lidar_visualizer import FrameData, SceneRef  # noqa: E402
from model_effects import EvaluationRepository, EvaluationSample  # noqa: E402


class _Annotations:
    def get(self, table, token):
        if table == "sample":
            return {"anns": [f"annotation:{token}"]}
        if table == "sample_annotation":
            return {
                "instance_token": token.split(":", 1)[1].rsplit("-", 1)[0] + "-object",
                "category_name": "vehicle.car",
            }
        raise KeyError((table, token))


class _Repository:
    def __init__(self):
        self.scenes = tuple(
            SceneRef(
                f"token-{name}", f"scene-{name}", f"scene-{name}", "val",
                tuple(f"sample-{name}-{frame}" for frame in range(40)),
            )
            for name in ("a", "b")
        )
        self.nusc = _Annotations()

    def get_scene(self, scene_token):
        return next((scene for scene in self.scenes if scene.scene_token == scene_token), None) or self._missing(scene_token)

    def get_scene_by_id(self, scene_id):
        return next((scene for scene in self.scenes if scene.scene_id == scene_id), None) or self._missing(scene_id)

    @staticmethod
    def _missing(identity):
        raise KeyError(identity)

    def get_frame(self, scene_token, frame_index):
        scene = self.get_scene(scene_token)
        return FrameData(
            scene, frame_index, scene.sample_tokens[frame_index],
            np.array([[1.0, 2.0, 0.5, 1.0], [2.0, 3.0, 1.0, 1.0]], dtype=np.float32),
            (), (), {},
        )

    @staticmethod
    def camera_image(frame, camera):
        return Image.new("RGB", (32, 18), "#637f92")


def _sample(sample_id, task, source_index, scene_id, feat_indices):
    return EvaluationSample(
        sample_id=sample_id, task=task, source_index=source_index,
        question=f"Question for {sample_id}", ground_truth=f"Ground truth for {sample_id}",
        prediction=f"Prediction for {sample_id}", scene_id=scene_id,
        feat_indices=feat_indices,
    )


class PaperCaseCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gradio  # noqa: F401
            import plotly  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        import demo_gradio

        cls.module = demo_gradio
        cls.repository = _Repository()
        cls.temporal = _sample(
            "temporal_understanding:000007", "temporal_understanding", 7,
            "scene-b", (10, 12, 14, 16, 18, 20, 22, 24, 26),
        )
        cls.other = _sample(
            "temporal_understanding:000008", "temporal_understanding", 8,
            "scene-a", (20, 22, 24, 26, 28),
        )
        cls.unmapped = _sample(
            "temporal_understanding:000009", "temporal_understanding", 9,
            "missing-scene", (0, 2, 4),
        )
        cls.effects = EvaluationRepository([
            _sample("existence:000000", "existence", 0, "scene-a", (0, 2, 4)),
            cls.temporal, cls.other, cls.unmapped,
        ])
        args = argparse.Namespace(
            model_base=None, pretrain_mm_mlp_adapter=None, stage2=None,
            stage3=None, feat_folder=None, gpu_id=0,
        )
        cls.demo = demo_gradio.create_demo(
            cls.repository, demo_gradio.OptionalInferenceEngine(args), cls.effects,
            evaluation_source="b3-saved-eval",
        )
        cls.load_callback = cls._callback("load_paper_sample")
        cls.export_callback = cls._callback("export_paper_case")

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    @classmethod
    def _callback(cls, name):
        return next(
            block_fn for block_fn in cls.demo.fns.values()
            if getattr(block_fn.fn, "__name__", None) == name
        )

    def _loaded(self, sample_id):
        values = self.load_callback.fn(sample_id)
        self.assertEqual(len(values), len(self.load_callback.outputs))
        by_label = {}
        for component, value in zip(self.load_callback.outputs, values):
            if type(component).__name__ == "State":
                by_label["origin"] = value
            elif getattr(component, "label", None):
                by_label[component.label] = value
        return by_label

    def _export_inputs(self, loaded, **changes):
        values = {
            "scene_token": loaded["nuScenes 场景"]["value"],
            "frames": loaded["帧号（2–8 帧，推荐 5 帧）"],
            "title": "TEST CASE", "question": loaded["QUESTION"],
            "baseline_label": loaded["左侧答案来源"],
            "baseline_answer": loaded["参考答案 / GT"], "baseline_highlights": "",
            "b4dl_label": loaded["右侧答案来源"],
            "b4dl_answer": loaded["右侧答案 / B4DL"], "b4dl_highlights": "",
            "boxes": False, "tracks": False, "origin": loaded["origin"],
            "front_instances": [], "back_instances": [],
            "bev_range": 35,
        }
        values.update(changes)
        parameters = list(inspect.signature(self.export_callback.fn).parameters)
        self.assertEqual(len(parameters), len(self.export_callback.inputs))
        return [values[name] for name in parameters]

    def _assert_cleared_artifacts_and_annotations(self, loaded):
        self.assertIsNone(loaded["PAPER FIGURE PREVIEW"])
        self.assertIsNone(loaded["下载论文图"])
        self.assertEqual(loaded["前方关注短语 / 黄色（两组答案共用）"], "")
        self.assertEqual(loaded["后方关注短语 / 绿色（两组答案共用）"], "")
        self.assertEqual(loaded["前方关注目标 / 黄色真值框"]["value"], [])
        self.assertEqual(loaded["后方关注目标 / 绿色真值框"]["value"], [])

    def _artifact(self):
        return SimpleNamespace(
            image=Image.new("RGB", (32, 18)), png_path="case.png", pdf_path="case.pdf",
            metadata_path="case.json", frame_indices=(10, 14, 18, 22, 26),
        )

    def test_initial_paper_case_prefers_temporal_sample_and_saved_input_frames(self):
        self.assertEqual(self.load_callback.inputs[0].value, self.temporal.sample_id)
        component_values = {
            component.label: component.value
            for component in self.load_callback.outputs
            if getattr(component, "label", None)
        }
        self.assertEqual(component_values["nuScenes 场景"], "token-b")
        self.assertEqual(component_values["帧号（2–8 帧，推荐 5 帧）"], "10, 14, 18, 22, 26")
        self.assertEqual(component_values["左侧答案来源"], "Ground Truth")

    def test_switching_sample_clears_old_preview_highlights_and_target_instances(self):
        self._loaded(self.temporal.sample_id)
        loaded = self._loaded(self.other.sample_id)
        self._assert_cleared_artifacts_and_annotations(loaded)
        self.assertEqual(loaded["origin"], {"sample_id": self.other.sample_id})
        self.assertEqual(loaded["右侧答案 / B4DL"], self.other.prediction)
        self.assertEqual(loaded["nuScenes 场景"]["value"], "token-a")

    def test_unmapped_sample_clears_scene_and_blocks_export_even_after_scene_override(self):
        loaded = self._loaded(self.unmapped.sample_id)
        self._assert_cleared_artifacts_and_annotations(loaded)
        self.assertIsNone(loaded["nuScenes 场景"]["value"])
        self.assertEqual(loaded["帧号（2–8 帧，推荐 5 帧）"], "")
        self.assertIn("error", loaded["origin"])
        with patch.object(self.module, "build_paper_case") as build:
            result = self.export_callback.fn(*self._export_inputs(
                loaded, scene_token="token-a", frames="0, 2, 4"
            ))
        self.assertEqual(len(result), len(self.export_callback.outputs))
        self.assertIsNone(result[0])
        build.assert_not_called()
        self.assertIn("未映射", result[2])

    def test_unchanged_saved_answers_retain_evaluation_provenance(self):
        loaded = self._loaded(self.temporal.sample_id)
        with patch.object(self.module, "build_paper_case", return_value=self._artifact()) as build:
            result = self.export_callback.fn(*self._export_inputs(loaded))
        self.assertEqual(len(result), len(self.export_callback.outputs))
        self.assertIsNotNone(result[0])
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["provenance"]["mode"], "evaluation")
        self.assertEqual(kwargs["provenance"]["sample_id"], self.temporal.sample_id)
        self.assertEqual(kwargs["b4dl_answer"], self.temporal.prediction)
        self.assertEqual(kwargs["answer_source"], "saved evaluation (unchanged)")
        self.assertEqual(kwargs["range_m"], 35)

    def test_edited_prediction_is_marked_and_preserves_original_saved_text(self):
        loaded = self._loaded(self.temporal.sample_id)
        with patch.object(self.module, "build_paper_case", return_value=self._artifact()) as build:
            self.export_callback.fn(*self._export_inputs(loaded, b4dl_answer="User edited answer"))
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["provenance"]["mode"], "edited")
        self.assertEqual(kwargs["provenance"]["original_prediction"], self.temporal.prediction)
        self.assertEqual(kwargs["b4dl_answer"], "User edited answer")
        self.assertIn("MANUAL EDITED", kwargs["source_note"])
        for text in (self.temporal.sample_id, "b3-saved-eval", "Camera/GT auxiliary", "LiDAR"):
            self.assertIn(text, kwargs["source_note"])

    def test_clearing_sample_enters_manual_mode_and_discards_saved_answers(self):
        loaded = self._loaded(None)
        self._assert_cleared_artifacts_and_annotations(loaded)
        self.assertIsNone(loaded["origin"])
        self.assertEqual(loaded["QUESTION"], "")
        self.assertEqual(loaded["参考答案 / GT"], "")
        self.assertEqual(loaded["右侧答案 / B4DL"], "")


if __name__ == "__main__":
    unittest.main()
