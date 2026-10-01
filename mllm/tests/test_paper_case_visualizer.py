import tempfile
import unittest
import json
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


MLLM_ROOT = Path(__file__).resolve().parents[1]
VTIMELLM_ROOT = MLLM_ROOT / "vtimellm"
for root in (str(VTIMELLM_ROOT), str(MLLM_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from lidar_visualizer import BoxRender, FrameData, SceneRef, TrackRender
from paper_case_visualizer import (
    build_paper_case,
    compose_paper_case,
    parse_frame_indices,
    render_bev_image,
    select_frame_indices,
    split_highlights,
    render_annotated_camera,
    _fit_cell,
    _font,
    _wrapped_lines,
    _draw_highlighted_answer,
)


class _NoCameraNuScenes:
    @staticmethod
    def get(table, token):
        if table == "sample":
            index = int(token.split("-")[-1])
            return {"data": {"LIDAR_TOP": f"lidar-{index}"}, "timestamp": 1_000_000 + index * 500_000}
        if table == "sample_data":
            return {"calibrated_sensor_token": "calibration"}
        if table == "calibrated_sensor":
            return {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
        if table == "sample_annotation":
            return {"instance_token": "instance-car"}
        raise KeyError((table, token))


class _PaperRepository:
    def __init__(self):
        self.scene = SceneRef(
            "scene-token", "000001", "scene-demo", "val",
            tuple(f"sample-{index}" for index in range(9)),
        )
        self.nusc = _NoCameraNuScenes()

    def get_scene(self, scene_token):
        if scene_token != self.scene.scene_token:
            raise KeyError(scene_token)
        return self.scene

    def get_frame(self, scene_token, frame_index):
        self.get_scene(scene_token)
        corners = np.array(
            [
                [6, 6, 3, 3, 6, 6, 3, 3],
                [2, -2, -2, 2, 2, -2, -2, 2],
                [0, 0, 0, 0, 2, 2, 2, 2],
            ],
            dtype=np.float32,
        )
        x = np.linspace(-20, 20, 240, dtype=np.float32)
        points = np.column_stack((x, np.sin(x) * 8, np.cos(x), np.ones_like(x)))
        return FrameData(
            scene=self.scene,
            frame_index=int(frame_index),
            sample_token=f"sample-{frame_index}",
            points=points,
            boxes=(BoxRender("box", "vehicle.car", corners),),
            tracks=(
                TrackRender(
                    "box", "vehicle.car",
                    np.array([[1, 0, 0], [3, 0.5, 0], [5, 1, 0]], dtype=np.float32),
                ),
            ),
            camera_paths={},
        )

    @staticmethod
    def camera_image(frame, camera):
        color = "#637f92" if camera == "CAM_FRONT" else "#8b7566"
        return Image.new("RGB", (640, 320), color)


class FrameSelectionTests(unittest.TestCase):
    def test_evenly_selects_five_frames_including_scene_ends(self):
        self.assertEqual(select_frame_indices(9), (0, 2, 4, 6, 8))
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            select_frame_indices(1)

    def test_explicit_frames_are_sorted_and_validated(self):
        self.assertEqual(parse_frame_indices("8, 0, 4", 9), (0, 4, 8))
        with self.assertRaisesRegex(ValueError, "不能重复"):
            parse_frame_indices("0, 0, 4", 9)
        with self.assertRaisesRegex(ValueError, "超出"):
            parse_frame_indices("0, 4, 9", 9)

    def test_highlight_lists_accept_chinese_and_ascii_separators(self):
        self.assertEqual(
            split_highlights("vehicles in front，后方车辆; turning left"),
            ("vehicles in front", "后方车辆", "turning left"),
        )


class PaperRenderingTests(unittest.TestCase):
    def test_bev_renderer_draws_points_boxes_and_tracks(self):
        repository = _PaperRepository()
        image = render_bev_image(repository.get_frame("scene-token", 0), size=(500, 260))
        self.assertEqual(image.size, (500, 260))
        self.assertGreater(len(image.getcolors(maxcolors=100_000)), 8)

    def test_camera_letterbox_preserves_both_image_edges(self):
        source = Image.new("RGB", (400, 100), "blue")
        ImageDraw.Draw(source).rectangle((0, 0, 39, 99), fill="red")
        ImageDraw.Draw(source).rectangle((360, 0, 399, 99), fill="green")
        cell = _fit_cell(source, (100, 100))
        self.assertEqual(cell.getpixel((5, 50)), (255, 0, 0))
        self.assertEqual(cell.getpixel((95, 50)), (0, 128, 0))
        self.assertEqual(cell.getpixel((50, 5)), (255, 255, 255))

    def test_bev_calibrates_sensor_points_and_uses_equal_scale(self):
        frame = _PaperRepository().get_frame("scene-token", 0)
        # A +90-degree sensor rotation plus 2m translation: sensor (0,-8)
        # is ego (10,0), and sensor (10,2) is ego (0,10).
        frame = replace(frame, points=np.array([[0, -8, 0, 1], [10, 2, 1, 1]]), boxes=(), tracks=())
        calibration = {"rotation": [2 ** -0.5, 0, 0, 2 ** -0.5], "translation": [2, 0, 0]}
        image = render_bev_image(frame, (400, 200), 20, False, False, calibration)
        background = (247, 247, 249)
        # The 200px square map is centred in the rectangular output.
        # Points sit 50px above and left of ego, respectively.
        for x, y in ((200, 50), (150, 100)):
            neighbourhood = [image.getpixel((px, py)) for px in range(x - 1, x + 2)
                             for py in range(y - 1, y + 2)]
            self.assertTrue(any(pixel != background for pixel in neighbourhood))
        self.assertEqual(image.getpixel((300, 100)), (255, 255, 255))
        self.assertEqual(image.getpixel((200, 150)), background)

    def test_target_instance_camera_boxes_survive_disabled_all_boxes(self):
        repository = _PaperRepository()
        class CameraBox:
            token = "box"
            name = "vehicle.car"
            @staticmethod
            def corners():
                return np.array([[1, 2, 2, 1, 1, 2, 2, 1],
                                 [1, 1, 2, 2, 1, 1, 2, 2],
                                 [10, 10, 10, 10, 11, 11, 11, 11]])
        class CameraNuScenes(_NoCameraNuScenes):
            @staticmethod
            def get(table, token):
                value = _NoCameraNuScenes.get(table, token)
                if table == "sample":
                    value["data"]["CAM_FRONT"] = "camera"
                return value
            @staticmethod
            def get_sample_data(token):
                return "unused", [CameraBox()], np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
        repository.nusc = CameraNuScenes()
        frame = repository.get_frame("scene-token", 0)
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera.png"
            Image.new("RGB", (640, 320), "#637f92").save(camera_path)
            frame = replace(frame, camera_paths={"CAM_FRONT": str(camera_path)})
            image = render_annotated_camera(repository, frame, "CAM_FRONT", False, {"instance-car": "#ffe84a"})
            pixels = np.asarray(image)
            self.assertGreater(np.count_nonzero(np.all(pixels == [255, 232, 74], axis=2)), 10)
            untouched = render_annotated_camera(repository, frame, "CAM_FRONT", False, {"other-instance": "#ffe84a"})
            self.assertEqual(untouched.getpixel((60, 60)), (99, 127, 146))

    def test_missing_camera_never_projects_ground_truth_boxes(self):
        repository = _PaperRepository()
        frame = repository.get_frame("scene-token", 0)
        captured = []
        original = ImageDraw.ImageDraw.text
        def record(instance, xy, value, *args, **kwargs):
            captured.append((str(value), kwargs.get("font").size))
            return original(instance, xy, value, *args, **kwargs)
        with patch.object(repository.nusc, "get", side_effect=AssertionError("Missing camera must skip projections")):
            with patch.object(ImageDraw.ImageDraw, "text", new=record):
                image = render_annotated_camera(repository, frame, "CAM_FRONT", True, {"instance-car": "#ffe84a"})
        self.assertIn(("IMAGE UNAVAILABLE", 30), captured)
        self.assertIn(("CAM_FRONT", 38), captured)
        self.assertEqual(image.size, (640, 360))
        self.assertEqual(np.count_nonzero(np.all(np.asarray(image) == [255, 232, 74], axis=2)), 0)

    def test_target_bev_boxes_survive_disabled_all_boxes(self):
        frame = _PaperRepository().get_frame("scene-token", 0)
        image = render_bev_image(frame, show_boxes=False, show_tracks=False, box_colors={"box": "#70f08b"})
        self.assertGreater(np.count_nonzero(np.all(np.asarray(image) == [112, 240, 139], axis=2)), 10)

    def test_sdk_box_corner_order_produces_complete_bev_footprint(self):
        frame = _PaperRepository().get_frame("scene-token", 0)
        # nuScenes' first four corners describe one vertical side, so using
        # them as a footprint incorrectly collapses a rectangle to a line.
        corners = np.array([[12, 12, 12, 12, 8, 8, 8, 8],
                            [2, -2, -2, 2, 2, -2, -2, 2],
                            [1, 1, -1, -1, 1, 1, -1, -1]])
        frame = replace(frame, points=np.empty((0, 4)),
                        boxes=(BoxRender("box", "vehicle.car", corners),), tracks=())
        image = render_bev_image(frame, (200, 200), 20, False, False, box_colors={"box": "#70f08b"})
        for point in ((100, 40), (100, 60), (90, 50), (110, 50)):
            self.assertEqual(image.getpixel(point), (112, 240, 139))

    def test_long_chinese_and_unbroken_text_wrap_without_truncation(self):
        text = "前方车辆持续向左转弯后方车辆保持行驶" * 60 + "\n" + "continuousword" * 45 + "END"
        canvas = Image.new("RGB", (350, 10000), "white")
        draw = ImageDraw.Draw(canvas)
        lines = _wrapped_lines(draw, text, _font(23), 300)
        self.assertEqual("".join(text[start:end] for start, end in lines), text.replace("\n", ""))
        self.assertTrue(all(draw.textlength(text[start:end], font=_font(23)) <= 300 for start, end in lines))
        captured = []
        original = ImageDraw.ImageDraw.text
        def record(instance, xy, value, *args, **kwargs):
            captured.append(str(value))
            return original(instance, xy, value, *args, **kwargs)
        with patch.object(ImageDraw.ImageDraw, "text", new=record):
            _draw_highlighted_answer(draw, (0, 0, 300, 10000), text, (), "#ffe84a")
        self.assertEqual("".join(captured), text.replace("\n", ""))

    def test_composer_expands_both_question_and_answer_height(self):
        source = Image.new("RGB", (320, 180), "white")
        def compose(question, answer):
            return compose_paper_case([source] * 5, [source] * 5, [source] * 5,
                                      [0, 1, 2, 3, 4], question, answer, answer)
        short = compose("Question?", "Short answer")
        long = compose("车辆在场景中如何移动？" * 100, "后方车辆持续行驶。" * 300)
        self.assertGreater(long.height, short.height + 600)

    def test_fully_missing_camera_rows_are_omitted_and_partial_rows_remain(self):
        source = Image.new("RGB", (320, 180), "white")
        args = ([source] * 5, [source] * 5, [source] * 5, [0, 1, 2, 3, 4], "Q", "GT", "Prediction")
        complete = compose_paper_case(*args)
        lidar_only = compose_paper_case(*args, front_available=[False] * 5, back_available=[False] * 5)
        partial = compose_paper_case(*args, front_available=[True, False, False, False, False], back_available=[False] * 5)
        self.assertGreater(complete.height, partial.height)
        self.assertGreater(partial.height, lidar_only.height)
        self.assertEqual(complete.height - partial.height, partial.height - lidar_only.height)

    def test_front_back_highlights_use_same_colors_in_both_answers(self):
        source = Image.new("RGB", (320, 180), "white")
        image = compose_paper_case([source] * 2, [source] * 2, [source] * 2, [0, 1],
                                   "What moves?", "Front vehicle / back vehicle", "Front vehicle / back vehicle",
                                   front_highlights=("Front vehicle",), back_highlights=("back vehicle",))
        pixels = np.asarray(image)
        for half in (pixels[:, :1000], pixels[:, 1000:]):
            for color in ([255, 232, 74], [112, 240, 139]):
                self.assertGreater(np.count_nonzero(np.all(half == color, axis=2)), 50)

    def test_composer_requires_synchronized_rows(self):
        image = Image.new("RGB", (320, 180), "white")
        with self.assertRaisesRegex(ValueError, "数量一致"):
            compose_paper_case(
                [image, image], [image], [image, image], [0, 1],
                "Question?", "Baseline", "Ours",
            )

    def test_build_exports_png_and_pdf(self):
        repository = _PaperRepository()
        with tempfile.TemporaryDirectory() as directory:
            artifact = build_paper_case(
                repository=repository,
                scene_token="scene-token",
                frame_indices=(0, 2, 4, 6, 8),
                question="What dynamic movement is observed throughout the frames?",
                baseline_answer="Vehicles in front move forward.",
                b4dl_answer="Vehicles in front move forward while rear vehicles move away.",
                baseline_highlights=("Vehicles in front",),
                b4dl_highlights=("rear vehicles",),
                output_dir=directory,
                front_instances=("instance-car",),
                source_note="Prediction loaded from a frozen result file.",
                answer_source="model result JSONL",
                provenance={"prediction": {"path": "pred.jsonl", "edited": False}},
            )
            self.assertEqual(artifact.frame_indices, (0, 2, 4, 6, 8))
            self.assertEqual(artifact.image.width, 2000)
            self.assertTrue(Path(artifact.png_path).is_file())
            self.assertTrue(Path(artifact.pdf_path).is_file())
            self.assertTrue(Path(artifact.metadata_path).is_file())
            with Image.open(artifact.png_path) as rendered:
                self.assertEqual(rendered.size, artifact.image.size)
                self.assertAlmostEqual(rendered.info["dpi"][0], 300, places=1)
            metadata = json.loads(Path(artifact.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["answers"]["left"]["label"], "Ground Truth")
            self.assertEqual(metadata["answers"]["right"]["label"], "B4DL prediction")
            self.assertEqual(metadata["provenance"]["prediction"]["edited"], False)
            self.assertEqual(metadata["frames"][0]["timestamp_us"], 1_000_000)
            self.assertEqual(metadata["frames"][-1]["relative_seconds"], 4.0)
            self.assertEqual(metadata["frames"][0]["coordinate_frame"], "ego")
            self.assertEqual(metadata["frames"][0]["target_annotation_tokens"], ["box"])
            self.assertEqual(metadata["overlays"]["front_instances"], ["instance-car"])
            self.assertIn("frozen result file", metadata["source_note"])
            self.assertEqual(metadata["frames"][0]["camera_availability"], {"CAM_FRONT": False, "CAM_BACK": False})
            self.assertEqual(len(metadata["missing_camera_panels"]), 10)
            self.assertIn("MISSING CAMERA IMAGES", metadata["source_note"])
            self.assertIn("Missing CAM_FRONT", " ".join(metadata["frames"][0]["warnings"]))
            self.assertEqual(metadata["rendered_sensor_rows"], ["LiDAR"])
            self.assertEqual(metadata["evidence_mode"], "lidar-only")
            self.assertIn("LiDAR-only evidence", metadata["source_note"])

    def test_missing_calibration_and_time_are_disclosed_in_metadata(self):
        repository = _PaperRepository()
        repository.nusc = type("EmptyNuScenes", (), {"get": staticmethod(lambda table, token: {"data": {}})})()
        with tempfile.TemporaryDirectory() as directory:
            artifact = build_paper_case(repository, "scene-token", (0, 1), "Question", "GT", "Prediction", output_dir=directory)
            metadata = json.loads(Path(artifact.metadata_path).read_text(encoding="utf-8"))
            self.assertIsNone(metadata["frames"][0]["timestamp_us"])
            self.assertIsNone(metadata["frames"][1]["relative_seconds"])
            self.assertEqual(metadata["frames"][0]["coordinate_frame"], "LIDAR_TOP sensor")
            self.assertIn("Calibration unavailable", metadata["source_note"])

    def test_frame_labels_and_timestamps_are_validated(self):
        repository = _PaperRepository()
        with self.assertRaisesRegex(ValueError, "数量一致"):
            build_paper_case(repository, "scene-token", (0, 1), "Q", "A", "B", frame_labels=("Only one",))
        with self.assertRaisesRegex(ValueError, "同时"):
            build_paper_case(repository, "scene-token", (0, 1), "Q", "A", "B", front_instances=("same",), back_instances=("same",))


if __name__ == "__main__":
    unittest.main()
