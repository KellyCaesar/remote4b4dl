"""Publication-ready qualitative case figures for B4DL.

The renderer is deliberately independent from Gradio and the model stack.  It
loads already indexed nuScenes frames, projects available 3D annotations into
front/back cameras, draws a static LiDAR BEV, and composes a paper-style figure
that can be exported as PNG or PDF.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:  # package import
    from .lidar_visualizer import (
        BOX_EDGES,
        CATEGORY_COLORS,
        FrameData,
        NuScenesSceneRepository,
        category_group,
        quaternion_rotation_matrix,
    )
except ImportError:  # direct script / demo_gradio import
    from lidar_visualizer import (  # type: ignore
        BOX_EDGES,
        CATEGORY_COLORS,
        FrameData,
        NuScenesSceneRepository,
        category_group,
        quaternion_rotation_matrix,
    )


PAPER_BG = "#ffffff"
PAPER_PANEL = "#ffffff"
PAPER_INK = "#172028"
PAPER_MUTED = "#66727c"
PAPER_LINE = "#9da6ac"
PAPER_YELLOW = "#ffe84a"
PAPER_GREEN = "#70f08b"
PAPER_CYAN = "#00a7a0"


@dataclass(frozen=True)
class PaperCaseArtifact:
    image: Image.Image
    png_path: str
    pdf_path: str
    frame_indices: Tuple[int, ...]
    metadata_path: str = ""


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def select_frame_indices(
    frame_count: int,
    count: int = 5,
    requested: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Validate explicit frame indices or choose evenly spaced scene frames."""
    if frame_count < 2:
        raise ValueError("论文案例图至少需要 2 个可用场景帧")
    if requested is not None and len(requested) > 0:
        indices = tuple(int(value) for value in requested)
        if len(indices) < 2 or len(indices) > 8:
            raise ValueError("论文案例图需要 2 到 8 个帧号")
        if len(set(indices)) != len(indices):
            raise ValueError("帧号不能重复")
        invalid = [value for value in indices if value < 0 or value >= frame_count]
        if invalid:
            raise ValueError(
                f"帧号 {invalid} 超出场景范围 [0, {frame_count - 1}]"
            )
        return tuple(sorted(indices))
    count = max(2, min(int(count), frame_count, 8))
    values = np.linspace(0, frame_count - 1, num=count)
    return tuple(int(round(value)) for value in values)


def parse_frame_indices(value: str, frame_count: int, count: int = 5) -> Tuple[int, ...]:
    """Parse ``0, 8, 16, 24, 32`` or fall back to even spacing when blank."""
    text = str(value or "").strip()
    if not text:
        return select_frame_indices(frame_count, count=count)
    parts = [part.strip() for part in re.split(r"[,，;；\s]+", text) if part.strip()]
    try:
        requested = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("帧号必须是用逗号分隔的整数") from exc
    return select_frame_indices(frame_count, count=count, requested=requested)


def _fit_cell(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Letterbox the complete sensor view; never crop evidence at cell edges."""
    contained = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    cell = Image.new("RGB", size, PAPER_BG)
    cell.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return cell


def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Project all box corners: SDK corner order is not a BEV footprint."""
    values = sorted(set(points))
    if len(values) <= 1:
        return values
    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    lower, upper = [], []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def render_bev_image(
    frame: FrameData,
    size: Tuple[int, int] = (640, 640),
    range_m: float = 52.0,
    show_boxes: bool = True,
    show_tracks: bool = True,
    calibrated_sensor: Optional[Mapping[str, Sequence[float]]] = None,
    box_colors: Optional[Mapping[str, str]] = None,
) -> Image.Image:
    """Draw equal-metre axes: calibrated ego forward up, ego left to the left.

    FrameData is in LIDAR_TOP coordinates. Without calibration the panel is
    labelled as sensor coordinates instead of claiming an ego orientation.
    """
    width, height = size
    if width <= 0 or height <= 0 or not math.isfinite(range_m) or range_m <= 0:
        raise ValueError("BEV size and range must be positive and finite")
    extent = min(width, height)
    # Use a square map so all layers share the same +/- metre range.
    map_image = Image.new("RGB", (extent, extent), "#f7f7f9")
    draw = ImageDraw.Draw(map_image)
    cx = cy = extent / 2.0
    scale = extent / (2.0 * float(range_m))
    rotation = (quaternion_rotation_matrix(calibrated_sensor["rotation"])
                if calibrated_sensor is not None else np.eye(3))
    translation = (np.asarray(calibrated_sensor["translation"], dtype=np.float64)
                   if calibrated_sensor is not None else np.zeros(3))

    def ego_points(values) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)[..., :3] @ rotation.T + translation

    def xy(point) -> Tuple[float, float]:
        return cx - float(point[1]) * scale, cy - float(point[0]) * scale

    points = np.asarray(frame.points)
    if len(points):
        stride = max(1, len(points) // 22_000)
        sampled = ego_points(points[::stride])
        low = float(np.percentile(sampled[:, 2], 5))
        high = float(np.percentile(sampled[:, 2], 95))
        span = max(0.1, high - low)
        for point in sampled:
            px, py = xy(point)
            if 0 <= px < extent and 0 <= py < extent:
                t = max(0.0, min(1.0, (float(point[2]) - low) / span))
                color = (int(127 - 42 * t), int(130 - 60 * t), int(139 + 25 * t))
                draw.point((px, py), fill=color)

    if show_tracks:
        for track in frame.tracks:
            if len(track.points) >= 2:
                color = (box_colors or {}).get(track.token, "#8a8396")
                draw.line([xy(point) for point in ego_points(track.points)], fill=color, width=2)
    for box in frame.boxes:
        if show_boxes or box.token in (box_colors or {}):
            corners = ego_points(np.asarray(box.corners).T).T
            color = (box_colors or {}).get(box.token, CATEGORY_COLORS[category_group(box.category)])
            polygon = _convex_hull([xy(corner) for corner in corners.T])
            if len(polygon) >= 2:
                draw.line(polygon + polygon[:1], fill=color, width=3)

    if calibrated_sensor is not None:
        ego = [(cx, cy - 9), (cx - 6, cy + 7), (cx + 6, cy + 7)]
        draw.polygon(ego, fill=PAPER_INK)
    axes = "EGO: x ↑  y ←" if calibrated_sensor is not None else "SENSOR: x ↑  y ←"
    draw.text((8, extent - 22), axes, font=_font(12), fill=PAPER_MUTED)
    image = Image.new("RGB", size, PAPER_BG)
    image.paste(map_image, ((width - extent) // 2, (height - extent) // 2))
    return image


def _camera_available(frame: FrameData, camera: str) -> bool:
    path = frame.camera_paths.get(camera)
    return bool(path and Path(path).is_file())


def _missing_camera_image(camera: str) -> Image.Image:
    """Use an explicit, readable placeholder, never a camera-like backdrop."""
    image = Image.new("RGB", (640, 360), "#f3f3f5")
    draw = ImageDraw.Draw(image)
    for y, label, size, bold in ((128, camera, 38, True),
                                  (185, "IMAGE UNAVAILABLE", 30, False)):
        font = _font(size, bold)
        width = draw.textlength(label, font=font)
        draw.text(((image.width - width) / 2, y), label, font=font, fill=PAPER_MUTED)
    return image


def render_annotated_camera(
    repository: NuScenesSceneRepository,
    frame: FrameData,
    camera: str,
    show_boxes: bool = True,
    instance_colors: Optional[Mapping[str, str]] = None,
) -> Image.Image:
    """Return a camera image with nuScenes boxes projected through calibration."""
    # A missing camera must not acquire projected boxes: those annotations
    # would make the placeholder look like real visual evidence.
    if not _camera_available(frame, camera):
        return _missing_camera_image(camera)
    image = repository.camera_image(frame, camera).copy()
    if not show_boxes and not instance_colors:
        return image
    try:
        sample = repository.nusc.get("sample", frame.sample_token)
        camera_token = sample.get("data", {}).get(camera)
        if not camera_token:
            return image
        _, boxes, intrinsic = repository.nusc.get_sample_data(camera_token)
        matrix = np.asarray(intrinsic, dtype=np.float64)
        if matrix.shape != (3, 3):
            return image
        draw = ImageDraw.Draw(image)
        for box in boxes:
            annotation_token = str(getattr(box, "token", "") or "")
            instance_token = ""
            if instance_colors and annotation_token:
                try:
                    instance_token = repository.nusc.get("sample_annotation", annotation_token)["instance_token"]
                except (KeyError, TypeError, AttributeError):
                    pass
            target_color = (instance_colors or {}).get(instance_token)
            if not show_boxes and target_color is None:
                continue
            corners = np.asarray(box.corners(), dtype=np.float64)
            if corners.shape != (3, 8):
                continue
            depth = corners[2]
            projected = matrix @ corners
            projected[:2] /= np.maximum(projected[2:3], 1e-6)
            color = target_color or CATEGORY_COLORS[category_group(str(getattr(box, "name", "")))]
            for start, end in BOX_EDGES:
                if depth[start] <= 0.1 or depth[end] <= 0.1:
                    continue
                a = tuple(projected[:2, start])
                b = tuple(projected[:2, end])
                draw.line((a, b), fill=color, width=4)
        return image
    except Exception:
        # Custom/test datasets may not expose camera calibration or annotations.
        return image


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width: float,
) -> List[Tuple[int, int]]:
    """Wrap by measured glyph width, preserving offsets and explicit newlines.

    Space-separated words are kept together when possible. Chinese text and
    long unbroken tokens can wrap at any glyph, without dropping a suffix.
    """
    value = str(text or "")
    lines: List[Tuple[int, int]] = []
    start = position = 0
    last_space: Optional[int] = None
    while position < len(value):
        character = value[position]
        if character == "\n":
            lines.append((start, position))
            position += 1
            start, last_space = position, None
            continue
        if position > start and draw.textlength(value[start:position + 1], font=font) > width:
            end = last_space + 1 if last_space is not None and last_space > start else position
            lines.append((start, end))
            start = position = end
            last_space = None
            continue
        if character.isspace():
            last_space = position
        position += 1
    lines.append((start, len(value)))
    return lines


def _text_height(draw: ImageDraw.ImageDraw, text: str, width: int, size: int,
                 spacing: int = 7, bold: bool = False) -> int:
    return len(_wrapped_lines(draw, text, _font(size, bold), width)) * (size + spacing)


def _draw_fitted_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    max_size: int,
    min_size: int = 15,
    bold: bool = False,
    fill: str = PAPER_INK,
    spacing: int = 5,
) -> None:
    x0, y0, x1, y1 = box
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, bold)
        lines = _wrapped_lines(draw, text, font, x1 - x0)
        line_height = size + spacing
        if line_height * len(lines) <= y1 - y0 or size == min_size:
            for row, (start, end) in enumerate(lines):
                draw.text((x0, y0 + row * line_height), str(text or "")[start:end], font=font, fill=fill)
            return


def _phrase_spans(text: str, phrases: Iterable[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            spans.append(match.span())
    return spans


def _draw_highlighted_answer(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    phrases: Sequence[str],
    highlight: str,
    front_highlights: Sequence[str] = (),
    back_highlights: Sequence[str] = (),
) -> None:
    """Draw wrapped text while preserving configured phrase highlights."""
    x0, y0, x1, y1 = box
    font = _font(23)
    line_height = 30
    spans = [(start, end, highlight) for start, end in _phrase_spans(text, phrases)]
    spans += [(start, end, PAPER_YELLOW) for start, end in _phrase_spans(text, front_highlights)]
    spans += [(start, end, PAPER_GREEN) for start, end in _phrase_spans(text, back_highlights)]
    for row, (line_start, line_end) in enumerate(_wrapped_lines(draw, text, font, x1 - x0)):
        y = y0 + row * line_height
        for start, end, color in spans:
            start, end = max(start, line_start), min(end, line_end)
            if start < end:
                left = x0 + draw.textlength(text[line_start:start], font=font)
                right = x0 + draw.textlength(text[line_start:end], font=font)
                draw.rectangle((left, y + 2, right, y + 27), fill=color)
        draw.text((x0, y), text[line_start:line_end], font=font, fill=PAPER_INK)


def compose_paper_case(
    front_images: Sequence[Image.Image],
    back_images: Sequence[Image.Image],
    bev_images: Sequence[Image.Image],
    frame_indices: Sequence[int],
    question: str,
    baseline_answer: str,
    b4dl_answer: str,
    baseline_label: str = "Ground Truth",
    b4dl_label: str = "B4DL prediction",
    baseline_highlights: Sequence[str] = (),
    b4dl_highlights: Sequence[str] = (),
    title: str = "QUALITATIVE CASE STUDY",
    source_note: str = "",
    frame_labels: Optional[Sequence[str]] = None,
    frame_timestamps: Optional[Sequence[Optional[int]]] = None,
    front_highlights: Sequence[str] = (),
    back_highlights: Sequence[str] = (),
    target_legend: bool = False,
    front_available: Optional[Sequence[bool]] = None,
    back_available: Optional[Sequence[bool]] = None,
) -> Image.Image:
    """Compose synchronized views and answer comparison into one figure."""
    columns = len(frame_indices)
    if columns < 2 or not (
        len(front_images) == len(back_images) == len(bev_images) == columns
    ):
        raise ValueError("三组视图必须与帧号数量一致，且至少包含 2 帧")

    canvas_w = 2000
    for values in (frame_labels, frame_timestamps, front_available, back_available):
        if values is not None and len(values) != columns:
            raise ValueError("帧标签与时间戳必须与帧号数量一致")
    margin = 30
    label_w = 110
    gap = 8
    content_w = canvas_w - margin * 2 - label_w
    cell_w = (content_w - gap * (columns - 1)) // columns
    camera_h = max(96, round(cell_w * 9 / 16))
    rows = []
    if front_available is None or any(front_available):
        rows.append(("Front", front_images, camera_h))
    if back_available is None or any(back_available):
        rows.append(("Back", back_images, camera_h))
    rows.append(("LiDAR", bev_images, cell_w))
    row_heights = tuple(height for _, _, height in rows)
    top = 100
    row_gap = 10
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    question_h = max(66, _text_height(probe, question, canvas_w - 2 * margin - 155, 25) + 30)
    panel_gap = 16
    panel_w = (canvas_w - 2 * margin - panel_gap) // 2
    answer_h = max(146, max(_text_height(probe, str(answer or ""), panel_w - 48, 23)
                            for answer in (baseline_answer, b4dl_answer)) + 86)
    legend_h = 26 if front_highlights or back_highlights or target_legend else 0
    source_h = _text_height(probe, source_note, canvas_w - 2 * margin, 15, 5) if source_note else 0
    sensor_bottom = top + sum(row_heights) + (len(rows) - 1) * row_gap
    canvas_h = sensor_bottom + 20 + question_h + 16 + answer_h + legend_h + source_h + 40
    image = Image.new("RGB", (canvas_w, canvas_h), PAPER_BG)
    draw = ImageDraw.Draw(image)

    draw.text((margin, 16), title, font=_font(23, True), fill=PAPER_INK)
    camera_status = ""
    if front_available is not None and back_available is not None:
        if not any(front_available) and not any(back_available):
            camera_status = "LiDAR-only evidence · RGB unavailable"
        elif not all(front_available) or not all(back_available):
            camera_status = "Partial RGB evidence · Missing camera images"
    if camera_status:
        status_font = _font(17, True)
        status_width = draw.textlength(camera_status, font=status_font)
        draw.text((canvas_w - margin - status_width, 20), camera_status,
                  font=status_font, fill=PAPER_INK)
    arrow_y = 54
    arrow_x0 = margin + label_w
    arrow_x1 = canvas_w - margin
    draw.line((arrow_x0, arrow_y, arrow_x1, arrow_y), fill=PAPER_INK, width=4)
    draw.polygon(
        [(arrow_x1, arrow_y), (arrow_x1 - 18, arrow_y - 9), (arrow_x1 - 18, arrow_y + 9)],
        fill=PAPER_INK,
    )
    draw.text((arrow_x0 + (arrow_x1 - arrow_x0) / 2 - 25, 29), "Time", font=_font(17, True), fill=PAPER_INK)
    timestamps = list(frame_timestamps) if frame_timestamps is not None else [None] * columns
    origin = next((value for value in timestamps if value is not None), None)
    labels = list(frame_labels) if frame_labels is not None else [
        f"F{index:03d}" + (f"  {(timestamp - origin) / 1_000_000:+.1f}s"
                          if timestamp is not None and origin is not None else "")
        for index, timestamp in zip(frame_indices, timestamps)
    ]
    for column, label in enumerate(labels):
        x = margin + label_w + column * (cell_w + gap)
        _draw_fitted_text(draw, (x, 70, x + cell_w, 96), label, 15, 11, bold=True)

    for row_index, (label, images, cell_h) in enumerate(rows):
        y = top + sum(row_heights[:row_index]) + row_index * row_gap
        draw.text((margin, y + cell_h / 2 - 14), label, font=_font(21, True), fill=PAPER_INK)
        for column, (source, frame_index) in enumerate(zip(images, frame_indices)):
            x = margin + label_w + column * (cell_w + gap)
            cell = _fit_cell(source, (cell_w, cell_h))
            image.paste(cell, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=PAPER_LINE, width=1)

    question_y = sensor_bottom + 20
    draw.rounded_rectangle(
        (margin, question_y, canvas_w - margin, question_y + question_h),
        radius=4, fill=PAPER_PANEL, outline=PAPER_INK, width=2,
    )
    draw.text((margin + 18, question_y + 16), "Question:", font=_font(21, True), fill=PAPER_INK)
    _draw_fitted_text(
        draw,
        (margin + 137, question_y + 14, canvas_w - margin - 18, question_y + question_h - 12),
        question, max_size=25, min_size=25, spacing=7,
    )

    answers_y = question_y + question_h + 16
    panels = (
        (baseline_label, baseline_answer, baseline_highlights, PAPER_YELLOW),
        (b4dl_label, b4dl_answer, b4dl_highlights, PAPER_GREEN),
    )
    for column, (label, answer, phrases, color) in enumerate(panels):
        x = margin + column * (panel_w + panel_gap)
        draw.rounded_rectangle(
            (x, answers_y, x + panel_w, answers_y + answer_h),
            radius=4, fill=PAPER_PANEL, outline=PAPER_INK, width=2,
        )
        draw.text((x + 24, answers_y + 16), label, font=_font(21, True), fill=PAPER_INK)
        draw.line((x + 24, answers_y + 50, x + panel_w - 24, answers_y + 50), fill=PAPER_LINE, width=1)
        _draw_highlighted_answer(
            draw,
            (x + 24, answers_y + 66, x + panel_w - 24, answers_y + answer_h - 20),
            str(answer or ""), phrases, color, front_highlights, back_highlights,
        )

    footer_y = answers_y + answer_h + 12
    if legend_h:
        for offset, label, color in ((0, "Front target", PAPER_YELLOW), (180, "Back target", PAPER_GREEN)):
            draw.rectangle((margin + offset, footer_y + 3, margin + offset + 17, footer_y + 17), fill=color)
            draw.text((margin + offset + 25, footer_y), label, font=_font(15), fill=PAPER_INK)
        footer_y += legend_h
    if source_note:
        _draw_fitted_text(draw, (margin, footer_y, canvas_w - margin, footer_y + source_h),
                          source_note, 15, 15, fill=PAPER_MUTED, spacing=5)
    return image


def split_highlights(value: str) -> Tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"[,，;；\n]+", str(value or ""))
        if item.strip()
    )


def build_paper_case(
    repository: NuScenesSceneRepository,
    scene_token: str,
    frame_indices: Optional[Sequence[int]],
    question: str,
    baseline_answer: str,
    b4dl_answer: str,
    baseline_label: str = "Ground Truth",
    b4dl_label: str = "B4DL prediction",
    baseline_highlights: Sequence[str] = (),
    b4dl_highlights: Sequence[str] = (),
    title: str = "QUALITATIVE CASE STUDY",
    show_boxes: bool = True,
    show_tracks: bool = True,
    output_dir: Optional[str] = None,
    source_note: str = "",
    frame_labels: Optional[Sequence[str]] = None,
    frame_timestamps: Optional[Sequence[Optional[int]]] = None,
    front_highlights: Sequence[str] = (),
    back_highlights: Sequence[str] = (),
    front_instances: Sequence[str] = (),
    back_instances: Sequence[str] = (),
    answer_source: str = "user supplied",
    provenance: Optional[Mapping[str, object]] = None,
    range_m: float = 52.0,
) -> PaperCaseArtifact:
    """Load synchronized scene evidence and write PNG/PDF artifacts."""
    scene = repository.get_scene(scene_token)
    selected = (
        select_frame_indices(len(scene.sample_tokens), requested=frame_indices)
        if frame_indices is not None
        else select_frame_indices(len(scene.sample_tokens))
    )
    for values in (frame_labels, frame_timestamps):
        if values is not None and len(values) != len(selected):
            raise ValueError("帧标签与时间戳必须与帧号数量一致")
    if set(front_instances) & set(back_instances):
        raise ValueError("同一目标实例不能同时标为前方和后方")
    instance_colors = {str(token): PAPER_YELLOW for token in front_instances}
    instance_colors.update({str(token): PAPER_GREEN for token in back_instances})
    fronts: List[Image.Image] = []
    backs: List[Image.Image] = []
    bevs: List[Image.Image] = []
    frame_metadata = []
    timestamps: List[Optional[int]] = []
    for position, index in enumerate(selected):
        frame = repository.get_frame(scene_token, index)
        sample = {}
        calibrated = None
        warnings = list(getattr(frame, "warnings", ()))
        camera_availability = {camera: _camera_available(frame, camera)
                               for camera in ("CAM_FRONT", "CAM_BACK")}
        warnings.extend(f"Missing {camera} image; camera panel is a placeholder"
                        for camera, available in camera_availability.items() if not available)
        try:
            sample = repository.nusc.get("sample", frame.sample_token)
            lidar_token = sample.get("data", {}).get("LIDAR_TOP")
            if lidar_token:
                lidar_data = repository.nusc.get("sample_data", lidar_token)
                calibrated = repository.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        except (KeyError, TypeError, AttributeError) as exc:
            warnings.append(f"LiDAR calibration unavailable: {exc}")
        timestamp = (frame_timestamps[position] if frame_timestamps is not None
                     else sample.get("timestamp"))
        if timestamp is not None:
            timestamp = int(timestamp)
        timestamps.append(timestamp)
        box_colors = {}
        for box in frame.boxes:
            if not instance_colors:
                break
            try:
                instance_token = repository.nusc.get("sample_annotation", box.token)["instance_token"]
                if instance_token in instance_colors:
                    box_colors[box.token] = instance_colors[instance_token]
            except (KeyError, TypeError, AttributeError):
                pass
        fronts.append(render_annotated_camera(repository, frame, "CAM_FRONT", show_boxes, instance_colors))
        backs.append(render_annotated_camera(repository, frame, "CAM_BACK", show_boxes, instance_colors))
        bevs.append(render_bev_image(frame, range_m=range_m, show_boxes=show_boxes,
                                    show_tracks=show_tracks, calibrated_sensor=calibrated,
                                    box_colors=box_colors))
        frame_metadata.append({
            "frame_index": index, "sample_token": frame.sample_token,
            "timestamp_us": timestamp,
            "timestamp_source": "provided" if frame_timestamps is not None else "nuScenes sample",
            "coordinate_frame": "ego" if calibrated is not None else "LIDAR_TOP sensor",
            "target_annotation_tokens": list(box_colors),
            "camera_paths": {str(key): str(value) if value else None
                             for key, value in frame.camera_paths.items()},
            "camera_availability": camera_availability,
            "warnings": warnings,
        })
    overlay_note = ("nuScenes GT boxes" if show_boxes or instance_colors else "No box overlays")
    if show_tracks:
        overlay_note += "; GT annotation tracks"
    coordinate_note = ("Ego: forward up, left to the left" if all(
        item["coordinate_frame"] == "ego" for item in frame_metadata
    ) else "Calibration unavailable in one or more frames: sensor axes shown")
    automatic_note = f"{overlay_note}. {coordinate_note}. Text source: {answer_source}."
    missing_cameras = [f"F{record['frame_index']:03d}/{camera}"
                       for record in frame_metadata
                       for camera, available in record["camera_availability"].items() if not available]
    front_available = [record["camera_availability"]["CAM_FRONT"] for record in frame_metadata]
    back_available = [record["camera_availability"]["CAM_BACK"] for record in frame_metadata]
    rendered_rows = (["Front"] if any(front_available) else []) + (["Back"] if any(back_available) else []) + ["LiDAR"]
    missing_warning = ""
    if len(rendered_rows) == 1:
        missing_warning = "MISSING CAMERA IMAGES: RGB unavailable; LiDAR-only evidence. Front/Back rows omitted."
    elif missing_cameras:
        missing_warning = "MISSING CAMERA IMAGES: unavailable panels are placeholders; fully unavailable rows omitted."
    if missing_warning:
        automatic_note = missing_warning + " " + automatic_note
    if instance_colors:
        automatic_note += " Yellow/green identify selected front/back GT instances, not prediction correctness."
    combined_source_note = " ".join(value for value in (source_note, automatic_note) if value)
    image = compose_paper_case(
        fronts, backs, bevs, selected, question, baseline_answer, b4dl_answer,
        baseline_label=baseline_label, b4dl_label=b4dl_label,
        baseline_highlights=baseline_highlights, b4dl_highlights=b4dl_highlights,
        title=title,
        source_note=combined_source_note, frame_labels=frame_labels,
        frame_timestamps=timestamps, front_highlights=front_highlights,
        back_highlights=back_highlights, target_legend=bool(instance_colors),
        front_available=front_available, back_available=back_available,
    )
    directory = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="b4dl-paper-case-"))
    directory.mkdir(parents=True, exist_ok=True)
    identity = re.sub(r"[^a-zA-Z0-9_-]+", "-", scene.scene_id or scene.name).strip("-") or "scene"
    png_path = directory / f"paper-case-{identity}.png"
    pdf_path = directory / f"paper-case-{identity}.pdf"
    metadata_path = directory / f"paper-case-{identity}.json"
    image.save(png_path, format="PNG", dpi=(300, 300))
    image.save(pdf_path, format="PDF", resolution=300.0)
    origin = next((value for value in timestamps if value is not None), None)
    for record in frame_metadata:
        record["relative_seconds"] = ((record["timestamp_us"] - origin) / 1_000_000
                                      if record["timestamp_us"] is not None and origin is not None else None)
    metadata = {
        "schema_version": 1, "scene_token": scene_token, "scene_id": scene.scene_id,
        "scene_name": scene.name, "frames": frame_metadata,
        "relative_time_origin": "first selected frame with an available timestamp",
        "source_note": combined_source_note, "frame_labels": list(frame_labels) if frame_labels is not None else None,
        "question": question, "answers": {
            "left": {"label": baseline_label, "text": baseline_answer},
            "right": {"label": b4dl_label, "text": b4dl_answer},
        },
        "answer_source": answer_source, "answers_are_renderer_inputs": True,
        "missing_camera_panels": missing_cameras,
        "warnings": [missing_warning] if missing_warning else [],
        "rendered_sensor_rows": rendered_rows,
        "evidence_mode": "lidar-only" if len(rendered_rows) == 1 else "lidar+camera",
        "provenance": dict(provenance or {}),
        "overlays": {"source": "nuScenes ground-truth annotations", "all_boxes": show_boxes,
                     "tracks": show_tracks, "front_instances": list(front_instances),
                     "back_instances": list(back_instances),
                     "yellow": "front target", "green": "back target"},
        "highlights": {"front": list(front_highlights), "back": list(back_highlights),
                       "legacy_left": list(baseline_highlights), "legacy_right": list(b4dl_highlights)},
        "bev_range_m": range_m, "camera_fit": "contain without crop", "dpi": 300,
        "artifacts": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PaperCaseArtifact(image, str(png_path), str(pdf_path), selected, str(metadata_path))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 B4DL 论文风格定性案例图")
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--scene-metadata")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--frames", default="", help="逗号分隔帧号；留空自动均匀选择 5 帧")
    parser.add_argument("--question", required=True)
    parser.add_argument("--baseline-answer", required=True)
    parser.add_argument("--b4dl-answer", required=True)
    parser.add_argument("--baseline-label", default="Ground Truth")
    parser.add_argument("--b4dl-label", default="B4DL prediction")
    parser.add_argument("--baseline-highlights", default="")
    parser.add_argument("--b4dl-highlights", default="")
    parser.add_argument("--front-highlights", default="", help="两张答案卡共同使用的前方关注短语（黄色）")
    parser.add_argument("--back-highlights", default="", help="两张答案卡共同使用的后方关注短语（绿色）")
    parser.add_argument("--front-instances", default="", help="黄色目标的 nuScenes instance_token，逗号分隔")
    parser.add_argument("--back-instances", default="", help="绿色目标的 nuScenes instance_token，逗号分隔")
    parser.add_argument("--source-note", default="")
    parser.add_argument("--answer-source", default="user supplied")
    parser.add_argument("--provenance-file", help="可选 JSON 来源信息，原样写入 sidecar")
    parser.add_argument("--frame-labels", help="已排序所选帧的标签，逗号分隔")
    parser.add_argument("--frame-timestamps", help="已排序所选帧的绝对时间戳（微秒），逗号分隔")
    parser.add_argument("--bev-range", type=float, default=52.0, help="BEV 两轴相同的正负量程（米）")
    parser.add_argument("--title", default="QUALITATIVE CASE STUDY")
    parser.add_argument("--output-dir", default="paper_cases")
    parser.add_argument("--no-boxes", action="store_true")
    parser.add_argument("--no-tracks", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repository = NuScenesSceneRepository(
        args.nuscenes_root,
        version=args.nuscenes_version,
        scene_metadata=args.scene_metadata,
    )
    scene = repository.get_scene(args.scene_token)
    frames = parse_frame_indices(args.frames, len(scene.sample_tokens))
    artifact = build_paper_case(
        repository, args.scene_token, frames, args.question,
        args.baseline_answer, args.b4dl_answer,
        baseline_label=args.baseline_label,
        b4dl_label=args.b4dl_label,
        baseline_highlights=split_highlights(args.baseline_highlights),
        b4dl_highlights=split_highlights(args.b4dl_highlights),
        front_highlights=split_highlights(args.front_highlights),
        back_highlights=split_highlights(args.back_highlights),
        front_instances=split_highlights(args.front_instances),
        back_instances=split_highlights(args.back_instances),
        source_note=args.source_note, answer_source=args.answer_source,
        provenance=json.loads(Path(args.provenance_file).read_text(encoding="utf-8")) if args.provenance_file else None,
        frame_labels=tuple(args.frame_labels.split(",")) if args.frame_labels is not None else None,
        frame_timestamps=tuple(int(value.strip()) for value in args.frame_timestamps.split(",")) if args.frame_timestamps else None,
        range_m=args.bev_range,
        title=args.title,
        show_boxes=not args.no_boxes,
        show_tracks=not args.no_tracks,
        output_dir=args.output_dir,
    )
    print(artifact.png_path)
    print(artifact.pdf_path)
    print(artifact.metadata_path)


if __name__ == "__main__":
    main()
