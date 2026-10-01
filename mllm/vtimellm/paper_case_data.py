"""Resolve saved evaluation samples into traceable paper-case evidence.

This module uses only the standard library.  Scene metadata must resolve before
any answer can be paired with images; invalid saved indices never fall back to
an unrelated scene or a full-scene montage.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Optional, Protocol, Sequence, Tuple


class _Scene(Protocol):
    scene_token: str
    sample_tokens: Sequence[str]


class _SceneRepository(Protocol):
    def get_scene_by_id(self, scene_id: str) -> _Scene: ...


class _EvaluationSample(Protocol):
    sample_id: str
    task: str
    scene_id: Optional[str]
    scene_token: Optional[str]
    feat_indices: Optional[Sequence[int]]
    feat_range: Optional[Sequence[int]]


@dataclass(frozen=True)
class PaperSampleSelection:
    scene_token: str
    frame_indices: Tuple[int, ...]
    source_note: str


def _integer(value: int, label: str) -> int:
    # operator.index accepts integer scalar types without truncating floats.
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为整数")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{label} 必须为整数") from exc


def select_evidence_frames(
    frame_count: int,
    feat_indices: Optional[Sequence[int]] = None,
    *,
    count: int = 5,
) -> Tuple[int, ...]:
    """Choose 2--8 chronological representatives from validated input frames.

    ``None`` means no saved input frames; only then are scene representatives
    allowed.  Repeated input rows share one displayed evidence frame, but every
    supplied row is validated before selection.  Fewer than two distinct frames
    cannot support a temporal montage and are rejected.
    """
    frame_count = _integer(frame_count, "场景帧数")
    count = _integer(count, "展示帧数")
    if frame_count < 2:
        raise ValueError("论文案例图至少需要 2 个可用场景帧")
    if not 2 <= count <= 8:
        raise ValueError("论文案例图展示帧数必须在 2 到 8 之间")

    if feat_indices is None:
        candidates = tuple(range(frame_count))
    else:
        supplied = tuple(_integer(value, "feat_indices 帧号") for value in feat_indices)
        invalid = [value for value in supplied if not 0 <= value < frame_count]
        if invalid:
            raise ValueError(
                f"feat_indices 帧号 {invalid} 超出场景范围 [0, {frame_count - 1}]；"
                "不能用其他场景帧代替"
            )
        candidates = tuple(sorted(set(supplied)))
        if len(candidates) < 2:
            raise ValueError("feat_indices 至少需要 2 个不同输入帧；不能扩展为全场景")

    selected_count = min(count, len(candidates))
    positions = (
        round(position * (len(candidates) - 1) / (selected_count - 1))
        for position in range(selected_count)
    )
    return tuple(candidates[position] for position in positions)


def _range_frames(feat_range: Sequence[int], frame_count: int) -> Tuple[int, ...]:
    if len(feat_range) != 2:
        raise ValueError("feat_range 必须为包含首尾帧的 [start, end]")
    start, end = (_integer(value, "feat_range 帧号") for value in feat_range)
    if start < 0 or end < start or end >= frame_count:
        raise ValueError(
            f"feat_range [{start}, {end}] 无效，场景范围为 [0, {frame_count - 1}]"
        )
    return select_evidence_frames(frame_count, tuple(range(start, end + 1)))


def _single_line(value: str) -> str:
    return " ".join(str(value).split())


def resolve_paper_sample(
    repository: _SceneRepository,
    sample: _EvaluationSample,
    *,
    evaluation_source: str = "saved evaluation",
) -> PaperSampleSelection:
    """Resolve one sample or raise ``ValueError`` to block mismatched export.

    Saved ``feat_indices`` take priority.  A saved inclusive ``feat_range`` can
    narrow the evidence when exact indices are absent, but does not establish
    the evaluated input: legacy evaluation may resolve a question-specific
    sequence instead.  The footer records that limitation explicitly.
    """
    if not sample.scene_id:
        raise ValueError("评测样本缺少 scene_id，无法可靠关联场景，禁止导出")
    try:
        scene = repository.get_scene_by_id(sample.scene_id)
    except KeyError as exc:
        raise ValueError(
            f"scene_id {sample.scene_id} 未映射，无法可靠关联场景，禁止导出"
        ) from exc
    if sample.scene_token and sample.scene_token != scene.scene_token:
        raise ValueError("评测样本 scene_token 与 scene_id 映射冲突，禁止导出")

    frame_count = len(scene.sample_tokens)
    if sample.feat_indices is not None:
        frames = select_evidence_frames(frame_count, sample.feat_indices)
        frame_source = "Frames: saved feat_indices"
    elif sample.feat_range is not None:
        frames = _range_frames(sample.feat_range, frame_count)
        frame_source = "Recorded range; exact input frames not saved"
    else:
        frames = select_evidence_frames(frame_count)
        frame_source = "Scene evidence; input frames not saved"

    source_note = (
        f"Sample: {_single_line(sample.sample_id)} | {_single_line(sample.task)} | "
        f"Eval: {_single_line(evaluation_source) or 'saved evaluation'} | "
        f"{frame_source} | Camera/GT auxiliary; model input: LiDAR"
    )
    return PaperSampleSelection(scene.scene_token, frames, source_note)
