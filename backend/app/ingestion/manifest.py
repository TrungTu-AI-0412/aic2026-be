from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, model_validator

from app.schemas.ingestions import IngestionEntity


class VariableFrameRateError(ValueError):
    """Raised when a single frame rate cannot map frame indexes to time."""


class ManifestRow(BaseModel):
    """Base for manifest rows that become Qdrant points.

    Subclasses decide how a row maps to a point id and a payload, so the
    ingestion pipeline never has to branch on the entity itself.
    """

    video_id: str = Field(min_length=1)
    shot_id: int = Field(ge=0)
    path: str = Field(min_length=1)

    def point_parts(self) -> tuple[str, ...]:
        raise NotImplementedError

    def payload(self) -> dict:
        raise NotImplementedError


class KeyframeManifestRow(ManifestRow):
    """One sampled keyframe. `original_frame_id` indexes the source video."""

    original_frame_id: int = Field(ge=0)
    pts_sec: float = Field(ge=0)

    def point_parts(self) -> tuple[str, ...]:
        return (self.video_id, str(self.original_frame_id))

    def payload(self) -> dict:
        return {
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "original_frame_id": self.original_frame_id,
            "pts_sec": self.pts_sec,
            "path": self.path,
        }


class ClipManifestRow(ManifestRow):
    """One shot detected by shot-boundary detection.

    `start_frame` and `end_frame` are source-video frame indexes and are
    both inclusive. `path` points at the source video, not an extracted
    clip file; the frame range narrows it.
    """

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> "ClipManifestRow":
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame {self.end_frame} precedes start_frame {self.start_frame}"
            )
        if self.end_sec < self.start_sec:
            raise ValueError(
                f"end_sec {self.end_sec} precedes start_sec {self.start_sec}"
            )
        return self

    def point_parts(self) -> tuple[str, ...]:
        return (self.video_id, f"shot{self.shot_id}")

    def payload(self) -> dict:
        return {
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "path": self.path,
        }


class VideoManifestRow(BaseModel):
    """Probe output for one source video.

    This manifest never reaches Qdrant; it is the rebuild/audit record that
    makes frame-index to timestamp mapping reproducible. Frame rate is kept
    as a fraction because rates like 30000/1001 drift once rounded to float.
    """

    video_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    fps_num: int = Field(gt=0)
    fps_den: int = Field(gt=0)
    nb_frames: int | None = Field(default=None, ge=0)
    duration_sec: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = 0
    is_vfr: bool = False
    codec: str = Field(min_length=1)

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    def require_constant_frame_rate(self) -> Fraction:
        """Return the frame rate, refusing variable-frame-rate sources.

        Every stage downstream stores positions as frame indexes and derives
        timestamps from a single rate. On a VFR source that arithmetic is
        quietly wrong, so callers must opt into PTS-based handling instead of
        getting a plausible but incorrect answer.
        """
        if self.is_vfr:
            raise VariableFrameRateError(
                f"'{self.video_id}' is variable frame rate; frame index to "
                "timestamp mapping needs presentation timestamps. Re-encode the "
                "source to a constant frame rate before ingesting it."
            )
        return self.fps

    def frame_to_sec(self, frame_id: int) -> float:
        return frame_id * self.fps_den / self.fps_num

    def sec_to_frame(self, sec: float) -> int:
        return round(sec * self.fps_num / self.fps_den)


KEYFRAME_ARROW_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("shot_id", pa.int32()),
        ("original_frame_id", pa.int64()),
        ("pts_sec", pa.float64()),
        ("path", pa.string()),
    ]
)

CLIP_ARROW_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("shot_id", pa.int32()),
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("start_sec", pa.float64()),
        ("end_sec", pa.float64()),
        ("path", pa.string()),
    ]
)

VIDEO_ARROW_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("path", pa.string()),
        ("fps_num", pa.int32()),
        ("fps_den", pa.int32()),
        ("nb_frames", pa.int64()),
        ("duration_sec", pa.float64()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("rotation", pa.int32()),
        ("is_vfr", pa.bool_()),
        ("codec", pa.string()),
    ]
)

ARROW_SCHEMAS: dict[IngestionEntity, pa.Schema] = {
    IngestionEntity.FRAMES: KEYFRAME_ARROW_SCHEMA,
    IngestionEntity.CLIPS: CLIP_ARROW_SCHEMA,
}

ROW_MODELS: dict[IngestionEntity, type[ManifestRow]] = {
    IngestionEntity.FRAMES: KeyframeManifestRow,
    IngestionEntity.CLIPS: ClipManifestRow,
}

REQUIRED_COLUMNS: dict[IngestionEntity, set[str]] = {
    entity: set(schema.names) for entity, schema in ARROW_SCHEMAS.items()
}

VIDEO_REQUIRED_COLUMNS = set(VIDEO_ARROW_SCHEMA.names)


def validate_columns(manifest_path: str, entity: IngestionEntity) -> None:
    _validate_against(manifest_path, REQUIRED_COLUMNS[entity])


def validate_video_columns(manifest_path: str) -> None:
    _validate_against(manifest_path, VIDEO_REQUIRED_COLUMNS)


def count_rows(manifest_path: str) -> int:
    return pq.ParquetFile(manifest_path).metadata.num_rows


def iter_rows(manifest_path: str, entity: IngestionEntity) -> Iterator[ManifestRow]:
    yield from _iter_as(manifest_path, ROW_MODELS[entity])


def iter_video_rows(manifest_path: str) -> Iterator[VideoManifestRow]:
    yield from _iter_as(manifest_path, VideoManifestRow)


def write_rows(
    rows: Iterable[BaseModel], out_path: str, schema: pa.Schema
) -> int:
    """Write rows with an explicit Arrow schema.

    The schema is passed rather than inferred so that all-null optional
    columns keep their declared type instead of collapsing to null.
    """
    records = [row.model_dump() for row in rows]
    table = pa.Table.from_pylist(records, schema=schema)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return table.num_rows


def _validate_against(manifest_path: str, required: set[str]) -> None:
    schema_names = set(pq.ParquetFile(manifest_path).schema_arrow.names)
    missing = required - schema_names
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")


def _iter_as[RowT: BaseModel](
    manifest_path: str, model: type[RowT]
) -> Iterator[RowT]:
    parquet_file = pq.ParquetFile(manifest_path)
    for batch in parquet_file.iter_batches():
        for record in batch.to_pylist():
            yield model.model_validate(record)
