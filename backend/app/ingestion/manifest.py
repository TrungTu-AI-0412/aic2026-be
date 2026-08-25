from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.ingestions import IngestionEntity


class VariableFrameRateError(ValueError):
    """Raised when a single frame rate cannot map frame indexes to time."""


class EnrichmentFields(BaseModel):
    """Retrieval signals that are optional to the manifest contract.

    These come from the organiser's own artefacts and from ASR, and every one
    is optional so a manifest written before they existed still validates.
    They stay out of the required Arrow schemas for the same reason.

    They are declared once and shared by frames and clips because a shot's
    speech and entities describe the shot, not the entity type used to index
    it.
    """

    objects: list[str] = Field(default_factory=list)
    object_counts: dict[str, int] = Field(default_factory=dict)
    asr_text: str = ""
    asr_text_corrected: str = ""
    asr_entities: list[str] = Field(default_factory=list)
    # On-screen text, unioned over the shot. Held as recognised, including the
    # all-caps undiacriticked forms the news ticker uses: `app.features.sparse`
    # folds diacritics, so "Tam DUnG LuU Thong" still answers a query typed
    # "tạm dừng lưu thông". Normalising here would throw away the original.
    ocr_text: str = ""
    ocr_regions: int = 0
    # On-screen text a Vietnamese VLM quoted out of the frame. Kept beside
    # `ocr_text` rather than replacing it: the VLM reads Vietnamese type far
    # better, but the recogniser sometimes catches small print the VLM skips,
    # and both feed the one `ocr` sparse vector.
    ocr_text_vlm: str = ""
    # Prose description of the frame. A paraphrase, not a transcription — it
    # gets its own sparse vector so 465 characters of scene description cannot
    # swamp a headline.
    caption_vi: str = ""
    title: str = ""
    author: str = ""
    channel_id: str = ""
    publish_date: str = ""
    keywords: list[str] = Field(default_factory=list)
    watch_url: str = ""

    @field_validator("object_counts", mode="before")
    @classmethod
    def _accept_arrow_map(cls, value: object) -> object:
        """Arrow renders `map<string, int32>` as a list of (key, value) pairs.

        `RecordBatch.to_pylist()` does not turn a map column back into a dict,
        so a manifest carrying `object_counts` failed validation on its very
        first row and ingestion never started. Accepting the pair list here
        keeps the fix at the one field whose Arrow type is a map.
        """
        if isinstance(value, list):
            return dict(value)
        return value

    def enrichment_payload(self) -> dict:
        """Only the fields that carry something.

        Writing empty strings and empty lists for every point would inflate
        the collection without making anything findable, and an empty value
        indexed as a keyword is a term that matches nothing.
        """
        values = {
            "objects": self.objects,
            "object_counts": self.object_counts,
            "asr_text": self.asr_text,
            "asr_text_corrected": self.asr_text_corrected,
            "asr_entities": self.asr_entities,
            "ocr_text": self.ocr_text,
            "ocr_regions": self.ocr_regions,
            "ocr_text_vlm": self.ocr_text_vlm,
            "caption_vi": self.caption_vi,
            "title": self.title,
            "author": self.author,
            "channel_id": self.channel_id,
            "publish_date": self.publish_date,
            "keywords": self.keywords,
            "watch_url": self.watch_url,
        }
        return {name: value for name, value in values.items() if value}


class ManifestRow(EnrichmentFields):
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
    """One sampled keyframe.

    `keyframe_n` is the keyframe's 1-based position within its video and is
    the row's identity. `original_frame_id` indexes the source video and is
    what a submission reports, but it does **not** identify a keyframe: the
    organiser's own `map-keyframes` CSVs give two consecutive keyframes the
    same `frame_idx` in 192 of 873 videos (614 keyframes), because the frame
    index is derived from a rounded presentation timestamp. Deriving the point
    id from it made those keyframes overwrite each other in Qdrant with no
    error raised.
    """

    keyframe_n: int = Field(ge=1)
    original_frame_id: int = Field(ge=0)
    pts_sec: float = Field(ge=0)

    def point_parts(self) -> tuple[str, ...]:
        return (self.video_id, f"kf{self.keyframe_n}")

    def payload(self) -> dict:
        return {
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "keyframe_n": self.keyframe_n,
            "original_frame_id": self.original_frame_id,
            "pts_sec": self.pts_sec,
            "path": self.path,
            **self.enrichment_payload(),
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
            **self.enrichment_payload(),
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
        ("keyframe_n", pa.int32()),
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


def existing_video_ids(manifest_path: str) -> set[str]:
    """`video_id`s already recorded in a manifest; empty when there is no file.

    This is what lets a stage be re-run over a bigger slice of the dataset
    without redoing the videos an earlier, smaller run already covered.
    """
    if not Path(manifest_path).is_file():
        return set()
    column = pq.read_table(manifest_path, columns=["video_id"]).column("video_id")
    return set(column.to_pylist())


def write_rows(
    rows: Iterable[BaseModel], out_path: str, schema: pa.Schema, append: bool = False
) -> int:
    """Write rows with an explicit Arrow schema, returning the manifest size.

    The schema is passed rather than inferred so that all-null optional
    columns keep their declared type instead of collapsing to null.

    Appending rewrites the whole file from the old rows plus the new ones.
    Parquet cannot extend a file in place, and a manifest of a few million
    rows is small enough that rewriting it costs less than the row-group
    bookkeeping needed to avoid it.
    """
    records = [row.model_dump() for row in rows]
    table = pa.Table.from_pylist(records, schema=schema)
    out = Path(out_path)

    if append and out.is_file():
        table = pa.concat_tables([pq.read_table(out, schema=schema), table])

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
