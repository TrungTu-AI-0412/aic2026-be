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


class VideoMetaFields(BaseModel):
    """Per-video metadata, identical on every point from the same video.

    Shared by every entity because it describes the source video rather than
    what was indexed from it. `publish_date` and `keywords` are deliberately
    absent: the date is a `dd/mm/yyyy` string that sorts wrong and filters
    badly, and keywords only ever existed to pad a frame-level speech vector
    that no longer exists.
    """

    title: str = ""
    author: str = ""
    channel_id: str = ""
    watch_url: str = ""

    def meta_payload(self) -> dict:
        values = {
            "title": self.title,
            "author": self.author,
            "channel_id": self.channel_id,
            "watch_url": self.watch_url,
        }
        return {name: value for name, value in values.items() if value}


class EnrichmentFields(VideoMetaFields):
    """Retrieval signals that are optional to the manifest contract.

    These come from the organiser's own artefacts, and every one is optional so
    a manifest written before they existed still validates. They stay out of
    the required Arrow schemas for the same reason.

    ASR text is *not* here. Segment-level speech is its own entity and its own
    collection, so a frame or clip payload no longer carries a pooled
    transcript.
    """

    objects: list[str] = Field(default_factory=list)
    object_counts: dict[str, int] = Field(default_factory=dict)

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

    # On-screen text, unioned over the shot. Held as recognised, including the
    # all-caps undiacriticked forms the news ticker uses: `app.features.sparse`
    # folds diacritics, so "Tam DUnG LuU Thong" still answers a query typed
    # "tạm dừng lưu thông". Normalising here would throw away the original.
    ocr_text: str = ""
    ocr_regions: int = 0

    def enrichment_payload(self) -> dict:
        """Only the fields that carry something.

        Writing empty strings and empty lists for every point would inflate
        the collection without making anything findable, and an empty value
        indexed as a keyword is a term that matches nothing.
        """
        values = {
            "objects": self.objects,
            "object_counts": self.object_counts,
            "ocr_text": self.ocr_text,
            "ocr_regions": self.ocr_regions,
            **self.meta_payload(),
        }
        return {name: value for name, value in values.items() if value}


class ManifestRow(EnrichmentFields):
    """Base for manifest rows that become Qdrant points.

    Subclasses decide how a row maps to a point id and a payload, so the
    ingestion pipeline never has to branch on the entity itself.

    Only `video_id` is common. `shot_id` and `path` belong to the entities that
    index a piece of *video*; an ASR segment has neither, and `path` is
    declared non-empty so it cannot simply be left blank.
    """

    video_id: str = Field(min_length=1)

    def point_parts(self) -> tuple[str, ...]:
        raise NotImplementedError

    def payload(self) -> dict:
        raise NotImplementedError


class ShotEntityRow(ManifestRow):
    """A row that indexes part of a source video, located by shot."""

    shot_id: int = Field(ge=0)
    path: str = Field(min_length=1)


class KeyframeManifestRow(ShotEntityRow):
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
    # The shot this keyframe was sampled from, in seconds. A keyframe on its own
    # is an instant, but the ASR overlap bonus has to ask whether a *span* of
    # speech covers it, and speech segments are far longer than one frame.
    shot_start_sec: float = Field(default=0.0, ge=0)
    shot_end_sec: float = Field(default=0.0, ge=0)

    def point_parts(self) -> tuple[str, ...]:
        return (self.video_id, f"kf{self.keyframe_n}")

    def payload(self) -> dict:
        return {
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "keyframe_n": self.keyframe_n,
            "original_frame_id": self.original_frame_id,
            "pts_sec": self.pts_sec,
            "shot_start_sec": self.shot_start_sec,
            "shot_end_sec": self.shot_end_sec,
            "path": self.path,
            **self.enrichment_payload(),
        }


class ClipManifestRow(ShotEntityRow):
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


class AsrSegmentManifestRow(ManifestRow, VideoMetaFields):
    """One speech segment, the unit ASR actually produces.

    A segment is a time range, not a frame, which is why speech gets its own
    collection rather than being pooled onto the keyframes it happens to cover:
    the two have no common key. Retrieval joins them back together on time.

    Only the corrected transcript is kept. Across the 40 023 segments in this
    corpus there is no row with raw text but no corrected text, so the raw
    column adds no reachable segment, and BM25 lowercases and strips
    punctuation anyway. The caveat to remember is that `text_corrected` is
    fluent but not accurate — an LLM added punctuation and capitalisation
    without fixing mishearings, so a wrong word now reads like a right one.

    Entities are split by type rather than pooled into one list, so a query can
    filter on a person without matching a location that shares the name.
    """

    segment: int = Field(ge=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    # The true segment length. `start`/`end` in the source are rounded to whole
    # seconds and disagree with this on 16% of rows, so anything needing real
    # duration must use this column.
    duration: float = Field(default=0.0, ge=0)
    text_corrected: str = ""
    speech_score: float = 0.0
    asr_persons: list[str] = Field(default_factory=list)
    asr_orgs: list[str] = Field(default_factory=list)
    asr_locations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "AsrSegmentManifestRow":
        if self.end_sec < self.start_sec:
            raise ValueError(
                f"end_sec {self.end_sec} precedes start_sec {self.start_sec}"
            )
        return self

    def point_parts(self) -> tuple[str, ...]:
        return (self.video_id, f"seg{self.segment}")

    def entity_terms(self) -> list[str]:
        """Every entity mention, for the lexical vector."""
        return [*self.asr_persons, *self.asr_orgs, *self.asr_locations]

    def payload(self) -> dict:
        values = {
            "video_id": self.video_id,
            "segment": self.segment,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration": self.duration,
            "text_corrected": self.text_corrected,
            "speech_score": self.speech_score,
            "asr_persons": self.asr_persons,
            "asr_orgs": self.asr_orgs,
            "asr_locations": self.asr_locations,
            **self.meta_payload(),
        }
        # `start_sec` is legitimately 0.0 on the first segment of every video and
        # has to survive, unlike an empty entity list that would index nothing.
        keep = {"video_id", "segment", "start_sec", "end_sec"}
        return {
            name: value for name, value in values.items() if value or name in keep
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
        ("shot_start_sec", pa.float64()),
        ("shot_end_sec", pa.float64()),
        ("path", pa.string()),
    ]
)

ASR_SEGMENT_ARROW_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("segment", pa.int32()),
        ("start_sec", pa.float64()),
        ("end_sec", pa.float64()),
        ("duration", pa.float64()),
        ("text_corrected", pa.string()),
        ("speech_score", pa.float64()),
        ("asr_persons", pa.list_(pa.string())),
        ("asr_orgs", pa.list_(pa.string())),
        ("asr_locations", pa.list_(pa.string())),
        ("title", pa.string()),
        ("author", pa.string()),
        ("channel_id", pa.string()),
        ("watch_url", pa.string()),
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
    IngestionEntity.ASR_SEGMENTS: ASR_SEGMENT_ARROW_SCHEMA,
}

ROW_MODELS: dict[IngestionEntity, type[ManifestRow]] = {
    IngestionEntity.FRAMES: KeyframeManifestRow,
    IngestionEntity.CLIPS: ClipManifestRow,
    IngestionEntity.ASR_SEGMENTS: AsrSegmentManifestRow,
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
        # Appending across a schema change is refused rather than reconciled.
        # `read_table(schema=...)` will happily back-fill a column the old file
        # lacks with nulls, which passes `validate_columns` (the column exists)
        # and only fails later when a row is parsed — i.e. after a multi-hour
        # decode pass has already been paid for. A stale manifest is a new
        # manifest, not something to merge into.
        existing = set(pq.ParquetFile(out).schema_arrow.names)
        missing = set(schema.names) - existing
        if missing:
            raise ValueError(
                f"'{out}' predates this schema and is missing "
                f"{sorted(missing)}; move it aside instead of resuming into it"
            )
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
