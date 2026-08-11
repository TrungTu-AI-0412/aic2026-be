from collections.abc import Iterator

import pyarrow.parquet as pq
from pydantic import BaseModel, Field

REQUIRED_COLUMNS = {"video_id", "frame_id", "path"}


class ManifestRow(BaseModel):
    video_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    path: str = Field(min_length=1)


def validate_columns(manifest_path: str) -> None:
    schema_names = set(pq.ParquetFile(manifest_path).schema_arrow.names)
    missing = REQUIRED_COLUMNS - schema_names
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")


def count_rows(manifest_path: str) -> int:
    return pq.ParquetFile(manifest_path).metadata.num_rows


def iter_rows(manifest_path: str) -> Iterator[ManifestRow]:
    parquet_file = pq.ParquetFile(manifest_path)
    for batch in parquet_file.iter_batches():
        for record in batch.to_pylist():
            yield ManifestRow.model_validate(record)
