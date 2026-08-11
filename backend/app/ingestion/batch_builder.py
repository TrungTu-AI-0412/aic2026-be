import argparse
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.schemas.ingestions import IngestionEntity

FRAME_NAME_PATTERN = re.compile(
    r"^(?P<video_id>L\d{2}_V\d{3})_(?P<frame_id>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def scan_frames(source_dir: Path) -> list[dict]:
    rows = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue

        match = FRAME_NAME_PATTERN.match(file_path.name)
        if not match:
            continue

        rows.append(
            {
                "video_id": match["video_id"],
                "frame_id": int(match["frame_id"]),
                "path": str(file_path),
            }
        )

    return rows


def scan_clips(source_dir: Path) -> list[dict]:
    raise NotImplementedError("clip manifest scanning is not implemented yet")


def build_manifest(source_dir: str, out_path: str, entity: str) -> int:
    source = Path(source_dir)
    if not source.is_dir():
        raise ValueError(f"source directory not found: {source_dir}")

    if entity == IngestionEntity.FRAMES.value:
        rows = scan_frames(source)
    elif entity == IngestionEntity.CLIPS.value:
        rows = scan_clips(source)
    else:
        raise ValueError(f"unknown entity '{entity}'")

    if not rows:
        raise ValueError(f"no {entity} files found under '{source_dir}'")

    table = pa.Table.from_pylist(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a local frame/clip directory and write a Parquet manifest."
    )
    parser.add_argument("--source", required=True, help="Directory already copied/mounted locally")
    parser.add_argument("--out", required=True, help="Output manifest .parquet path")
    parser.add_argument(
        "--entity",
        choices=[entity.value for entity in IngestionEntity],
        default=IngestionEntity.FRAMES.value,
    )
    args = parser.parse_args()

    count = build_manifest(args.source, args.out, args.entity)
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
