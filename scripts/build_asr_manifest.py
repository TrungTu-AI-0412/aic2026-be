"""Build the ASR segment manifest from the enriched transcript CSVs.

One row per speech segment that carries a transcript. A segment is a time
range, not a keyframe, so this manifest feeds its own collection rather than
being folded into `frames.parquet`; retrieval joins the two back together on
time.

Everything needed is already in the CSVs. The per-video metadata columns
(`title`, `author`, `channel_id`, `watch_url`) are constant within a file and
repeated on every row -- which is why the corpus is 50MB for 40k segments --
so `data/media-info/` is not consulted.

Three properties of this source drive the parsing, all measured rather than
assumed (see docs/asr-transcripts.md):

- `text_corrected` is empty as the literal string "nan", not a blank cell.
  Writing it through would index the token "nan" across thousands of segments.
- `entities` is a Python dict repr, not JSON, so it needs `literal_eval`.
- `has_speech` is unreliable in both directions -- 162 segments have a
  transcript with the flag False, 216 have the flag True and no text -- so rows
  are kept or dropped on whether text is actually present.
"""

import argparse
import ast
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ingestion.manifest import (  # noqa: E402
    ASR_SEGMENT_ARROW_SCHEMA,
    AsrSegmentManifestRow,
    write_rows,
)

# The organiser's NER buckets. `others` is dropped: 900 segments and no defined
# meaning, so an index over it is something no query could sensibly ask for.
ENTITY_FIELDS = {
    "persons": "asr_persons",
    "orgs": "asr_orgs",
    "locations": "asr_locations",
}

# Spellings pandas and the exporter use for a missing value. Compared
# lowercased, so "NaN" and "NAN" are covered too.
_EMPTY = {"", "nan", "none", "null"}

# Segments below this many words are dropped. The 801 one-word segments in this
# corpus are almost entirely fillers -- "Ừ", "À", "thì", "Ờ", "và", "Dạ" -- and
# they are not merely useless but actively harmful: a one-character transcript
# still produces a dense vector, and because retrieval scores are normalised
# with the best hit at 1.0, such a segment can outrank genuinely relevant speech
# and hand a frame the full overlap bonus for saying nothing.
#
# Deliberately not 2. Two-word segments include real content -- a person's name
# ("Xuân Sơn") is exactly the query a competition run hangs on -- so cutting
# there would lose more than it removes.
DEFAULT_MIN_WORDS = 2


def clean(value: str | None) -> str:
    """Normalise a CSV cell, treating the null spellings as empty."""
    text = (value or "").strip()
    return "" if text.lower() in _EMPTY else text


def parse_entities(value: str | None) -> dict[str, list[str]]:
    """Split the entity dict into one list per type.

    A malformed cell yields no entities rather than failing the video: entities
    are a narrowing aid on 22% of segments, and losing them for one segment is
    a far smaller loss than losing the segment's transcript.
    """
    empty = {name: [] for name in ENTITY_FIELDS.values()}
    text = clean(value)
    if not text:
        return empty

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return empty
    if not isinstance(parsed, dict):
        return empty

    result = dict(empty)
    for source, field in ENTITY_FIELDS.items():
        mentions = parsed.get(source) or []
        if isinstance(mentions, (list, tuple, set)):
            # De-duplicated because a name repeated inside one segment would
            # otherwise appear twice in a keyword index that answers set
            # membership, where the repeat means nothing.
            seen: dict[str, None] = {}
            for mention in mentions:
                cleaned = clean(str(mention))
                if cleaned:
                    seen.setdefault(cleaned, None)
            result[field] = list(seen)
    return result


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(clean(value) or default)
    except ValueError:
        return default


def rows_for_csv(
    csv_path: Path, min_words: int = DEFAULT_MIN_WORDS
) -> list[AsrSegmentManifestRow]:
    """Parse one `<video_id>_segments_enriched.csv` into manifest rows."""
    rows: list[AsrSegmentManifestRow] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            corrected = clean(record.get("text_corrected"))
            if len(corrected.split()) < min_words:
                continue

            start = _float(record.get("start"))
            end = _float(record.get("end"))
            # Rounded whole seconds in the source, so a segment can appear to
            # end before it starts once both round the wrong way.
            if end < start:
                start, end = end, start

            video_id = clean(record.get("video_id")) or csv_path.stem.split("_segments")[0]
            entities = parse_entities(record.get("entities"))

            rows.append(
                AsrSegmentManifestRow(
                    video_id=video_id,
                    segment=int(_float(record.get("segment"), 0)),
                    start_sec=start,
                    end_sec=end,
                    duration=_float(record.get("duration")),
                    text_corrected=corrected,
                    speech_score=_float(record.get("speech_score")),
                    title=clean(record.get("title")),
                    author=clean(record.get("author")),
                    channel_id=clean(record.get("channel_id")),
                    watch_url=clean(record.get("watch_url")),
                    **entities,
                )
            )

    return rows


def build(
    transcripts_dir: str,
    out_path: str,
    limit: int | None = None,
    min_words: int = DEFAULT_MIN_WORDS,
) -> int:
    source = Path(transcripts_dir)
    if not source.is_dir():
        raise SystemExit(f"transcripts directory not found: {source}")

    files = sorted(source.glob("*_segments_enriched.csv"))
    if not files:
        raise SystemExit(f"no *_segments_enriched.csv under '{source}'")
    files = files[:limit]

    rows: list[AsrSegmentManifestRow] = []
    skipped = 0
    for csv_path in files:
        parsed = rows_for_csv(csv_path, min_words)
        if not parsed:
            skipped += 1
        rows.extend(parsed)

    if not rows:
        raise SystemExit(f"no segments with a transcript under '{source}'")

    written = write_rows(rows, out_path, ASR_SEGMENT_ARROW_SCHEMA)
    with_entities = sum(1 for row in rows if row.entity_terms())
    print(
        f"{written} segments from {len(files)} videos "
        f"({with_entities} with an entity, {skipped} videos had no transcript)"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build asr_segments.parquet from the enriched transcript CSVs."
    )
    parser.add_argument(
        "--transcripts", required=True, help="directory of *_segments_enriched.csv"
    )
    parser.add_argument("--out", required=True, help="output .parquet path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="read at most this many videos, for a trial slice",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"drop segments below this word count (default {DEFAULT_MIN_WORDS})",
    )
    args = parser.parse_args()
    build(args.transcripts, args.out, args.limit, args.min_words)


if __name__ == "__main__":
    main()
