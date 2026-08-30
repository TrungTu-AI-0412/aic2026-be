"""Fold Vietnamese VLM captions into the frame manifest.

Source: an external public set (Apache-2.0) produced by a Vietnamese
vision-language model, describing each keyframe in prose. It is caption output,
not OCR, despite what the column name might suggest.

Reusing it replaces a job that would otherwise take 27-49 hours of VLM
inference. Measured before adopting it:

    coverage of our keyframes    97.4%  (172,684 / 177,321)
    distinct captions per video  100% median, no video below 50%
    caption length               465 characters median
    agrees with our OCR          32.6% of shared tokens vs 12.3% for a
                                 random other frame's caption — 2.7x chance

That last number is the one that matters. It confirms both that the captions
describe the frame they are attached to, and that joining on
`(video_id, keyframe_n)` lands them correctly; a misjoin would have scored at
the random baseline. A separate English caption set was measured the same way,
reached only 2.1x, and is deliberately not used — see docs/data-pipeline.md.

TWO COLUMNS, NOT ONE

`caption_vi` is the prose description. `ocr_text_vlm` is the on-screen text the
model quoted, pulled out of the quotation marks it puts around it. They serve
different sparse vectors and must not be pooled: the caption is a paraphrase,
the quoted text is a transcription, and mixing them would let 465 characters of
scene description swamp a headline.

The quoted text matters because the VLM reads Vietnamese type far better than
EasyOCR does. Same keyframe, L21_V001 #18:

    EasyOCR   CẢMH BÁO SẠT LỎ ... ĐẾl VdI XE 3 BÁNH TRỈ LÊNl~ ... CHÚ Ý_qUAILSÁT
    VLM       "CẢNH BÁO SẠT LỞ NGUY HIỂM" / "TẠM DỪNG LƯU THÔNG"
              "ĐỐI VỚI XE 3 BÁNH TRỞ LÊN" / "NGƯỜI DÂN ĐI LẠI CHÚ Ý QUAN SÁT"

Correct, with every diacritic. So `ocr_text_vlm` is added alongside `ocr_text`
rather than replacing it — both feed the one `ocr` sparse vector, and the
EasyOCR reading is never destroyed.

FRAMES ONLY, NOT CLIPS

Captions are per-keyframe, unlike speech and on-screen text which are
shot-level by nature. Concatenating a shot's captions would put thousands of
characters on one row, and the clip collection exists mainly for the 10.4% of
shots that hold no keyframe — which have no caption either.

    python scripts/join_captions.py --captions data/ext_ocr/ocr \\
        --frames data/frames.parquet
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# The model wraps on-screen text in straight or curly quotes.
_QUOTED = re.compile(r'"([^"]{2,160})"|“([^”]{2,160})”')

# The model hedges in Vietnamese when it cannot read something, and that hedge
# lands inside the quotes. Indexing it would make every unreadable sign match
# the same words.
_HEDGES = (
    "không rõ",
    "không thể",
    "không đọc",
    "không nhìn",
    "mờ",
    "bị che",
)
_MIN_SPAN_CHARS = 2


class JoinError(RuntimeError):
    pass


def quoted_spans(caption: str) -> list[str]:
    """On-screen text the model transcribed, in the order it appeared."""
    spans = []
    for straight, curly in _QUOTED.findall(caption):
        span = (straight or curly).strip()
        if len(span) < _MIN_SPAN_CHARS:
            continue
        lowered = span.lower()
        if any(h in lowered for h in _HEDGES):
            continue
        if span not in spans:
            spans.append(span)
    return spans


def read_captions(path: Path) -> dict[int, str]:
    """One video's captions, keyed by keyframe ordinal.

    The files are UTF-16; read as UTF-8 they decode to interleaved nulls and
    every row is silently lost rather than raising.
    """
    try:
        with open(path, encoding="utf-16") as handle:
            rows = list(csv.DictReader(handle))
    except (UnicodeError, UnicodeDecodeError):
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    out: dict[int, str] = {}
    for row in rows:
        name = (row.get("keyframe") or "").strip()
        caption = (row.get("response") or "").strip()
        if not caption or not name.endswith(".jpg"):
            continue
        try:
            out[int(name[:-4])] = caption
        except ValueError:
            continue
    return out


def load_all(directory: Path) -> dict[tuple[str, int], str]:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise JoinError(f"no .csv under {directory}")
    captions: dict[tuple[str, int], str] = {}
    for path in files:
        video_id = path.stem
        for keyframe_n, caption in read_captions(path).items():
            captions[(video_id, keyframe_n)] = caption
    return captions


def join(args: argparse.Namespace) -> int:
    captions = load_all(args.captions)

    frames = pq.read_table(args.frames)
    keys = list(
        zip(
            frames.column("video_id").to_pylist(),
            [int(n) for n in frames.column("keyframe_n").to_pylist()],
        )
    )

    caption_col = [captions.get(key, "") for key in keys]
    vlm_col = [" ".join(quoted_spans(c)) if c else "" for c in caption_col]

    for name, values in (("caption_vi", caption_col), ("ocr_text_vlm", vlm_col)):
        if name in frames.column_names:
            frames = frames.drop_columns([name])
        frames = frames.append_column(name, pa.array(values, pa.string()))
    pq.write_table(frames, args.frames)

    total = frames.num_rows
    with_caption = sum(1 for c in caption_col if c)
    with_vlm = sum(1 for c in vlm_col if c)
    easy = frames.column("ocr_text").to_pylist()
    gained = sum(1 for e, v in zip(easy, vlm_col) if v and not e)

    print(f"captions loaded    : {len(captions):,}")
    print(f"frames             : {total:,}")
    print(f"  caption_vi       : {with_caption:,} ({with_caption / total:.1%})")
    print(f"  ocr_text_vlm     : {with_vlm:,} ({with_vlm / total:.1%})")
    print(f"  on-screen text where EasyOCR found none: {gained:,}")
    print(f"-> {args.frames}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=Path("data/ext_ocr/ocr"))
    parser.add_argument("--frames", type=Path, default=Path("data/frames.parquet"))
    args = parser.parse_args(argv)
    try:
        return join(args)
    except JoinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
