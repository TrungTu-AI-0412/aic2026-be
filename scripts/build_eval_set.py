"""Generate candidate evaluation queries from the manifests.

No public ground truth exists for this competition — the organiser has never
released past query sets, and none of the 2025 write-ups publish theirs. So the
choice is between labelling by hand and deriving something automatically. This
does the latter, and the result is weaker than hand labelling in one specific
way that has to be understood before any number from it is believed.

⚠ THE CIRCULARITY

A query built from a shot's ASR text is answered by that same ASR text sitting
in the lexical index. The speech branch therefore wins by construction, and
"hybrid search improved recall@1" would be measuring nothing but the echo.

So every item records the `source` channel it was derived from, and a run that
wants an honest read on channel X must exclude items whose source is X. What
survives is still useful:

  * ASR-sourced queries evaluated with the speech vectors off measure whether
    the *visual* index can find a moment described by what was said in it.
  * Any source measures regressions. If a change drops recall@5 by 10 points,
    that is real regardless of which channel earned the points originally.

What this set CANNOT do is tell you the system's absolute accuracy, or settle a
comparison between two channels where one of them wrote the questions. For
that, someone has to watch video and type queries.

DISTINCTIVENESS

A shot whose transcript is "và bây giờ chúng ta cùng theo dõi" identifies
nothing; hundreds of shots say it. Candidates are ranked by inverse document
frequency, so what gets picked is the shot that says something only it says.
Named entities count double, since a person or place name is what an operator
would actually type.

Three corrections that the obvious version of this gets wrong:

  * Scoring the *sum* over every token rewards long shots, not distinctive
    ones. A static-camera lecture running four minutes outscores a sharp
    ten-second news item on volume alone. Only the rarest `TOP_TOKENS` count.
  * Taking the first N words of a shot usually yields its greeting. The query
    is cut from the highest-IDF *window* instead, which is where the shot
    actually says something.
  * A shot holding 150 keyframes spans minutes, so "the moment" it contains is
    not well defined and any system finds it. Those are dropped.

Usage:

    python scripts/build_eval_set.py --clips data/clips.parquet \\
        --frames data/frames.parquet --out data/eval_set.jsonl --limit 300
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.features.sparse import tokenize  # noqa: E402

# Below this, a "query" is a fragment rather than a description of a moment.
MIN_QUERY_TOKENS = 6
# A textual KIS hint is a sentence, not a whole shot's transcript.
MAX_QUERY_TOKENS = 32
# Only the rarest tokens decide the score, so a four-minute lecture cannot
# outrank a ten-second news item purely by saying more words.
TOP_TOKENS = 8
# Entities are what an operator types, so they carry more weight than filler.
ENTITY_WEIGHT = 2.0
# Without this one talkative video would supply most of the set and the
# average would describe that video rather than the corpus.
MAX_PER_VIDEO = 3
# ~1 keyframe/sec, so this caps the answer at roughly half a minute. Past that
# the target stops being a moment and becomes a segment.
MAX_ANSWER_KEYFRAMES = 30
# IDF rewards misrecognition: a word the ASR invented is unique by definition,
# so "Bi tơ ri Cu nốp" outranks "Angelina Jolie". A token appearing in fewer
# than this many shots is treated as noise and left out of the score.
MIN_TOKEN_DF = 3


class BuildError(RuntimeError):
    pass


def document_frequency(texts: list[str]) -> tuple[Counter, int]:
    """How many shots each token appears in, and the shot count."""
    frequency: Counter = Counter()
    for text in texts:
        frequency.update(set(tokenize(text)))
    return frequency, len(texts)


def distinctiveness(
    text: str,
    entities: list[str],
    frequency: Counter,
    total: int,
    top_tokens: int = TOP_TOKENS,
) -> float:
    """IDF of the shot's `top_tokens` rarest words; entity tokens count double.

    Deliberately not the sum over every token. That version scores length:
    a long transcript accumulates more terms and wins even when none of them
    are rare, which fills the set with static-camera lectures.

    Uses the token set, not the list — a word repeated five times says no more
    about which shot this is than a word said once.
    """
    entity_tokens = {t for entity in entities for t in tokenize(entity)}
    scores = [
        math.log(total / (1 + frequency.get(token, 0)))
        * (ENTITY_WEIGHT if token in entity_tokens else 1.0)
        for token in set(tokenize(text))
        if frequency.get(token, 0) >= MIN_TOKEN_DF
    ]
    return sum(sorted(scores, reverse=True)[:top_tokens])


def word_terms(
    word: str, frequency: Counter, total: int, memo: dict
) -> list[tuple[str, float]]:
    """(token, idf) for one word, memoised — the corpus repeats words heavily."""
    cached = memo.get(word)
    if cached is None:
        cached = [
            (token, math.log(total / (1 + frequency.get(token, 0))))
            for token in set(tokenize(word))
            if frequency.get(token, 0) >= MIN_TOKEN_DF
        ]
        memo[word] = cached
    return cached


def best_window(
    text: str,
    entities: list[str],
    frequency: Counter,
    total: int,
    memo: dict,
    max_tokens: int = MAX_QUERY_TOKENS,
) -> str:
    """The `max_tokens`-word span carrying the most IDF mass.

    Taking the first N words instead would hand back "Xin chào tất cả các bạn"
    for most broadcast shots — the shot is distinctive, its opening is not.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text

    entity_tokens = {t for entity in entities for t in tokenize(entity)}
    scores = [
        sum(
            idf * (ENTITY_WEIGHT if token in entity_tokens else 1.0)
            for token, idf in word_terms(word, frequency, total, memo)
        )
        for word in words
    ]

    running = sum(scores[:max_tokens])
    best_total, best_start = running, 0
    for start in range(1, len(words) - max_tokens + 1):
        running += scores[start + max_tokens - 1] - scores[start - 1]
        if running > best_total:
            best_total, best_start = running, start
    return " ".join(words[best_start : best_start + max_tokens])


def load_shot_frames(frames_path: str) -> dict[tuple[str, int], list[int]]:
    """Every keyframe's `original_frame_id`, grouped by the shot it belongs to."""
    table = pq.read_table(
        frames_path, columns=["video_id", "shot_id", "original_frame_id"]
    )
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for video_id, shot_id, frame_id in zip(
        table.column("video_id").to_pylist(),
        table.column("shot_id").to_pylist(),
        table.column("original_frame_id").to_pylist(),
    ):
        grouped[(video_id, int(shot_id))].append(int(frame_id))
    return grouped


def build(args: argparse.Namespace) -> int:
    clips = pq.read_table(
        args.clips,
        columns=[
            "video_id",
            "shot_id",
            "asr_text",
            "asr_text_corrected",
            "asr_entities",
            "start_sec",
        ],
    )
    shot_frames = load_shot_frames(args.frames)

    rows = clips.to_pylist()
    # Corrected text reads like a sentence; the raw text is the fallback for
    # the shots the correction pass left empty.
    texts = [(r["asr_text_corrected"] or r["asr_text"] or "") for r in rows]
    frequency, total = document_frequency([t for t in texts if t])
    if not total:
        raise BuildError(f"no shot in {args.clips} carries any ASR text")

    scored = []
    memo: dict = {}
    for row, text in zip(rows, texts):
        tokens = tokenize(text)
        if len(tokens) < MIN_QUERY_TOKENS:
            continue
        key = (row["video_id"], int(row["shot_id"]))
        frames = shot_frames.get(key)
        if not frames:
            # A shot with no sampled keyframe can never be returned, so a
            # query pointing at it would score zero for every configuration.
            continue
        if len(frames) > args.max_answer_keyframes:
            continue
        entities = list(row["asr_entities"] or [])
        scored.append(
            (
                distinctiveness(text, entities, frequency, total),
                key,
                best_window(text, entities, frequency, total, memo),
                entities,
                float(row["start_sec"]),
                sorted(frames),
            )
        )

    # An ASR span usually covers several consecutive shots, so those shots
    # carry near-identical text. Emitting one query per shot would ask the
    # same question repeatedly and mark every answer but one wrong. Group
    # them: the question has several correct shots, which is simply true.
    #
    # Grouped on the emitted window, not the shot's full text: two spans that
    # differ only at their edges still produce one identical question, and it
    # is the question being asked that has to be unique.
    groups: dict[str, list] = defaultdict(list)
    for candidate in scored:
        groups[candidate[2]].append(candidate)

    grouped = []
    for text, members in groups.items():
        videos = {key[0] for _, key, _, _, _, _ in members}
        if len(videos) > 1:
            # The same sentence in two different videos is ambiguous, and no
            # ranking can be called right or wrong. Not a usable question.
            continue
        best = max(members, key=lambda item: item[0])
        grouped.append((best[0], best[1][0], text, best[3], members))
    grouped.sort(key=lambda item: item[0], reverse=True)

    per_video: Counter = Counter()
    selected = []
    for score, video_id, text, entities, members in grouped:
        if len(selected) >= args.limit:
            break
        if per_video[video_id] >= args.max_per_video:
            continue
        per_video[video_id] += 1
        shots = sorted(key[1] for _, key, _, _, _, _ in members)
        frames = sorted({f for _, _, _, _, _, fs in members for f in fs})
        selected.append(
            {
                "query_id": f"asr-{len(selected) + 1:04d}",
                "query_text": text,
                "source": "asr",
                "video_id": video_id,
                "answer_shot_ids": shots,
                "answer_frame_ids": frames,
                "start_sec": round(
                    min(start for _, _, _, _, start, _ in members), 3
                ),
                "entities": entities,
                "distinctiveness": round(score, 3),
                "reviewed": False,
            }
        )

    if not selected:
        raise BuildError("no shot met the length and keyframe requirements")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    multi = sum(1 for item in selected if len(item["answer_shot_ids"]) > 1)
    print(f"shots with ASR text : {total:,} / {len(rows):,}")
    print(f"eligible candidates : {len(scored):,}")
    print(f"distinct questions  : {len(grouped):,}")
    print(f"selected            : {len(selected):,}  -> {args.out}")
    print(f"  with >1 right shot: {multi:,}")
    print(f"videos covered      : {len(per_video):,}")
    print(f"source              : asr (exclude the speech channel to use these)")
    print(f"reviewed            : 0 — every item is a candidate, not ground truth")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("data/clips.parquet"))
    parser.add_argument("--frames", type=Path, default=Path("data/frames.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/eval_set.jsonl"))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-per-video", type=int, default=MAX_PER_VIDEO)
    parser.add_argument(
        "--max-answer-keyframes", type=int, default=MAX_ANSWER_KEYFRAMES
    )
    args = parser.parse_args(argv)
    try:
        return build(args)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
