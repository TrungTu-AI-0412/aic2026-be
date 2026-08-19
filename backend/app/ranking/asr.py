"""Boost frames whose shot overlaps speech that matches the query.

Speech and keyframes have no common key: a segment is a time range and a
keyframe is a moment, and the two were produced by unrelated processes. So they
live in separate collections and are joined here, on time, at query time.

Two scores have to be combined along the way, and neither combination can use
Qdrant's RRF. Within the speech collection, dense and lexical results are fused
with an explicit weight because the transcript is fluent Vietnamese: semantic
similarity carries most of the signal, and the lexical half is there to catch
the names, numbers and dates a dense encoder blurs. RRF fuses ranks and has no
weight to give. Then the speech score is added to the frame score, which needs
the two on a comparable scale, so speech scores are normalised to 0..1 first.
"""

from dataclasses import replace

from app.vector_store.search import AsrSegment, ScoredFrame

# Scores from the two branches are min-max normalised before being weighted.
# Cosine similarity and an IDF-weighted lexical score share no scale, so a raw
# weighted sum would be dominated by whichever happened to have the wider range
# on a given query rather than by the weight asked for.
DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3

# Share of a frame's score the speech bonus can contribute. Low on purpose: the
# image is the primary evidence and speech is corroboration, so a strong
# transcript match should be able to reorder near-ties without letting a frame
# that looks nothing like the query reach the top on words alone.
DEFAULT_WEIGHT = 0.3

# How far outside a shot a segment may sit and still count as covering it.
# Segment bounds in the source are rounded to whole seconds, and 4.5% of video
# time has no segment at all, so an exact test would miss real overlaps.
DEFAULT_PAD_SEC = 1.0


def normalize_scores(segments: list[AsrSegment]) -> list[AsrSegment]:
    """Rescale scores to 0..1, so they can be added to a frame's score.

    A single hit normalises to 1.0, and so does a set of tied hits: with no
    spread there is no ranking information to preserve, and mapping them to 0
    would silently discard the branch.
    """
    if not segments:
        return []

    scores = [segment.score for segment in segments]
    low, high = min(scores), max(scores)
    if high <= low:
        return [replace(segment, score=1.0) for segment in segments]

    span = high - low
    return [
        replace(segment, score=(segment.score - low) / span) for segment in segments
    ]


def fuse_asr(
    dense: list[AsrSegment],
    sparse: list[AsrSegment],
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> list[AsrSegment]:
    """Combine the dense and lexical speech branches into one ranked list.

    Each branch is normalised on its own before weighting, then summed per
    segment. A segment found by only one branch keeps just that branch's
    contribution rather than being imputed a score it did not earn: unlike the
    frame/clip fusion, the two branches here disagree about *which* segments are
    relevant, not merely about how relevant they are.
    """
    combined: dict[tuple[str, int], AsrSegment] = {}

    for segments, weight in ((dense, dense_weight), (sparse, sparse_weight)):
        if weight <= 0:
            continue
        for segment in normalize_scores(segments):
            key = (segment.video_id, segment.segment)
            current = combined.get(key)
            contribution = weight * segment.score
            if current is None:
                combined[key] = replace(segment, score=contribution)
            else:
                combined[key] = replace(
                    current, score=current.score + contribution
                )

    ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
    return ranked


def apply_asr_bonus(
    frames: list[ScoredFrame],
    segments: list[AsrSegment],
    weight: float,
    pad_sec: float = DEFAULT_PAD_SEC,
) -> list[ScoredFrame]:
    """Add `weight * best overlapping speech score` to each frame.

    Additive rather than multiplicative so a frame with no matching speech is
    left exactly as it was, instead of being penalised. 4.5% of video time has
    no ASR segment at all and 22 of 873 videos have no transcript, so absence of
    speech must never be read as evidence against a frame.

    The best overlapping segment wins rather than the sum of them. A long shot
    can span several segments, and adding them up would reward shot length
    instead of relevance.
    """
    if not frames or not segments or weight <= 0:
        return frames

    by_video: dict[str, list[AsrSegment]] = {}
    for segment in normalize_scores(segments):
        by_video.setdefault(segment.video_id, []).append(segment)

    boosted: list[ScoredFrame] = []
    for frame in frames:
        candidates = by_video.get(frame.video_id)
        window = frame.time_window(pad_sec) if candidates else None
        if not candidates or window is None:
            boosted.append(frame)
            continue

        low, high = window
        best = max(
            (
                segment.score
                for segment in candidates
                if segment.start_sec <= high and segment.end_sec >= low
            ),
            default=0.0,
        )
        boosted.append(
            replace(frame, score=frame.score + weight * best) if best else frame
        )

    boosted.sort(key=lambda frame: frame.score, reverse=True)
    return boosted
