"""Collapse near-duplicate hits before they reach the results list.

Sampling extracts roughly one keyframe per second, so a single shot yields
several near-identical vectors. Without collapsing them one scene can occupy
most of the top ranks and push away the other candidates an operator needs to
scan. This is what `shot_id` in the payload is for.
"""

from app.vector_store.search import ScoredFrame

# Raw hits are over-fetched by this factor so that collapsing duplicates still
# leaves enough distinct shots to fill the requested page.
DEFAULT_OVERFETCH = 5

# The cap binds only on TRAKE's video-selection stage, the one caller that asks
# for more than 200: a page of results never needs 1000 raw hits, but choosing
# *videos* does. Dense hits cluster inside a few long videos - the top 1000
# frames of "close-up of a white lion head" come from 31 videos, the top 5000
# from 277 - so at 1000 that stage could not fill a 100-video pool. 5000 raw
# hits measured 91ms against 293k points, up from 26ms.
DEFAULT_CAP = 5000


def overfetch_limit(
    top_k: int, factor: int = DEFAULT_OVERFETCH, cap: int = DEFAULT_CAP
) -> int:
    return min(max(top_k, 1) * factor, cap)


def best_per_shot(frames: list[ScoredFrame]) -> dict[tuple[str, int], ScoredFrame]:
    """Index the best-scoring hit per (video, shot).

    Fusion needs the same collapse keyed by shot, so the mapping lives here
    rather than being rebuilt next to it.
    """
    best: dict[tuple[str, int], ScoredFrame] = {}

    for frame in frames:
        key = (frame.video_id, frame.shot_id)
        current = best.get(key)
        if current is None or frame.score > current.score:
            best[key] = frame

    return best


def dedupe_by_shot(frames: list[ScoredFrame], top_k: int) -> list[ScoredFrame]:
    """Keep the best-scoring frame per (video, shot), highest score first."""
    ranked = sorted(
        best_per_shot(frames).values(), key=lambda frame: frame.score, reverse=True
    )
    return ranked[:top_k]


def dedupe_by_video(frames: list[ScoredFrame], top_k: int) -> list[ScoredFrame]:
    """Keep only the single best hit per video.

    Useful when an operator wants breadth across videos rather than depth
    within one.
    """
    best: dict[str, ScoredFrame] = {}

    for frame in frames:
        current = best.get(frame.video_id)
        if current is None or frame.score > current.score:
            best[frame.video_id] = frame

    ranked = sorted(best.values(), key=lambda frame: frame.score, reverse=True)
    return ranked[:top_k]
