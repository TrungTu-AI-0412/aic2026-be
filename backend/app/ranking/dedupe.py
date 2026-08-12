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


def overfetch_limit(top_k: int, factor: int = DEFAULT_OVERFETCH, cap: int = 1000) -> int:
    return min(max(top_k, 1) * factor, cap)


def dedupe_by_shot(frames: list[ScoredFrame], top_k: int) -> list[ScoredFrame]:
    """Keep the best-scoring frame per (video, shot), highest score first."""
    best: dict[tuple[str, int], ScoredFrame] = {}

    for frame in frames:
        key = (frame.video_id, frame.shot_id)
        current = best.get(key)
        if current is None or frame.score > current.score:
            best[key] = frame

    ranked = sorted(best.values(), key=lambda frame: frame.score, reverse=True)
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
