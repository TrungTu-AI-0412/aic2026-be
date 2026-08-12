"""Fuse frame-level and clip-level hits into one ranked list.

Both collections are built from the same feature profile, so their scores are
cosine similarities in one shared space and can be combined arithmetically
without normalising first. They also share `(video_id, shot_id)`, which is the
key a shot is fused on: the frame hit carries the exact `original_frame_id`
that goes on a submission, the clip hit carries motion evidence that a single
keyframe cannot show. A shot both agree on should outrank one only half the
index found.
"""

from dataclasses import replace

from app.ranking.dedupe import best_per_shot
from app.vector_store.search import ScoredFrame

# Clips are mean-pooled over several frames, so they are a blurrier signal than
# a keyframe that matched exactly. Half weight keeps them as a tie-breaker and
# a recall net rather than something that can outvote the frame index.
DEFAULT_CLIP_WEIGHT = 0.5


def fuse_frames_and_clips(
    frames: list[ScoredFrame],
    clips: list[ScoredFrame],
    clip_weight: float = DEFAULT_CLIP_WEIGHT,
) -> list[ScoredFrame]:
    """Combine both result lists per shot, best fused score first."""
    if not clips or clip_weight <= 0:
        return list(frames)

    best_frames = best_per_shot(frames)
    best_clips = best_per_shot(clips)

    # A shot missing from one list is not evidence that it scores badly there:
    # it may simply have fallen outside that search's limit. Imputing the worst
    # score the list actually returned keeps one-sided shots competitive
    # without inventing a similarity nobody measured.
    # ponytail: if the two score scales ever drift apart, switch to reciprocal
    # rank fusion, which ignores magnitude entirely.
    frame_floor = min((hit.score for hit in best_frames.values()), default=0.0)
    clip_floor = min((hit.score for hit in best_clips.values()), default=0.0)

    fused: list[ScoredFrame] = []
    for key in best_frames.keys() | best_clips.keys():
        frame = best_frames.get(key)
        clip = best_clips.get(key)
        score = (frame.score if frame else frame_floor) + clip_weight * (
            clip.score if clip else clip_floor
        )
        # The frame hit is the carrier whenever there is one: only it knows
        # which exact frame to report.
        fused.append(replace(frame or clip, score=score))

    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused
