"""TransNetV2 shot boundary detection.

TransNetV2 is a small (7.6M parameter) 3D CNN trained specifically for shot
transition detection. Unlike frame differencing it recognises gradual
transitions - dissolves, fades, wipes - which is where the content detector
gives up.

Frames are decoded here with PyAV rather than through the upstream package's
own video path, which shells out to an ffmpeg binary. Decoding locally keeps
the competition machine free of a system ffmpeg install and, more importantly,
keeps frame indexing under our control: index in presentation order *is*
`original_frame_id`.
"""

from functools import lru_cache
from typing import Any

import av
import numpy as np
from tqdm import tqdm

# TransNetV2 was trained on 48x27 RGB frames; the input size is fixed.
INPUT_WIDTH = 48
INPUT_HEIGHT = 27

DEFAULT_THRESHOLD = 0.5


class TransNetUnavailableError(Exception):
    pass


class TransNetInferenceError(Exception):
    pass


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any]:
    """Load TransNetV2 once per process, on GPU when one is present.

    The upstream package bundles its weights, so nothing is downloaded at
    run time and ingestion works on a machine with no network access.
    """
    try:
        import torch
        from transnetv2_pytorch import TransNetV2
    except ImportError as exc:
        raise TransNetUnavailableError(
            "TransNetV2 is not installed; install requirements.txt or run "
            "shot detection with --detector content"
        ) from exc

    model = TransNetV2(device="auto")
    model.eval()
    return model, torch


def decode_analysis_frames(video_path: str) -> np.ndarray:
    """Decode a whole video into the model's 48x27 RGB input tensor.

    The sliding window TransNetV2 uses needs the frames in one contiguous
    array, so a long video is held in memory at roughly 3.9KB per frame -
    about 350MB per hour at 25fps.
    """
    frames: list[np.ndarray] = []

    try:
        with av.open(video_path) as container:
            if not container.streams.video:
                raise TransNetInferenceError(f"no video stream in {video_path}")

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            decoded = tqdm(
                container.decode(stream),
                total=stream.frames or None,
                desc="decode",
                unit="frame",
                leave=False,
            )
            for frame in decoded:
                frames.append(
                    frame.reformat(
                        width=INPUT_WIDTH, height=INPUT_HEIGHT, format="rgb24"
                    ).to_ndarray()
                )
    except av.FFmpegError as exc:
        raise TransNetInferenceError(f"cannot decode {video_path}: {exc}") from exc

    if not frames:
        raise TransNetInferenceError(f"no decodable frames in {video_path}")

    return np.stack(frames)


def predict_transition_scores(video_path: str) -> np.ndarray:
    """Return the per-frame probability that a frame is part of a transition."""
    model, torch = load_model()
    frames = decode_analysis_frames(video_path)

    with torch.no_grad():
        single_frame_pred, _ = model.predict_frames(
            torch.from_numpy(frames).to(model.device), quiet=True
        )

    scores = single_frame_pred.cpu().numpy().reshape(-1)
    if scores.shape[0] != frames.shape[0]:
        raise TransNetInferenceError(
            f"model returned {scores.shape[0]} scores for {frames.shape[0]} frames"
        )
    return scores


def scenes_from_scores(
    scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> list[tuple[int, int]]:
    """Convert per-frame transition scores into inclusive shot ranges.

    A shot ends *on* the first flagged frame, matching the reference
    implementation: for a hard cut the flagged frame still shows the outgoing
    content, so putting it in the following shot would misattribute it.

    Where a transition spans several frames the reference drops the remainder
    entirely. They are kept here, attached to the incoming shot, so that every
    frame belongs to exactly one shot and the manifest stays contiguous.
    Sampling trims shot edges anyway, so those frames are unlikely to be
    chosen as keyframes.
    """
    total = len(scores)
    if total == 0:
        return []

    flagged = scores > threshold
    ends: list[int] = []
    in_transition = False

    for index, is_transition in enumerate(flagged):
        if is_transition:
            if not in_transition and index != 0:
                ends.append(index)
            in_transition = True
        else:
            in_transition = False

    starts = [0, *[end + 1 for end in ends]]
    closes = [*ends, total - 1]

    return [
        (start, close)
        for start, close in zip(starts, closes, strict=True)
        if start <= close
    ]


def merge_short_scenes(
    scenes: list[tuple[int, int]], min_shot_frames: int
) -> list[tuple[int, int]]:
    """Fold scenes below the minimum length into their neighbour."""
    if not scenes:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in scenes:
        if merged and (end - start + 1) < min_shot_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # The opening scene has no predecessor to merge into, so it absorbs the
    # one that follows instead.
    if len(merged) > 1 and (merged[0][1] - merged[0][0] + 1) < min_shot_frames:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)

    return merged


def detect_ranges(
    video_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    min_shot_frames: int = 0,
) -> list[tuple[int, int]]:
    scores = predict_transition_scores(video_path)
    scenes = scenes_from_scores(scores, threshold)
    return merge_short_scenes(scenes, min_shot_frames) if min_shot_frames else scenes
