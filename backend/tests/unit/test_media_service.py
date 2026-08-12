import asyncio
from pathlib import Path

import av
import numpy as np
import pytest

from app.ingestion.manifest import VIDEO_ARROW_SCHEMA, VideoManifestRow, write_rows
from app.services.media import (
    FrameNotFoundError,
    LocalMediaService,
    VideoNotFoundError,
    _load_videos,
)
from tests.unit.test_feature_media import write_video

VIDEO_ID = "L01_V001"
RATE = 10


def build_root(tmp_path: Path, frame_count: int = 20) -> Path:
    source = tmp_path / "videos" / f"{VIDEO_ID}.mp4"
    source.parent.mkdir(parents=True)
    write_video(source, frame_count=frame_count, rate=RATE)

    keyframes = tmp_path / "keyframes" / VIDEO_ID
    keyframes.mkdir(parents=True)
    (keyframes / f"{VIDEO_ID}_000005.jpg").write_bytes(b"jpeg-bytes")

    write_rows(
        [
            VideoManifestRow(
                video_id=VIDEO_ID,
                path=str(source),
                fps_num=RATE,
                fps_den=1,
                nb_frames=frame_count,
                duration_sec=frame_count / RATE,
                width=32,
                height=32,
                codec="h264",
            )
        ],
        str(tmp_path / "manifests" / "videos.parquet"),
        VIDEO_ARROW_SCHEMA,
    )
    _load_videos.cache_clear()
    return tmp_path


def test_get_frame_returns_the_sampled_jpeg(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    frame = asyncio.run(service.get_frame(VIDEO_ID, 5))

    assert frame.content == b"jpeg-bytes"
    assert frame.media_type == "image/jpeg"


def test_unsampled_frame_and_unknown_video_are_distinguished(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    with pytest.raises(FrameNotFoundError):
        asyncio.run(service.get_frame(VIDEO_ID, 6))
    with pytest.raises(VideoNotFoundError):
        asyncio.run(service.get_frame("L99_V999", 5))


def test_video_id_cannot_escape_the_keyframe_root(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    with pytest.raises(VideoNotFoundError):
        asyncio.run(service.get_frame("../videos", 5))


def test_get_clip_encodes_the_requested_frame_range(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    clip = asyncio.run(service.get_clip(VIDEO_ID, 5, 14))

    assert clip.media_type == "video/mp4"
    with av.open(str(_write(tmp_path / "clip.mp4", clip.content))) as container:
        images = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    assert len(images) == 10
    # write_video ramps brightness by frame, so frames 5..14 mean ~50..~140.
    assert float(np.mean(images[0])) == pytest.approx(50, abs=8)
    assert float(np.mean(images[-1])) == pytest.approx(140, abs=8)


def test_get_clip_rejects_an_unknown_video(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    with pytest.raises(VideoNotFoundError):
        asyncio.run(service.get_clip("L99_V999", 0, 5))


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path
