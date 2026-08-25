import asyncio
from pathlib import Path

import av
import numpy as np
import pytest

import pyarrow as pa
import pyarrow.parquet as pq

from app.ingestion.manifest import VIDEO_ARROW_SCHEMA, VideoManifestRow, write_rows
from app.services.media import (
    FrameNotFoundError,
    LocalMediaService,
    VideoNotFoundError,
    _load_keyframe_table,
    _load_videos,
    _video_keyframes,
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
    # Keyframes every 2 frames, 3 per shot: frame ids 0,2,4,... shots 0,0,0,1,1,1.
    frame_ids = [n * 2 for n in range(6)]
    pq.write_table(
        pa.table(
            {
                "video_id": [VIDEO_ID] * 6,
                "shot_id": [n // 3 for n in range(6)],
                "keyframe_n": [n + 1 for n in range(6)],
                "original_frame_id": frame_ids,
                "pts_sec": [f / RATE for f in frame_ids],
                "shot_start_sec": [0.0 if n < 3 else 0.6 for n in range(6)],
                "shot_end_sec": [0.4 if n < 3 else 1.0 for n in range(6)],
                "path": [f"{VIDEO_ID}_{f:06d}.jpg" for f in frame_ids],
            }
        ),
        tmp_path / "manifests" / "frames.parquet",
    )
    _load_videos.cache_clear()
    _load_keyframe_table.cache_clear()
    _video_keyframes.cache_clear()
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


def test_get_video_path_returns_the_source_file(tmp_path):
    root = build_root(tmp_path)
    service = LocalMediaService(data_root=str(root))

    path = asyncio.run(service.get_video_path(VIDEO_ID))

    assert path == (root / "videos" / f"{VIDEO_ID}.mp4").resolve()
    with pytest.raises(VideoNotFoundError):
        asyncio.run(service.get_video_path("L99_V999"))


def test_get_video_path_refuses_a_file_outside_the_data_root(tmp_path):
    root = build_root(tmp_path)
    service = LocalMediaService(
        data_root=str(root / "keyframes"),
        videos_manifest=str(root / "manifests" / "videos.parquet"),
    )

    with pytest.raises(VideoNotFoundError):
        asyncio.run(service.get_video_path(VIDEO_ID))


def test_stream_endpoint_serves_byte_ranges(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.deps import get_media_service
    from app.api.endpoints import media as endpoint

    root = build_root(tmp_path)
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/videos")
    app.dependency_overrides[get_media_service] = lambda: LocalMediaService(
        data_root=str(root)
    )
    client = TestClient(app)

    whole = client.get(f"/videos/{VIDEO_ID}/stream")
    assert whole.status_code == 200
    assert whole.headers["accept-ranges"] == "bytes"
    assert whole.headers["content-type"] == "video/mp4"

    part = client.get(f"/videos/{VIDEO_ID}/stream", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert len(part.content) == 100

    assert client.get("/videos/L99_V999/stream").status_code == 404


def test_video_timeline_exposes_exact_rate_and_all_keyframes(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    timeline = asyncio.run(service.get_video_timeline(VIDEO_ID))

    assert (timeline.fps_num, timeline.fps_den) == (RATE, 1)
    assert timeline.frame_count == 20
    assert timeline.duration_sec == pytest.approx(2.0)
    assert [frame.frame_id for frame in timeline.keyframes] == [0, 2, 4, 6, 8, 10]
    assert timeline.keyframes[3].shot_start_sec == pytest.approx(0.6)


def test_source_frame_is_decoded_by_exact_original_frame_id(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    source_frame = asyncio.run(service.get_source_frame(VIDEO_ID, 7))

    assert source_frame.media_type == "image/jpeg"
    with av.open(str(_write(tmp_path / "source-frame.jpg", source_frame.content))) as image:
        decoded = next(image.decode(video=0)).to_ndarray(format="rgb24")
    assert float(np.mean(decoded)) == pytest.approx(70, abs=8)

    with pytest.raises(FrameNotFoundError, match="outside the video"):
        asyncio.run(service.get_source_frame(VIDEO_ID, 20))


def test_timeline_and_source_frame_endpoints(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.deps import get_media_service
    from app.api.endpoints import media as endpoint

    root = build_root(tmp_path)
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/videos")
    app.dependency_overrides[get_media_service] = lambda: LocalMediaService(
        data_root=str(root)
    )
    client = TestClient(app)

    timeline = client.get(f"/videos/{VIDEO_ID}/timeline")
    assert timeline.status_code == 200
    assert timeline.json()["fps_num"] == RATE
    assert len(timeline.json()["keyframes"]) == 6

    frame = client.get(f"/videos/{VIDEO_ID}/source-frames/7")
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert "immutable" in frame.headers["cache-control"]
    assert client.get(f"/videos/{VIDEO_ID}/source-frames/-1").status_code == 422
    assert client.get(f"/videos/{VIDEO_ID}/source-frames/20").status_code == 404


def test_frame_context_carries_metadata_and_neighbours(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    context = asyncio.run(service.get_frame_context(VIDEO_ID, 6, radius=1))

    assert (context.keyframe_n, context.shot_id) == (4, 1)
    assert context.pts_sec == pytest.approx(0.6)
    assert (context.shot_start_frame, context.shot_end_frame) == (6, 10)
    assert (context.fps, context.width) == (RATE, 32)
    assert [n.frame_id for n in context.neighbours] == [4, 6, 8]
    # The window crosses a shot boundary, so the panel can show where it is.
    assert [n.is_same_shot for n in context.neighbours] == [False, True, True]


def test_frame_context_window_is_clipped_at_the_edges(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    context = asyncio.run(service.get_frame_context(VIDEO_ID, 0, radius=25))

    assert [n.frame_id for n in context.neighbours] == [0, 2, 4, 6, 8, 10]


def test_frame_context_rejects_a_frame_that_is_not_a_keyframe(tmp_path):
    service = LocalMediaService(data_root=str(build_root(tmp_path)))

    with pytest.raises(FrameNotFoundError):
        asyncio.run(service.get_frame_context(VIDEO_ID, 7, radius=1))
    with pytest.raises(VideoNotFoundError):
        asyncio.run(service.get_frame_context("L99_V999", 0, radius=1))


def test_frame_context_without_a_frames_manifest_says_so(tmp_path):
    root = build_root(tmp_path)
    (root / "manifests" / "frames.parquet").unlink()
    _load_keyframe_table.cache_clear()
    _video_keyframes.cache_clear()
    service = LocalMediaService(data_root=str(root))

    with pytest.raises(FrameNotFoundError):
        asyncio.run(service.get_frame_context(VIDEO_ID, 0, radius=1))
