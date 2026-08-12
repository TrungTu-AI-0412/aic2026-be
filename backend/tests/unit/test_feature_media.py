from pathlib import Path

import av
import numpy as np
import pytest

from app.features import media


def clip(path: str) -> media.ClipSegment:
    return media.ClipSegment(
        video_id="L01_V001",
        shot_id=2,
        start_frame=5,
        end_frame=14,
        start_sec=0.5,
        end_sec=1.4,
        path=path,
    )


def write_video(path: Path, frame_count: int = 20, rate: int = 10) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for index in range(frame_count):
            image = np.full((32, 32, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_clip_sampling_covers_the_shot(tmp_path):
    source = tmp_path / "L01_V001.mp4"
    write_video(source)

    images = media.sample_clip_frames(clip(str(source)), frame_count=4)

    assert len(images) == 4
    means = [float(image.mean()) for image in images]
    assert means == sorted(means)
    assert means[0] == pytest.approx(50, abs=5)
    assert means[-1] == pytest.approx(140, abs=5)
