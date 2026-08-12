from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from app.ingestion.manifest import iter_video_rows, validate_video_columns
from app.ingestion.video import probe

WIDTH = 64
HEIGHT = 48


def write_video(
    path: Path,
    *,
    rate: Fraction | int = 25,
    frames: int = 20,
    rotation: int | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        if rotation:
            stream.set_display_rotation(rotation)

        for index in range(frames):
            image = np.full((HEIGHT, WIDTH, 3), index * 7 % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return path


class TestProbeVideo:
    def test_reads_core_metadata(self, tmp_path):
        source = write_video(tmp_path / "L01_V001.mp4", rate=25, frames=20)

        row = probe.probe_video(source)

        assert row.video_id == "L01_V001"
        assert row.path == str(source)
        assert row.width == WIDTH
        assert row.height == HEIGHT
        assert row.codec == "h264"
        assert row.duration_sec == pytest.approx(20 / 25, rel=0.05)

    def test_ntsc_rate_is_kept_as_an_exact_fraction(self, tmp_path):
        source = write_video(tmp_path / "L01_V002.mp4", rate=Fraction(30000, 1001))

        row = probe.probe_video(source)

        assert (row.fps_num, row.fps_den) == (30000, 1001)
        assert row.fps == Fraction(30000, 1001)
        # The whole point of keeping the fraction: a float rate would not
        # survive this round trip on a long video.
        assert row.sec_to_frame(row.frame_to_sec(50_000)) == 50_000

    def test_video_id_can_be_overridden(self, tmp_path):
        source = write_video(tmp_path / "clip.mp4")

        assert probe.probe_video(source, video_id="L21_V123").video_id == "L21_V123"

    @pytest.mark.parametrize("rotation", [90, 180, 270])
    def test_display_rotation_is_recovered(self, tmp_path, rotation):
        source = write_video(tmp_path / f"rot{rotation}.mp4", rotation=rotation)

        assert probe.probe_video(source).rotation == rotation

    def test_absent_display_matrix_reads_as_no_rotation(self, tmp_path):
        source = write_video(tmp_path / "flat.mp4")

        assert probe.probe_video(source).rotation == 0

    def test_constant_frame_rate_is_not_flagged_as_vfr(self, tmp_path):
        source = write_video(tmp_path / "cfr.mp4", rate=25)

        assert probe.probe_video(source).is_vfr is False

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(probe.VideoProbeError, match="video not found"):
            probe.probe_video(tmp_path / "absent.mp4")

    def test_non_video_file_raises(self, tmp_path):
        broken = tmp_path / "notes.mp4"
        broken.write_bytes(b"this is not a video")

        with pytest.raises(probe.VideoProbeError, match="cannot probe"):
            probe.probe_video(broken)


class TestRotationMatrixDecoding:
    def test_identity_matrix_is_zero(self):
        identity = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)

        assert probe._rotation_from_display_matrix(identity) == 0

    def test_degenerate_matrix_falls_back_to_zero(self):
        assert probe._rotation_from_display_matrix((0,) * 9) == 0

    def test_unsupported_angle_is_rejected(self):
        # 45 degrees: cos/sin scaled to 16.16 fixed point.
        skewed = (46341, 46341, 0, -46341, 46341, 0, 0, 0, 1073741824)

        with pytest.raises(probe.VideoProbeError, match="unsupported display rotation"):
            probe._rotation_from_display_matrix(skewed)


class TestProbeDirectory:
    def test_writes_a_valid_videos_manifest(self, tmp_path):
        source_dir = tmp_path / "videos"
        write_video(source_dir / "L01_V001.mp4", rate=25)
        write_video(source_dir / "nested" / "L01_V002.mp4", rate=Fraction(30000, 1001))
        (source_dir / "readme.txt").write_text("ignored", encoding="utf-8")

        out = tmp_path / "videos.parquet"
        count = probe.probe_directory(source_dir, str(out))

        assert count == 2
        validate_video_columns(str(out))
        rows = {row.video_id: row for row in iter_video_rows(str(out))}
        assert set(rows) == {"L01_V001", "L01_V002"}
        assert rows["L01_V002"].fps == Fraction(30000, 1001)

    def test_directory_without_videos_raises(self, tmp_path):
        (tmp_path / "readme.txt").write_text("nothing here", encoding="utf-8")

        with pytest.raises(probe.VideoProbeError, match="no video files found"):
            probe.probe_directory(tmp_path, str(tmp_path / "out.parquet"))

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(probe.VideoProbeError, match="source directory not found"):
            probe.probe_directory(tmp_path / "absent", str(tmp_path / "out.parquet"))
