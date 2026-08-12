from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from app.ingestion.manifest import (
    VIDEO_ARROW_SCHEMA,
    VariableFrameRateError,
    VideoManifestRow,
    iter_rows,
    write_rows,
)
from app.ingestion.video import probe, shot_detect
from app.schemas.ingestions import IngestionEntity

SIZE = 64

# Solid colours far enough apart that a cut is unambiguous.
SEGMENT_COLOURS = [(220, 30, 30), (30, 220, 30), (30, 30, 220)]


def write_segmented_video(
    path: Path, segment_frames: list[int], rate: int = 25
) -> Path:
    """Encode one solid-colour segment per entry, so cuts are known exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = SIZE
        stream.height = SIZE
        stream.pix_fmt = "yuv420p"

        for segment, frames in enumerate(segment_frames):
            colour = SEGMENT_COLOURS[segment % len(SEGMENT_COLOURS)]
            image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            image[:, :] = colour
            for _ in range(frames):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return path


def probe_row(path: Path, **overrides) -> VideoManifestRow:
    row = probe.probe_video(path)
    return row.model_copy(update=overrides) if overrides else row


class TestFindCuts:
    def test_no_scores_over_threshold_yields_a_single_shot(self):
        scores = iter((index, 1.0) for index in range(1, 50))

        cuts, total = shot_detect.find_cuts(scores, threshold=27.0)

        assert cuts == []
        assert total == 50

    def test_cut_positions_are_the_frames_that_start_a_shot(self):
        scores = iter([(1, 2.0), (16, 90.0), (17, 2.0), (40, 90.0)])

        cuts, total = shot_detect.find_cuts(
            scores, threshold=27.0, min_shot_frames=15
        )

        assert cuts == [16, 40]
        assert total == 41

    def test_only_the_first_score_in_a_burst_becomes_a_cut(self):
        # A dissolve produces a run of high scores across its duration.
        scores = iter([(index, 90.0) for index in range(20, 31)])

        cuts, _ = shot_detect.find_cuts(scores, threshold=27.0, min_shot_frames=15)

        assert cuts == [20]

    def test_a_cut_too_close_to_the_video_start_is_suppressed(self):
        # The opening shot is measured from frame 0, so an early cut would
        # leave it shorter than the minimum.
        scores = iter([(2, 90.0), (30, 90.0)])

        cuts, _ = shot_detect.find_cuts(scores, threshold=27.0, min_shot_frames=15)

        assert cuts == [30]

    def test_empty_input_reports_no_frames(self):
        cuts, total = shot_detect.find_cuts(iter([]))

        assert cuts == []
        assert total == 1


class TestDetectShots:
    def test_finds_boundaries_between_solid_segments(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20, 20, 20])

        shots = shot_detect.detect_shots(probe_row(source))

        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [
            (0, 19),
            (20, 39),
            (40, 59),
        ]
        assert [shot.shot_id for shot in shots] == [0, 1, 2]

    def test_shot_ranges_are_contiguous_and_cover_every_frame(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20, 25, 18])

        shots = shot_detect.detect_shots(probe_row(source))

        assert shots[0].start_frame == 0
        assert shots[-1].end_frame == 62
        for earlier, later in zip(shots, shots[1:], strict=False):
            assert later.start_frame == earlier.end_frame + 1

    def test_a_static_video_is_one_shot(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [40])

        shots = shot_detect.detect_shots(probe_row(source))

        assert len(shots) == 1
        assert (shots[0].start_frame, shots[0].end_frame) == (0, 39)

    def test_timestamps_follow_the_probed_frame_rate(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20, 20], rate=25)

        shots = shot_detect.detect_shots(probe_row(source))

        assert shots[1].start_sec == pytest.approx(0.8)
        assert shots[0].end_sec == pytest.approx(19 / 25)

    def test_raising_the_threshold_suppresses_detection(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20, 20, 20])

        shots = shot_detect.detect_shots(probe_row(source), threshold=250.0)

        assert len(shots) == 1

    def test_min_shot_frames_merges_short_segments(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20, 5, 20])

        shots = shot_detect.detect_shots(probe_row(source), min_shot_frames=15)

        # The 5-frame segment is below the minimum, so its opening cut is
        # dropped and it stays inside the preceding shot.
        assert len(shots) == 2
        assert shots[0].start_frame == 0

    def test_shots_point_at_the_source_video(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20, 20])

        shots = shot_detect.detect_shots(probe_row(source))

        assert all(shot.path == str(source) for shot in shots)

    def test_variable_frame_rate_is_refused(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [20])

        with pytest.raises(VariableFrameRateError, match="variable frame rate"):
            shot_detect.detect_shots(probe_row(source, is_vfr=True))

    def test_unreadable_video_raises(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        row = VideoManifestRow(
            video_id="L01_V001",
            path=str(broken),
            fps_num=25,
            fps_den=1,
            duration_sec=1.0,
            width=64,
            height=64,
            codec="h264",
        )

        with pytest.raises(shot_detect.ShotDetectionError, match="cannot decode"):
            shot_detect.detect_shots(row)


class TestBuildShotManifest:
    def test_writes_a_manifest_the_ingestion_layer_accepts(self, tmp_path):
        write_segmented_video(tmp_path / "videos" / "L01_V001.mp4", [20, 20])
        write_segmented_video(tmp_path / "videos" / "L01_V002.mp4", [30])

        videos_manifest = tmp_path / "videos.parquet"
        probe.probe_directory(tmp_path / "videos", str(videos_manifest))

        out = tmp_path / "shots.parquet"
        count = shot_detect.build_shot_manifest(str(videos_manifest), str(out))

        assert count == 3
        rows = list(iter_rows(str(out), IngestionEntity.CLIPS))
        by_video: dict[str, list[int]] = {}
        for row in rows:
            by_video.setdefault(row.video_id, []).append(row.shot_id)
        assert by_video == {"L01_V001": [0, 1], "L01_V002": [0]}

    def test_progress_is_reported_per_video(self, tmp_path):
        write_segmented_video(tmp_path / "videos" / "L01_V001.mp4", [20, 20])
        videos_manifest = tmp_path / "videos.parquet"
        probe.probe_directory(tmp_path / "videos", str(videos_manifest))

        seen: list[tuple[str, int]] = []
        shot_detect.build_shot_manifest(
            str(videos_manifest),
            str(tmp_path / "shots.parquet"),
            on_progress=lambda video_id, count: seen.append((video_id, count)),
        )

        assert seen == [("L01_V001", 2)]

    def test_empty_probe_manifest_raises(self, tmp_path):
        videos_manifest = tmp_path / "videos.parquet"
        write_rows([], str(videos_manifest), VIDEO_ARROW_SCHEMA)

        with pytest.raises(shot_detect.ShotDetectionError, match="no videos in"):
            shot_detect.build_shot_manifest(
                str(videos_manifest), str(tmp_path / "shots.parquet")
            )


class TestFrameRateIsPreservedExactly:
    def test_ntsc_shot_timestamps_use_the_exact_fraction(self, tmp_path):
        source = write_segmented_video(
            tmp_path / "L01_V001.mp4", [20, 20], rate=Fraction(30000, 1001)
        )

        shots = shot_detect.detect_shots(probe_row(source))

        assert shots[1].start_sec == pytest.approx(20 * 1001 / 30000)
