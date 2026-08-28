from pathlib import Path

import av
import numpy as np
import pytest

from app.ingestion.manifest import (
    ClipManifestRow,
    VariableFrameRateError,
    iter_rows,
    validate_columns,
)
from app.ingestion.video import probe, sampling, shot_detect
from app.schemas.ingestions import IngestionEntity
from tests.unit.test_shot_detect import write_segmented_video


def shot(start: int, end: int, shot_id: int = 0) -> ClipManifestRow:
    return ClipManifestRow(
        video_id="L01_V001",
        shot_id=shot_id,
        start_frame=start,
        end_frame=end,
        start_sec=start / 25,
        end_sec=end / 25,
        path="/data/videos/L01_V001.mp4",
    )


def read_jpeg(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


class TestPlanTargets:
    def test_one_second_of_shot_yields_one_keyframe(self):
        targets = sampling.plan_targets(shot(0, 24), fps=25.0)

        assert len(targets) == 1
        assert 0 <= targets[0] <= 24

    def test_count_scales_with_shot_duration(self):
        # Four seconds at 25fps.
        targets = sampling.plan_targets(shot(0, 99), fps=25.0)

        assert len(targets) == 4

    def test_rate_is_configurable(self):
        targets = sampling.plan_targets(
            shot(0, 99), fps=25.0, frames_per_second=2.0
        )

        assert len(targets) == 8

    def test_every_shot_gets_at_least_one_keyframe(self):
        targets = sampling.plan_targets(shot(10, 11), fps=25.0)

        assert len(targets) == 1
        assert 10 <= targets[0] <= 11

    def test_frames_per_shot_fixes_the_count_regardless_of_duration(self):
        """A fixed count decouples keyframes from how long a shot runs.

        The per-second rate gives a 1s shot one frame and a 40s shot forty; a
        fixed 3 covers both the same way and makes the corpus total predictable
        before the run.
        """
        short = sampling.plan_targets(shot(0, 24), fps=25.0, frames_per_shot=3)
        long = sampling.plan_targets(shot(0, 999), fps=25.0, frames_per_shot=3)

        assert len(short) == 3
        assert len(long) == 3

    def test_frames_per_shot_overrides_the_per_second_rate(self):
        targets = sampling.plan_targets(
            shot(0, 99), fps=25.0, frames_per_second=10.0, frames_per_shot=2
        )

        assert len(targets) == 2

    def test_frames_per_shot_is_clamped_to_the_usable_frames(self):
        """A two-frame shot yields two keyframes, not the three requested.

        Distinct frames are the ceiling: asking for more would repeat a frame
        index and, since identity is `(video_id, keyframe_n)`, quietly index the
        same image twice.
        """
        targets = sampling.plan_targets(shot(10, 11), fps=25.0, frames_per_shot=3)

        assert targets == [10, 11]

    def test_targets_avoid_the_shot_edges(self):
        targets = sampling.plan_targets(shot(0, 99), fps=25.0, boundary_inset=0.1)

        assert min(targets) >= 10
        assert max(targets) <= 89

    def test_targets_are_sorted_and_unique(self):
        targets = sampling.plan_targets(
            shot(0, 299), fps=25.0, frames_per_second=5.0
        )

        assert targets == sorted(set(targets))

    def test_targets_stay_inside_the_shot(self):
        targets = sampling.plan_targets(
            shot(100, 130), fps=25.0, frames_per_second=10.0
        )

        assert all(100 <= target <= 130 for target in targets)


class TestPlanWindows:
    def test_window_surrounds_the_target(self):
        windows = sampling.plan_windows(shot(0, 99), [50], sharpness_window=2)

        assert (windows[0].low, windows[0].target, windows[0].high) == (48, 50, 52)

    def test_windows_never_overlap(self):
        windows = sampling.plan_windows(shot(0, 99), [10, 13, 16], sharpness_window=3)

        for earlier, later in zip(windows, windows[1:], strict=False):
            assert later.low > earlier.high

    def test_windows_are_clamped_to_the_shot(self):
        windows = sampling.plan_windows(shot(10, 20), [10, 20], sharpness_window=5)

        assert windows[0].low == 10
        assert windows[-1].high == 20

    def test_zero_window_gives_single_frame_windows(self):
        windows = sampling.plan_windows(shot(0, 99), [50], sharpness_window=0)

        assert (windows[0].low, windows[0].high) == (50, 50)


class TestSharpness:
    def _frame(self, image: np.ndarray):
        return av.VideoFrame.from_ndarray(image, format="rgb24")

    def test_detailed_frame_scores_above_a_flat_one(self):
        flat = np.full((128, 128, 3), 128, dtype=np.uint8)
        detailed = np.zeros((128, 128, 3), dtype=np.uint8)
        detailed[::2] = 255

        assert sampling.sharpness(self._frame(detailed)) > sampling.sharpness(
            self._frame(flat)
        )

    def test_a_flat_frame_has_no_detail(self):
        flat = np.full((128, 128, 3), 90, dtype=np.uint8)

        assert sampling.sharpness(self._frame(flat)) == pytest.approx(0.0, abs=1e-6)


class TestWriteJpeg:
    def _frame(self, width: int, height: int):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[: height // 2, : width // 2] = 255
        return av.VideoFrame.from_ndarray(image, format="rgb24")

    def test_tall_frames_are_capped(self, tmp_path):
        destination = tmp_path / "frame.jpg"

        sampling.write_jpeg(self._frame(1920, 1080), destination, max_height=720)

        image = read_jpeg(destination)
        assert image.shape[0] == 720
        assert image.shape[1] == 1280

    def test_small_frames_are_not_upscaled(self, tmp_path):
        destination = tmp_path / "frame.jpg"

        sampling.write_jpeg(self._frame(320, 240), destination, max_height=720)

        assert read_jpeg(destination).shape[:2] == (240, 320)

    def test_quantiser_controls_file_size(self, tmp_path):
        rng = np.random.default_rng(0)
        noisy = av.VideoFrame.from_ndarray(
            rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), format="rgb24"
        )

        sampling.write_jpeg(noisy, tmp_path / "best.jpg", qscale=2)
        sampling.write_jpeg(noisy, tmp_path / "worst.jpg", qscale=25)

        assert (tmp_path / "best.jpg").stat().st_size > (
            tmp_path / "worst.jpg"
        ).stat().st_size

    def test_rotation_is_baked_into_the_file(self, tmp_path):
        # Marker in the top-left; a clockwise quarter turn moves it top-right.
        destination = tmp_path / "frame.jpg"

        sampling.write_jpeg(self._frame(320, 240), destination, rotation=90)

        image = read_jpeg(destination)
        assert image.shape[:2] == (320, 240)
        assert image[20, -20].mean() > 200
        assert image[20, 20].mean() < 60

    def test_half_turn_moves_the_marker_to_the_opposite_corner(self, tmp_path):
        destination = tmp_path / "frame.jpg"

        sampling.write_jpeg(self._frame(320, 240), destination, rotation=180)

        image = read_jpeg(destination)
        assert image.shape[:2] == (240, 320)
        assert image[-20, -20].mean() > 200
        assert image[20, 20].mean() < 60


class TestSampleVideo:
    def test_extracts_one_jpeg_per_planned_keyframe(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [50, 50], rate=25)
        video = probe.probe_video(source)
        shots = [shot(0, 49, 0), shot(50, 99, 1)]

        rows = sampling.sample_video(video, shots, tmp_path / "kf")

        assert len(rows) == 4
        assert all(Path(row.path).is_file() for row in rows)

    def test_rows_carry_the_owning_shot(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [50, 50], rate=25)
        video = probe.probe_video(source)
        shots = [shot(0, 49, 0), shot(50, 99, 1)]

        rows = sampling.sample_video(video, shots, tmp_path / "kf")

        for row in rows:
            expected = 0 if row.original_frame_id <= 49 else 1
            assert row.shot_id == expected

    def test_timestamps_follow_the_probed_rate(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [50], rate=25)
        video = probe.probe_video(source)

        rows = sampling.sample_video(video, [shot(0, 49)], tmp_path / "kf")

        for row in rows:
            assert row.pts_sec == pytest.approx(row.original_frame_id / 25)

    def test_extracted_files_are_named_after_the_frame(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [50], rate=25)
        video = probe.probe_video(source)

        rows = sampling.sample_video(video, [shot(0, 49)], tmp_path / "kf")

        for row in rows:
            assert Path(row.path).name == f"L01_V001_{row.original_frame_id:06d}.jpg"

    def test_frame_ids_are_unique_and_ordered(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [100], rate=25)
        video = probe.probe_video(source)

        rows = sampling.sample_video(video, [shot(0, 99)], tmp_path / "kf")

        ids = [row.original_frame_id for row in rows]
        assert ids == sorted(set(ids))

    def test_variable_frame_rate_is_refused(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [50], rate=25)
        video = probe.probe_video(source).model_copy(update={"is_vfr": True})

        with pytest.raises(VariableFrameRateError):
            sampling.sample_video(video, [shot(0, 49)], tmp_path / "kf")


class TestSharpestFrameIsChosen:
    def test_the_blurred_neighbour_loses(self, tmp_path):
        """One crisp frame among blurred ones must be the extracted keyframe."""
        source = tmp_path / "L01_V001.mp4"
        crisp_index = 12
        with av.open(str(source), "w") as container:
            stream = container.add_stream("libx264", rate=25)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "0"}

            for index in range(25):
                if index == crisp_index:
                    image = np.zeros((64, 64, 3), dtype=np.uint8)
                    image[::2] = 255  # high frequency detail
                else:
                    image = np.full((64, 64, 3), 120, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        video = probe.probe_video(source)
        # A window wide enough to reach the crisp frame from the shot centre.
        rows = sampling.sample_video(
            video, [shot(0, 24)], tmp_path / "kf", sharpness_window=6
        )

        assert [row.original_frame_id for row in rows] == [crisp_index]


class TestBuildKeyframeManifest:
    def _prepare(self, tmp_path):
        write_segmented_video(tmp_path / "videos" / "L01_V001.mp4", [50, 50])
        write_segmented_video(tmp_path / "videos" / "L01_V002.mp4", [50])
        videos_manifest = tmp_path / "videos.parquet"
        shots_manifest = tmp_path / "shots.parquet"
        probe.probe_directory(tmp_path / "videos", str(videos_manifest))
        shot_detect.build_shot_manifest(
            str(videos_manifest), str(shots_manifest), detector="content"
        )
        return videos_manifest, shots_manifest

    def test_writes_a_manifest_the_ingestion_layer_accepts(self, tmp_path):
        videos_manifest, shots_manifest = self._prepare(tmp_path)
        out = tmp_path / "keyframes.parquet"

        count = sampling.build_keyframe_manifest(
            str(videos_manifest),
            str(shots_manifest),
            str(tmp_path / "kf"),
            str(out),
        )

        assert count == 6
        validate_columns(str(out), IngestionEntity.FRAMES)
        rows = list(iter_rows(str(out), IngestionEntity.FRAMES))
        assert {row.video_id for row in rows} == {"L01_V001", "L01_V002"}
        assert all(Path(row.path).is_file() for row in rows)

    def test_a_video_without_shots_fails_loudly(self, tmp_path):
        videos_manifest, shots_manifest = self._prepare(tmp_path)
        # Drop one video's shots by rebuilding the manifest from a subset.
        kept = [
            row
            for row in iter_rows(str(shots_manifest), IngestionEntity.CLIPS)
            if row.video_id == "L01_V001"
        ]
        from app.ingestion.manifest import CLIP_ARROW_SCHEMA, write_rows

        partial = tmp_path / "partial.parquet"
        write_rows(kept, str(partial), CLIP_ARROW_SCHEMA)

        with pytest.raises(sampling.SamplingError, match="has no shots"):
            sampling.build_keyframe_manifest(
                str(videos_manifest),
                str(partial),
                str(tmp_path / "kf"),
                str(tmp_path / "out.parquet"),
            )

    def test_progress_is_reported_per_video(self, tmp_path):
        videos_manifest, shots_manifest = self._prepare(tmp_path)
        seen: list[tuple[str, int]] = []

        sampling.build_keyframe_manifest(
            str(videos_manifest),
            str(shots_manifest),
            str(tmp_path / "kf"),
            str(tmp_path / "keyframes.parquet"),
            on_progress=lambda video_id, count: seen.append((video_id, count)),
        )

        # Sorted, not positional: videos are sampled in worker processes and
        # progress fires as each finishes. Reporting a video the moment it lands
        # is the point of the callback, so completion order is correct here and
        # only the set of (video, count) pairs is the contract.
        assert sorted(seen) == [("L01_V001", 4), ("L01_V002", 2)]
