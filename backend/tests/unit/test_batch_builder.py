from fractions import Fraction
from pathlib import Path

import pytest

from app.ingestion import batch_builder
from app.ingestion.manifest import (
    CLIP_ARROW_SCHEMA,
    VIDEO_ARROW_SCHEMA,
    ClipManifestRow,
    VideoManifestRow,
    iter_rows,
    write_rows,
)
from app.schemas.ingestions import IngestionEntity


def make_keyframes(root: Path, video_id: str, frame_ids: list[int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        (root / f"{video_id}_{frame_id}.jpg").write_bytes(b"jpeg")
    return root


def make_videos_manifest(
    path: Path, video_id: str = "L01_V001", fps_num: int = 25, fps_den: int = 1
) -> str:
    row = VideoManifestRow(
        video_id=video_id,
        path=f"/data/videos/{video_id}.mp4",
        fps_num=fps_num,
        fps_den=fps_den,
        nb_frames=1000,
        duration_sec=40.0,
        width=1280,
        height=720,
        rotation=0,
        is_vfr=False,
        codec="h264",
    )
    write_rows([row], str(path), VIDEO_ARROW_SCHEMA)
    return str(path)


def make_shots_manifest(path: Path, ranges: list[tuple[int, int]]) -> str:
    rows = [
        ClipManifestRow(
            video_id="L01_V001",
            shot_id=shot_id,
            start_frame=start,
            end_frame=end,
            start_sec=start / 25,
            end_sec=end / 25,
            path="/data/videos/L01_V001.mp4",
        )
        for shot_id, (start, end) in enumerate(ranges)
    ]
    write_rows(rows, str(path), CLIP_ARROW_SCHEMA)
    return str(path)


class TestParseFps:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("25", Fraction(25)), ("30000/1001", Fraction(30000, 1001))],
    )
    def test_accepts_integer_and_fractional_rates(self, text, expected):
        assert batch_builder.parse_fps(text) == expected

    @pytest.mark.parametrize("text", ["0", "-25", "abc", "25/0"])
    def test_rejects_invalid_rates(self, text):
        with pytest.raises(ValueError):
            batch_builder.parse_fps(text)


class TestFrameRateResolution:
    def test_requires_a_declared_frame_rate(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [0, 10])

        with pytest.raises(ValueError, match="either --videos-manifest or --fps"):
            batch_builder.build_keyframe_manifest(
                str(source), str(tmp_path / "out.parquet")
            )

    def test_probe_manifest_wins_over_the_fallback(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [50])
        videos = make_videos_manifest(tmp_path / "videos.parquet", fps_num=50)
        out = tmp_path / "keyframes.parquet"

        batch_builder.build_keyframe_manifest(
            str(source), str(out), videos_manifest=videos, fps="10"
        )

        row = next(iter_rows(str(out), IngestionEntity.FRAMES))
        assert row.pts_sec == pytest.approx(1.0)

    def test_video_missing_from_probe_manifest_without_fallback_raises(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L02_V009", [0])
        videos = make_videos_manifest(tmp_path / "videos.parquet")

        with pytest.raises(ValueError, match="no frame rate for 'L02_V009'"):
            batch_builder.build_keyframe_manifest(
                str(source), str(tmp_path / "out.parquet"), videos_manifest=videos
            )


class TestKeyframeManifestWithoutShots:
    def test_each_keyframe_becomes_its_own_shot(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [30, 10, 20])
        out = tmp_path / "keyframes.parquet"

        count = batch_builder.build_keyframe_manifest(str(source), str(out), fps="25")

        assert count == 3
        rows = list(iter_rows(str(out), IngestionEntity.FRAMES))
        # Ordered by frame id, and no two keyframes share a shot, so
        # shot-level dedupe cannot collapse unrelated frames.
        assert [row.original_frame_id for row in rows] == [10, 20, 30]
        assert [row.shot_id for row in rows] == [0, 1, 2]

    def test_timestamps_use_the_declared_rate(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [30000])
        out = tmp_path / "keyframes.parquet"

        batch_builder.build_keyframe_manifest(str(source), str(out), fps="30000/1001")

        row = next(iter_rows(str(out), IngestionEntity.FRAMES))
        assert row.pts_sec == pytest.approx(1001.0)

    def test_shot_ids_restart_per_video(self, tmp_path):
        source = tmp_path / "kf"
        make_keyframes(source, "L01_V001", [0, 5])
        make_keyframes(source, "L01_V002", [0, 5])
        out = tmp_path / "keyframes.parquet"

        batch_builder.build_keyframe_manifest(str(source), str(out), fps="25")

        rows = list(iter_rows(str(out), IngestionEntity.FRAMES))
        by_video: dict[str, list[int]] = {}
        for row in rows:
            by_video.setdefault(row.video_id, []).append(row.shot_id)
        assert by_video == {"L01_V001": [0, 1], "L01_V002": [0, 1]}

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "kf").mkdir()

        with pytest.raises(ValueError, match="no keyframe files found"):
            batch_builder.build_keyframe_manifest(
                str(tmp_path / "kf"), str(tmp_path / "out.parquet"), fps="25"
            )


class TestKeyframeManifestWithShots:
    def test_keyframes_inherit_the_containing_shot(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [5, 15, 45])
        shots = make_shots_manifest(tmp_path / "shots.parquet", [(0, 9), (10, 39), (40, 99)])
        out = tmp_path / "keyframes.parquet"

        batch_builder.build_keyframe_manifest(
            str(source), str(out), fps="25", shots_manifest=shots
        )

        rows = list(iter_rows(str(out), IngestionEntity.FRAMES))
        assert [(row.original_frame_id, row.shot_id) for row in rows] == [
            (5, 0),
            (15, 1),
            (45, 2),
        ]

    def test_shot_boundaries_are_inclusive_at_both_ends(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [10, 39])
        shots = make_shots_manifest(tmp_path / "shots.parquet", [(10, 39)])
        out = tmp_path / "keyframes.parquet"

        batch_builder.build_keyframe_manifest(
            str(source), str(out), fps="25", shots_manifest=shots
        )

        rows = list(iter_rows(str(out), IngestionEntity.FRAMES))
        assert [row.shot_id for row in rows] == [0, 0]

    def test_keyframe_outside_every_shot_fails_loudly(self, tmp_path):
        source = make_keyframes(tmp_path / "kf", "L01_V001", [500])
        shots = make_shots_manifest(tmp_path / "shots.parquet", [(0, 99)])

        with pytest.raises(ValueError, match="fall outside every shot"):
            batch_builder.build_keyframe_manifest(
                str(source),
                str(tmp_path / "out.parquet"),
                fps="25",
                shots_manifest=shots,
            )


class TestShotCsvImport:
    def _write_csv(self, path: Path, text: str) -> str:
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_imports_boundaries_and_assigns_ids_in_start_order(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path / "shots.csv",
            "video_id,start_frame,end_frame\n"
            "L01_V001,50,99\n"
            "L01_V001,0,49\n",
        )
        out = tmp_path / "shots.parquet"

        count = batch_builder.build_shot_manifest(csv_path, str(out), fps="25")

        assert count == 2
        rows = list(iter_rows(str(out), IngestionEntity.CLIPS))
        assert [(row.shot_id, row.start_frame, row.end_frame) for row in rows] == [
            (0, 0, 49),
            (1, 50, 99),
        ]
        assert rows[1].start_sec == pytest.approx(2.0)

    def test_declared_shot_ids_are_honoured(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path / "shots.csv",
            "video_id,shot_id,start_frame,end_frame\nL01_V001,7,0,49\n",
        )
        out = tmp_path / "shots.parquet"

        batch_builder.build_shot_manifest(csv_path, str(out), fps="25")

        assert next(iter_rows(str(out), IngestionEntity.CLIPS)).shot_id == 7

    def test_source_video_path_comes_from_the_probe_manifest(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path / "shots.csv",
            "video_id,start_frame,end_frame\nL01_V001,0,49\n",
        )
        videos = make_videos_manifest(tmp_path / "videos.parquet")
        out = tmp_path / "shots.parquet"

        batch_builder.build_shot_manifest(csv_path, str(out), videos_manifest=videos)

        assert next(iter_rows(str(out), IngestionEntity.CLIPS)).path == (
            "/data/videos/L01_V001.mp4"
        )

    def test_missing_columns_are_reported(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path / "shots.csv", "video_id,start_frame\nL01_V001,0\n"
        )

        with pytest.raises(ValueError, match="missing columns: \\['end_frame'\\]"):
            batch_builder.build_shot_manifest(
                csv_path, str(tmp_path / "out.parquet"), fps="25"
            )

    def test_empty_csv_raises(self, tmp_path):
        csv_path = self._write_csv(
            tmp_path / "shots.csv", "video_id,start_frame,end_frame\n"
        )

        with pytest.raises(ValueError, match="no shots found"):
            batch_builder.build_shot_manifest(
                csv_path, str(tmp_path / "out.parquet"), fps="25"
            )


class TestShotIndex:
    def test_returns_none_for_unknown_video(self):
        index = batch_builder.ShotIndex([])

        assert index.find("L01_V001", 0) is None

    def test_returns_none_for_a_gap_between_shots(self):
        index = batch_builder.ShotIndex(
            [
                ClipManifestRow(
                    video_id="L01_V001",
                    shot_id=0,
                    start_frame=0,
                    end_frame=9,
                    start_sec=0.0,
                    end_sec=0.4,
                    path="/v.mp4",
                ),
                ClipManifestRow(
                    video_id="L01_V001",
                    shot_id=1,
                    start_frame=20,
                    end_frame=29,
                    start_sec=0.8,
                    end_sec=1.2,
                    path="/v.mp4",
                ),
            ]
        )

        assert index.find("L01_V001", 15) is None
        assert index.find("L01_V001", 9) == 0
        assert index.find("L01_V001", 20) == 1
