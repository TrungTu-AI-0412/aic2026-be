from fractions import Fraction

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from app.ingestion import manifest
from app.schemas.ingestions import IngestionEntity


def _keyframe(**overrides) -> manifest.KeyframeManifestRow:
    values = {
        "video_id": "L01_V001",
        "shot_id": 3,
        "keyframe_n": 12,
        "original_frame_id": 415,
        "pts_sec": 16.6,
        "path": "/data/keyframes/L01_V001/000415.jpg",
    }
    return manifest.KeyframeManifestRow(**{**values, **overrides})


def _clip(**overrides) -> manifest.ClipManifestRow:
    values = {
        "video_id": "L01_V001",
        "shot_id": 3,
        "start_frame": 400,
        "end_frame": 430,
        "start_sec": 16.0,
        "end_sec": 17.2,
        "path": "/data/videos/L01_V001.mp4",
    }
    return manifest.ClipManifestRow(**{**values, **overrides})


def _video(**overrides) -> manifest.VideoManifestRow:
    values = {
        "video_id": "L01_V001",
        "path": "/data/videos/L01_V001.mp4",
        "fps_num": 30000,
        "fps_den": 1001,
        "nb_frames": 54321,
        "duration_sec": 1812.9,
        "width": 1280,
        "height": 720,
        "rotation": 0,
        "is_vfr": False,
        "codec": "h264",
    }
    return manifest.VideoManifestRow(**{**values, **overrides})


class TestKeyframeRowMapping:
    def test_point_parts_use_keyframe_n(self):
        assert _keyframe().point_parts() == ("L01_V001", "kf12")

    def test_keyframes_sharing_a_frame_id_stay_distinct_points(self):
        """Two keyframes can round to the same `original_frame_id`.

        The organiser's `map-keyframes` does exactly this in 192 of 873
        videos. Identity must come from `keyframe_n`, or the second keyframe
        silently overwrites the first during upsert.
        """
        first = _keyframe(keyframe_n=1, original_frame_id=0, pts_sec=0.0)
        second = _keyframe(keyframe_n=2, original_frame_id=0, pts_sec=0.033)
        assert first.point_parts() != second.point_parts()

    def test_payload_carries_shot_id_for_scene_dedupe(self):
        payload = _keyframe().payload()
        assert payload == {
            "video_id": "L01_V001",
            "shot_id": 3,
            "keyframe_n": 12,
            "original_frame_id": 415,
            "pts_sec": 16.6,
            "shot_start_sec": 0.0,
            "shot_end_sec": 0.0,
            "path": "/data/keyframes/L01_V001/000415.jpg",
        }

    def test_shot_range_survives_a_zero_start(self):
        """The first shot of a video legitimately starts at 0.0 seconds.

        The range is written unconditionally rather than dropped when empty:
        the ASR overlap bonus reads it, and a missing `shot_start_sec` would be
        indistinguishable from a shot that genuinely begins at zero.
        """
        payload = _keyframe(shot_start_sec=0.0, shot_end_sec=4.5).payload()

        assert payload["shot_start_sec"] == 0.0
        assert payload["shot_end_sec"] == 4.5

    def test_negative_original_frame_id_is_rejected(self):
        with pytest.raises(ValidationError):
            _keyframe(original_frame_id=-1)

    def test_keyframe_n_must_be_positive(self):
        with pytest.raises(ValidationError):
            _keyframe(keyframe_n=0)


class TestEnrichmentPayload:
    def test_enrichment_reaches_the_payload(self):
        payload = _keyframe(
            objects=["Person", "Boat"],
            ocr_text="TẠM DỪNG LƯU THÔNG",
            title="Bản tin 60 giây",
        ).payload()

        assert payload["objects"] == ["Person", "Boat"]
        assert payload["ocr_text"] == "TẠM DỪNG LƯU THÔNG"
        assert payload["title"] == "Bản tin 60 giây"

    def test_frames_no_longer_carry_asr_text(self):
        """Speech lives in its own collection, keyed by time, not by frame.

        Pooling a shot's transcript onto every keyframe as well would let the
        ASR-overlap bonus and a lexical frame match score the same speech twice.
        """
        row = _keyframe()

        assert not hasattr(row, "asr_text")
        assert not hasattr(row, "asr_entities")
        assert "asr_text" not in row.payload()

    def test_publish_date_and_keywords_are_gone(self):
        """Neither survived: one sorts wrong as a string, the other only ever
        padded the frame speech vector that no longer exists."""
        row = _keyframe()

        assert not hasattr(row, "publish_date")
        assert not hasattr(row, "keywords")

    def test_absent_enrichment_leaves_the_payload_minimal(self):
        """Empty values are dropped rather than indexed as matchless terms."""
        payload = _keyframe().payload()

        assert set(payload) == {
            "video_id",
            "shot_id",
            "keyframe_n",
            "original_frame_id",
            "pts_sec",
            "shot_start_sec",
            "shot_end_sec",
            "path",
        }

    def test_clips_carry_enrichment_too(self):
        payload = _clip(ocr_text="bản tin", objects=["Person"]).payload()

        assert payload["ocr_text"] == "bản tin"
        assert payload["objects"] == ["Person"]

    def test_a_manifest_without_enrichment_columns_still_validates(self):
        """Manifests written before enrichment existed must keep working."""
        row = manifest.KeyframeManifestRow(
            video_id="L01_V001",
            shot_id=1,
            keyframe_n=1,
            original_frame_id=0,
            pts_sec=0.0,
            path="/data/k.jpg",
        )

        assert row.objects == []
        assert row.ocr_text == ""


class TestClipRowMapping:
    def test_point_parts_are_namespaced_by_shot(self):
        assert _clip().point_parts() == ("L01_V001", "shot3")

    def test_payload_carries_inclusive_frame_range(self):
        payload = _clip().payload()
        assert payload["start_frame"] == 400
        assert payload["end_frame"] == 430
        assert payload["path"] == "/data/videos/L01_V001.mp4"

    def test_single_frame_shot_is_valid_because_end_is_inclusive(self):
        row = _clip(start_frame=400, end_frame=400, start_sec=16.0, end_sec=16.0)
        assert row.start_frame == row.end_frame

    def test_reversed_frame_range_is_rejected(self):
        with pytest.raises(ValidationError, match="precedes start_frame"):
            _clip(start_frame=430, end_frame=400)

    def test_reversed_time_range_is_rejected(self):
        with pytest.raises(ValidationError, match="precedes start_sec"):
            _clip(start_sec=17.2, end_sec=16.0)


class TestVideoRowFrameRateMapping:
    def test_fps_stays_an_exact_fraction(self):
        assert _video().fps == Fraction(30000, 1001)

    def test_frame_to_sec_and_back_is_lossless_on_ntsc_rates(self):
        video = _video()
        # 29.97 rounded to float drifts by whole frames over a long video;
        # the fractional rate must survive the round trip exactly.
        for frame_id in (0, 1, 1500, 54320):
            assert video.sec_to_frame(video.frame_to_sec(frame_id)) == frame_id

    def test_integer_frame_rate_round_trip(self):
        video = _video(fps_num=25, fps_den=1)
        assert video.frame_to_sec(50) == pytest.approx(2.0)
        assert video.sec_to_frame(2.0) == 50

    def test_nb_frames_is_optional_because_containers_may_omit_it(self):
        assert _video(nb_frames=None).nb_frames is None

    def test_unsupported_rotation_is_rejected(self):
        with pytest.raises(ValidationError):
            _video(rotation=45)

    def test_zero_frame_rate_denominator_is_rejected(self):
        with pytest.raises(ValidationError):
            _video(fps_den=0)


class TestParquetRoundTrip:
    def test_keyframe_manifest_round_trip(self, tmp_path):
        out = tmp_path / "keyframes.parquet"
        rows = [_keyframe(original_frame_id=i, shot_id=i // 2) for i in range(5)]

        written = manifest.write_rows(
            rows, str(out), manifest.KEYFRAME_ARROW_SCHEMA
        )
        assert written == 5

        manifest.validate_columns(str(out), IngestionEntity.FRAMES)
        assert manifest.count_rows(str(out)) == 5

        read_back = list(manifest.iter_rows(str(out), IngestionEntity.FRAMES))
        assert all(isinstance(row, manifest.KeyframeManifestRow) for row in read_back)
        assert [row.original_frame_id for row in read_back] == [0, 1, 2, 3, 4]

    def test_clip_manifest_round_trip(self, tmp_path):
        out = tmp_path / "shots.parquet"
        rows = [_clip(shot_id=i, start_frame=i * 30, end_frame=i * 30 + 29) for i in range(4)]

        manifest.write_rows(rows, str(out), manifest.CLIP_ARROW_SCHEMA)
        manifest.validate_columns(str(out), IngestionEntity.CLIPS)

        read_back = list(manifest.iter_rows(str(out), IngestionEntity.CLIPS))
        assert all(isinstance(row, manifest.ClipManifestRow) for row in read_back)
        assert [row.shot_id for row in read_back] == [0, 1, 2, 3]

    def test_video_manifest_round_trip(self, tmp_path):
        out = tmp_path / "videos.parquet"
        manifest.write_rows([_video()], str(out), manifest.VIDEO_ARROW_SCHEMA)

        manifest.validate_video_columns(str(out))
        read_back = list(manifest.iter_video_rows(str(out)))
        assert read_back[0].fps == Fraction(30000, 1001)
        assert read_back[0].codec == "h264"

    def test_all_null_nb_frames_keeps_its_declared_type(self, tmp_path):
        out = tmp_path / "videos.parquet"
        manifest.write_rows(
            [_video(nb_frames=None)], str(out), manifest.VIDEO_ARROW_SCHEMA
        )

        # Without an explicit schema pyarrow would infer `null` here and the
        # column would no longer round-trip as an integer.
        schema = pq.ParquetFile(str(out)).schema_arrow
        assert schema.field("nb_frames").type == manifest.VIDEO_ARROW_SCHEMA.field(
            "nb_frames"
        ).type
        assert list(manifest.iter_video_rows(str(out)))[0].nb_frames is None


class TestColumnValidation:
    def test_missing_columns_are_reported(self, tmp_path):
        out = tmp_path / "keyframes.parquet"
        manifest.write_rows([_keyframe()], str(out), manifest.KEYFRAME_ARROW_SCHEMA)

        with pytest.raises(ValueError, match="missing required columns"):
            manifest.validate_columns(str(out), IngestionEntity.CLIPS)

    def test_entities_declare_distinct_required_columns(self):
        frames = manifest.REQUIRED_COLUMNS[IngestionEntity.FRAMES]
        clips = manifest.REQUIRED_COLUMNS[IngestionEntity.CLIPS]

        assert "original_frame_id" in frames
        assert "original_frame_id" not in clips
        assert {"video_id", "shot_id", "path"} <= frames & clips


class TestArrowMapColumns:
    BASE = {
        "video_id": "L21_V001",
        "shot_id": 0,
        "keyframe_n": 1,
        "original_frame_id": 0,
        "pts_sec": 0.0,
        "path": "keyframes/L21_V001/001.jpg",
    }

    def test_object_counts_accepts_arrows_pair_list(self) -> None:
        """`RecordBatch.to_pylist()` renders `map<string,int32>` as pairs.

        It does not rebuild a dict, so every manifest carrying object_counts
        failed validation on its first row and ingestion never started.
        """
        row = manifest.KeyframeManifestRow.model_validate(
            {**self.BASE, "object_counts": [("Person", 2), ("Boat", 1)]}
        )

        assert row.object_counts == {"Person": 2, "Boat": 1}

    def test_a_plain_dict_still_validates(self) -> None:
        row = manifest.KeyframeManifestRow.model_validate(
            {**self.BASE, "object_counts": {"Person": 2}}
        )

        assert row.object_counts == {"Person": 2}


class TestOcrEnrichment:
    BASE = {
        "video_id": "L21_V001",
        "shot_id": 0,
        "keyframe_n": 1,
        "original_frame_id": 0,
        "pts_sec": 0.0,
        "path": "keyframes/L21_V001/001.jpg",
    }

    def test_ocr_text_is_kept_verbatim(self) -> None:
        """Undiacriticked all-caps ticker text must not be normalised here.

        `app.features.sparse` folds diacritics at encode time, so the raw form
        still answers a query typed with them. Normalising at ingest would
        discard the original with nothing gained.
        """
        row = manifest.KeyframeManifestRow.model_validate(
            {**self.BASE, "ocr_text": "Tam DUnG LuU Thong", "ocr_regions": 4}
        )

        assert row.ocr_text == "Tam DUnG LuU Thong"
        assert row.enrichment_payload()["ocr_text"] == "Tam DUnG LuU Thong"

    def test_a_shot_with_no_on_screen_text_writes_no_ocr_payload(self) -> None:
        row = manifest.KeyframeManifestRow.model_validate(self.BASE)

        assert "ocr_text" not in row.enrichment_payload()

    def test_manifests_predating_ocr_still_validate(self) -> None:
        assert manifest.KeyframeManifestRow.model_validate(self.BASE).ocr_text == ""


class TestAppendAcrossSchemaChange:
    def test_resuming_into_an_older_manifest_is_refused(self, tmp_path):
        """Silent null back-fill is the expensive failure mode.

        `read_table(schema=...)` happily invents a column the old file lacks and
        fills it with nulls. That passes `validate_columns` -- the column now
        exists -- and only fails when a row is parsed, i.e. after a multi-hour
        decode pass has already been paid for. Observed producing 5 314
        null-`keyframe_n` rows against a real stale manifest.
        """
        out = tmp_path / "frames.parquet"
        old_schema = pa.schema(
            [
                ("video_id", pa.string()),
                ("shot_id", pa.int32()),
                ("original_frame_id", pa.int64()),
                ("pts_sec", pa.float64()),
                ("path", pa.string()),
            ]
        )
        pq.write_table(
            pa.table(
                {
                    "video_id": ["L01_V001"],
                    "shot_id": [0],
                    "original_frame_id": [0],
                    "pts_sec": [0.0],
                    "path": ["k.jpg"],
                },
                schema=old_schema,
            ),
            out,
        )

        with pytest.raises(ValueError, match="predates this schema"):
            manifest.write_rows(
                [_keyframe()], str(out), manifest.KEYFRAME_ARROW_SCHEMA, append=True
            )

    def test_appending_to_a_matching_manifest_still_works(self, tmp_path):
        out = tmp_path / "frames.parquet"
        manifest.write_rows(
            [_keyframe(keyframe_n=1)], str(out), manifest.KEYFRAME_ARROW_SCHEMA
        )

        total = manifest.write_rows(
            [_keyframe(keyframe_n=2)],
            str(out),
            manifest.KEYFRAME_ARROW_SCHEMA,
            append=True,
        )

        assert total == 2
