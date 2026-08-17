"""Tests for the organiser-artefact manifest builder.

Run from the repository root; `scripts` is not importable from `backend/`.
"""

import json
from pathlib import Path

from scripts.build_frames_manifest import (
    assign_shot,
    entities_between,
    frame_upper_bound,
    load_asr_csv,
    load_keyframes,
    load_objects,
    load_transcript_spans,
    text_between,
)

SHOTS = [(0, 99), (101, 199), (201, 299)]
STARTS = [start for start, _ in SHOTS]


class TestAssignShot:
    def test_frame_inside_a_shot(self) -> None:
        assert assign_shot(STARTS, SHOTS, 50) == 0
        assert assign_shot(STARTS, SHOTS, 150) == 1
        assert assign_shot(STARTS, SHOTS, 299) == 2

    def test_frame_in_the_gap_joins_the_following_shot(self) -> None:
        """Detectors leave the cut frame itself unassigned."""
        assert assign_shot(STARTS, SHOTS, 100) == 1
        assert assign_shot(STARTS, SHOTS, 200) == 2

    def test_frame_before_the_first_shot_joins_it(self) -> None:
        """Some videos start their first shot at frame 1, orphaning frame 0."""
        shots = [(1, 99), (100, 199)]
        assert assign_shot([1, 100], shots, 0) == 0

    def test_frame_past_the_last_shot_joins_it(self) -> None:
        assert assign_shot(STARTS, SHOTS, 5000) == 2


class TestLoadKeyframes:
    def test_n_and_frame_idx_stay_separate(self, tmp_path: Path) -> None:
        """The filename ordinal must never be read as the frame index."""
        csv_path = tmp_path / "L21_V001.csv"
        csv_path.write_text(
            "n,pts_time,fps,frame_idx\n"
            "1,0.0,25.0,0\n"
            "2,5.4,25.0,135\n"
            "3,10.8,25.0,270\n",
            encoding="utf-8",
        )

        rows, fps = load_keyframes(csv_path)

        assert fps == 25.0
        assert rows == [(1, 0, 0.0), (2, 135, 5.4), (3, 270, 10.8)]

    def test_duplicate_frame_idx_is_preserved(self, tmp_path: Path) -> None:
        """Two keyframes may round to the same frame index; keep both rows."""
        csv_path = tmp_path / "L21_V006.csv"
        csv_path.write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,0.0333333,30.0,0\n",
            encoding="utf-8",
        )

        rows, _ = load_keyframes(csv_path)

        assert [n for n, _, _ in rows] == [1, 2]
        assert [frame for _, frame, _ in rows] == [0, 0]


class TestLoadObjects:
    def _write(self, path: Path, pairs: list[tuple[str, float]]) -> None:
        path.write_text(
            json.dumps(
                {
                    "detection_class_entities": [name for name, _ in pairs],
                    "detection_scores": [str(score) for _, score in pairs],
                }
            ),
            encoding="utf-8",
        )

    def test_counts_repeated_entities_above_the_threshold(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "001.json"
        self._write(path, [("Person", 0.9), ("Person", 0.7), ("Boat", 0.5)])

        names, counts = load_objects(path, 0.3)

        assert names == ["Boat", "Person"]
        assert counts == {"Person": 2, "Boat": 1}

    def test_stops_at_the_first_score_below_the_threshold(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "001.json"
        self._write(path, [("Person", 0.9), ("Boat", 0.1), ("Car", 0.8)])

        names, _ = load_objects(path, 0.3)

        assert names == ["Person"]

    def test_frame_with_no_confident_detection(self, tmp_path: Path) -> None:
        path = tmp_path / "001.json"
        self._write(path, [("Person", 0.15)])

        assert load_objects(path, 0.3) == ([], {})


class TestTranscriptSpans:
    def test_overlapping_windows_are_trimmed_to_the_next_start(
        self, tmp_path: Path
    ) -> None:
        """YouTube caption windows overlap; the next start is the real end."""
        path = tmp_path / "L21_V001.json"
        path.write_text(
            json.dumps(
                {
                    "segments": [
                        {"start": 0.0, "duration": 5.0, "text": "một"},
                        {"start": 2.0, "duration": 5.0, "text": "hai"},
                        {"start": 4.0, "duration": 3.0, "text": "ba"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        spans = load_transcript_spans(path)

        assert spans == [(0.0, 2.0, "một"), (2.0, 4.0, "hai"), (4.0, 7.0, "ba")]

    def test_text_is_not_duplicated_across_a_window(self, tmp_path: Path) -> None:
        path = tmp_path / "L21_V001.json"
        path.write_text(
            json.dumps(
                {
                    "segments": [
                        {"start": 0.0, "duration": 5.0, "text": "một"},
                        {"start": 2.0, "duration": 5.0, "text": "hai"},
                        {"start": 4.0, "duration": 3.0, "text": "ba"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        spans = load_transcript_spans(path)

        assert text_between(spans, 0.0, 7.0) == "một hai ba"
        assert text_between(spans, 2.5, 3.5) == "hai"
        assert text_between(spans, 100.0, 200.0) == ""

    def test_no_segments_yields_no_text(self) -> None:
        assert text_between([], 0.0, 10.0) == ""


class TestLoadAsrCsv:
    HEADER = "segment,start,end,duration,text,has_speech,text_corrected,entities\n"

    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "L21_V001_segments_enriched.csv"
        path.write_text(self.HEADER + body, encoding="utf-8")
        return path

    def test_keeps_both_raw_and_corrected_text(self, tmp_path: Path) -> None:
        """Corrected text is fluent but does not fix misheard words."""
        path = self._write(
            tmp_path,
            '1,0,5,5,"sục lúng",True,"Sục lún.","{\'locations\': []}"\n',
        )

        rows = load_asr_csv(path)

        assert len(rows) == 1
        start, end, text, corrected, _ = rows[0]
        assert (start, end) == (0.0, 5.0)
        assert text == "sục lúng"
        assert corrected == "Sục lún."

    def test_drops_segments_with_no_text(self, tmp_path: Path) -> None:
        """Music stings carry no words and would only pad the index."""
        path = self._write(
            tmp_path,
            '1,0,3,3,"",False,"","{}"\n2,5,9,4,"xin chào",True,"Xin chào.","{}"\n',
        )

        assert [r[2] for r in load_asr_csv(path)] == ["xin chào"]

    def test_keeps_text_even_when_has_speech_is_false(self, tmp_path: Path) -> None:
        """162 real segments carry a transcript while the VAD flag reads False."""
        path = self._write(
            tmp_path,
            '1,0,5,5,"có tiếng nói",False,"Có tiếng nói.","{}"\n',
        )

        assert [r[2] for r in load_asr_csv(path)] == ["có tiếng nói"]

    def test_flattens_entities_across_categories(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            "1,0,5,5,\"a\",True,\"A.\","
            "\"{'persons': ['Nguyễn Văn A'], 'orgs': ['HTV'], "
            "'locations': ['Hà Nội'], 'others': []}\"\n",
        )

        assert load_asr_csv(path)[0][4] == ["Nguyễn Văn A", "HTV", "Hà Nội"]

    def test_malformed_entities_do_not_break_the_row(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, '1,0,5,5,"a",True,"A.","{not python}"\n')

        rows = load_asr_csv(path)

        assert len(rows) == 1
        assert rows[0][4] == []


class TestEntitiesBetween:
    ASR = [
        (0.0, 5.0, "a", "A.", ["Hà Nội"]),
        (5.0, 10.0, "b", "B.", ["HTV", "Hà Nội"]),
        (10.0, 15.0, "c", "C.", ["Huế"]),
    ]

    def test_collects_overlapping_segments_without_duplicates(self) -> None:
        assert entities_between(self.ASR, 0.0, 10.0) == ["Hà Nội", "HTV"]

    def test_window_outside_every_segment_is_empty(self) -> None:
        assert entities_between(self.ASR, 100.0, 200.0) == []

    def test_touching_boundary_does_not_count(self) -> None:
        """A shot starting exactly where a segment ends does not inherit it."""
        assert entities_between(self.ASR, 5.0, 6.0) == ["HTV", "Hà Nội"]


class TestFrameUpperBound:
    def test_uses_the_declared_length_when_it_is_the_larger(self) -> None:
        assert frame_upper_bound(1163, 30.0, 34000) == 34890

    def test_a_keyframe_past_the_declared_length_raises_the_bound(self) -> None:
        """media-info reports whole seconds, so length * fps lands short.

        43 of the 873 videos here have their last keyframe beyond that
        product. Bounding on it alone would reject frames the organiser
        themselves sampled.
        """
        assert frame_upper_bound(1163, 30.0, 34895) == 34896

    def test_missing_length_falls_back_to_what_was_observed(self) -> None:
        assert frame_upper_bound(0, 30.0, 500) == 501
