"""Parsing rules for the enriched transcript CSVs.

Every case here is a real property of the source measured over all 40k
segments, not a hypothetical: see docs/asr-transcripts.md.
"""

import csv

import pytest

from scripts.build_asr_manifest import (
    DEFAULT_MIN_WORDS,
    build,
    clean,
    parse_entities,
    rows_for_csv,
)

COLUMNS = [
    "segment", "start", "end", "duration", "text", "speech_score", "has_speech",
    "video_id", "title", "author", "channel_id", "watch_url", "text_corrected",
    "entities",
]


def write_csv(path, rows):
    target = path / "L21_V001_segments_enriched.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    return target


def row(**overrides):
    base = {
        "segment": 1, "start": 0, "end": 3, "duration": 2.7,
        "text": "xin chao", "text_corrected": "Xin chào.", "speech_score": 0.97,
        "has_speech": True, "video_id": "L21_V001", "title": "Bản tin",
        "author": "60 Giây", "channel_id": "UC123",
        "watch_url": "https://youtube.com/watch?v=x", "entities": "{}",
    }
    base.update(overrides)
    return base


class TestClean:
    @pytest.mark.parametrize("value", ["nan", "NaN", "NAN", "none", "", "  ", None])
    def test_null_spellings_become_empty(self, value):
        """`text_corrected` is empty as the literal string "nan", not a blank
        cell. Writing it through would index "nan" across thousands of rows."""
        assert clean(value) == ""

    def test_real_text_survives_with_whitespace_trimmed(self):
        assert clean("  Xin chào.  ") == "Xin chào."


class TestParseEntities:
    def test_a_python_dict_repr_is_split_by_type(self):
        """The column is a Python repr with single quotes, not JSON."""
        parsed = parse_entities(
            "{'persons': ['Xuân Sơn'], 'orgs': ['HTV'], "
            "'locations': ['Hà Nội'], 'others': ['abc']}"
        )

        assert parsed["asr_persons"] == ["Xuân Sơn"]
        assert parsed["asr_orgs"] == ["HTV"]
        assert parsed["asr_locations"] == ["Hà Nội"]

    def test_others_is_discarded(self):
        """900 segments and no defined meaning, so nothing could query it."""
        parsed = parse_entities("{'others': ['abc'], 'persons': []}")

        assert "others" not in parsed
        assert not any(parsed.values())

    def test_repeated_mentions_are_deduplicated(self):
        parsed = parse_entities("{'persons': ['Xuân Sơn', 'Xuân Sơn']}")

        assert parsed["asr_persons"] == ["Xuân Sơn"]

    def test_a_malformed_cell_yields_no_entities_rather_than_raising(self):
        """Entities narrow 22% of segments; losing them for one row is far
        cheaper than losing that row's transcript."""
        parsed = parse_entities("{'persons': [unclosed")

        assert parsed == {"asr_persons": [], "asr_orgs": [], "asr_locations": []}

    @pytest.mark.parametrize("value", ["", "nan", "[]", "42"])
    def test_non_dict_values_yield_no_entities(self, value):
        assert parse_entities(value) == {
            "asr_persons": [], "asr_orgs": [], "asr_locations": []
        }


class TestRowsForCsv:
    def test_segments_without_a_corrected_transcript_are_skipped(self, tmp_path):
        path = write_csv(tmp_path, [
            row(segment=1, text_corrected="Xin chào."),
            row(segment=2, text_corrected="nan", text="music only"),
        ])

        rows = rows_for_csv(path)

        assert [item.segment for item in rows] == [1]

    def test_presence_of_text_decides_not_the_has_speech_flag(self, tmp_path):
        """162 segments have a transcript with has_speech False, and 216 have
        the flag True and no text, so the flag is unusable in both directions."""
        path = write_csv(tmp_path, [
            row(segment=1, has_speech=False, text_corrected="Có tiếng nói."),
            row(segment=2, has_speech=True, text_corrected="nan"),
        ])

        rows = rows_for_csv(path)

        assert [item.segment for item in rows] == [1]

    def test_single_word_filler_segments_are_dropped(self, tmp_path):
        """801 segments in this corpus are one word -- "Ừ", "À", "thì", "Ờ".

        They are worse than useless: a one-word transcript still yields a dense
        vector, and because scores are normalised with the best hit at 1.0, such
        a segment can outrank real speech and hand a frame the full overlap
        bonus for saying nothing. Observed doing exactly that before this filter.
        """
        path = write_csv(tmp_path, [
            row(segment=1, text_corrected="Ờ"),
            row(segment=2, text_corrected="thì"),
            row(segment=3, text_corrected="Tại Thành phố Cần Thơ xảy ra sạt lở."),
        ])

        assert [item.segment for item in rows_for_csv(path)] == [3]

    def test_two_word_segments_survive(self, tmp_path):
        """The threshold stops at one word on purpose: a two-word segment can be
        a person's name, which is exactly what a competition query hangs on."""
        path = write_csv(tmp_path, [row(text_corrected="Xuân Sơn")])

        assert len(rows_for_csv(path)) == 1

    def test_the_threshold_is_adjustable(self, tmp_path):
        path = write_csv(tmp_path, [row(text_corrected="Xuân Sơn")])

        assert rows_for_csv(path, min_words=3) == []
        assert len(rows_for_csv(path, min_words=2)) == 1

    def test_the_default_threshold_keeps_two_word_content(self):
        assert DEFAULT_MIN_WORDS == 2

    def test_timing_and_metadata_reach_the_row(self, tmp_path):
        path = write_csv(tmp_path, [row(start=10, end=14, duration=3.6)])

        item = rows_for_csv(path)[0]

        assert (item.start_sec, item.end_sec) == (10.0, 14.0)
        assert item.duration == 3.6
        assert item.title == "Bản tin"
        assert item.channel_id == "UC123"

    def test_inverted_bounds_are_repaired(self, tmp_path):
        """start/end are rounded to whole seconds, so both can round the wrong
        way and make a segment appear to end before it starts."""
        path = write_csv(tmp_path, [row(start=9, end=8)])

        item = rows_for_csv(path)[0]

        assert item.start_sec <= item.end_sec

    def test_the_raw_text_column_is_not_carried(self, tmp_path):
        path = write_csv(tmp_path, [row(text="raw mishearing")])

        item = rows_for_csv(path)[0]

        assert not hasattr(item, "text")
        assert item.text_corrected == "Xin chào."

    def test_point_id_is_stable_and_distinct_per_segment(self, tmp_path):
        path = write_csv(tmp_path, [row(segment=1), row(segment=2)])

        first, second = rows_for_csv(path)

        assert first.point_parts() == ("L21_V001", "seg1")
        assert first.point_parts() != second.point_parts()

    def test_entity_terms_pool_all_three_types(self, tmp_path):
        path = write_csv(tmp_path, [
            row(entities="{'persons': ['A'], 'orgs': ['B'], 'locations': ['C']}")
        ])

        assert rows_for_csv(path)[0].entity_terms() == ["A", "B", "C"]


class TestBuild:
    def test_a_manifest_is_written_and_counted(self, tmp_path):
        source = tmp_path / "t"
        source.mkdir()
        write_csv(source, [row(segment=1), row(segment=2, text_corrected="nan")])
        out = tmp_path / "asr.parquet"

        assert build(str(source), str(out)) == 1
        assert out.is_file()

    def test_a_corpus_with_no_transcripts_at_all_fails_loudly(self, tmp_path):
        source = tmp_path / "t"
        source.mkdir()
        write_csv(source, [row(text_corrected="nan")])

        with pytest.raises(SystemExit):
            build(str(source), str(tmp_path / "asr.parquet"))

    def test_a_missing_directory_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit):
            build(str(tmp_path / "nope"), str(tmp_path / "asr.parquet"))
