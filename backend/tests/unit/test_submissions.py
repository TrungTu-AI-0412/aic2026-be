import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from app.schemas.submissions import ExportFormat, ExportRequest
from app.services.submissions import FrameOutOfBoundsError, VideoNotFoundError
from app.submissions import formats
from app.submissions.service import LocalSubmissionService, load_bounds


def kis_request(**overrides) -> ExportRequest:
    payload = {
        "task": "kis",
        "candidates": [
            {"task": "kis", "video_id": "L22_V001", "frame_id": 1200},
            {"task": "kis", "video_id": "L26_V190", "frame_id": 40},
        ],
    }
    payload.update(overrides)
    return ExportRequest.model_validate(payload)


def qa_request(answer: str = "Hà Nội") -> ExportRequest:
    return ExportRequest.model_validate(
        {
            "task": "qa",
            "candidates": [
                {
                    "task": "qa",
                    "video_id": "L22_V001",
                    "frame_id": 1200,
                    "answer": answer,
                }
            ],
        }
    )


def trake_request(*frames: int) -> ExportRequest:
    return ExportRequest.model_validate(
        {
            "task": "trake",
            "event_slot_count": len(frames),
            "candidates": [
                {
                    "task": "trake",
                    "video_id": "L22_V001",
                    "event_frame_ids": list(frames),
                }
            ],
        }
    )


class TestCsv:
    def test_has_no_header_row(self) -> None:
        """A grader that counts lines would score the header as answer one."""
        first = formats.to_csv(kis_request()).decode().splitlines()[0]

        assert first == "L22_V001,1200"

    def test_kis_row_shape(self) -> None:
        assert formats.to_csv(kis_request()).decode() == (
            "L22_V001,1200\nL26_V190,40\n"
        )

    def test_qa_appends_the_answer(self) -> None:
        assert formats.to_csv(qa_request()).decode() == "L22_V001,1200,Hà Nội\n"

    def test_trake_spreads_events_across_columns(self) -> None:
        assert formats.to_csv(trake_request(10, 20, 30)).decode() == (
            "L22_V001,10,20,30\n"
        )

    def test_answer_containing_a_comma_is_quoted(self) -> None:
        """Otherwise the answer splits into two fields and shifts the row."""
        rendered = formats.to_csv(qa_request("Hà Nội, Việt Nam")).decode()

        assert rendered == 'L22_V001,1200,"Hà Nội, Việt Nam"\n'

    def test_carries_no_byte_order_mark(self) -> None:
        """A BOM rides on the first field, corrupting the best answer's id."""
        assert not formats.to_csv(kis_request()).startswith(b"\xef\xbb\xbf")

    def test_lines_end_with_lf_only(self) -> None:
        assert b"\r" not in formats.to_csv(kis_request())

    def test_diacritics_survive_as_utf8(self) -> None:
        assert "Hà Nội" in formats.to_csv(qa_request()).decode("utf-8")


class TestJson:
    def test_carries_the_same_fields_as_the_csv(self) -> None:
        request = trake_request(10, 20, 30)
        rows = json.loads(formats.to_json(request))

        assert [r["fields"] for r in rows] == [["L22_V001", "10", "20", "30"]]

    def test_keeps_diacritics_unescaped(self) -> None:
        assert "Hà Nội" in formats.to_json(qa_request()).decode("utf-8")


class TestRender:
    def test_csv_is_the_default_format(self) -> None:
        _, media_type, filename = formats.render(kis_request())

        assert media_type == "text/csv"
        assert filename == "submission-kis.csv"

    def test_json_format_switches_both_type_and_extension(self) -> None:
        _, media_type, filename = formats.render(
            kis_request(format=ExportFormat.json)
        )

        assert media_type == "application/json"
        assert filename == "submission-kis.json"


def write_bounds(tmp_path, rows: dict[str, int]):
    path = tmp_path / "video_bounds.parquet"
    pq.write_table(
        pa.table(
            {
                "video_id": list(rows),
                "fps": [30.0] * len(rows),
                "length_sec": [0] * len(rows),
                "frame_upper_bound": list(rows.values()),
            }
        ),
        path,
    )
    return path


class TestLoadBounds:
    def test_missing_manifest_is_an_empty_map_not_an_error(self, tmp_path) -> None:
        load_bounds.cache_clear()

        assert load_bounds(str(tmp_path / "absent.parquet")) == {}

    def test_reads_the_bound_per_video(self, tmp_path) -> None:
        load_bounds.cache_clear()
        path = write_bounds(tmp_path, {"L22_V001": 34890})

        assert load_bounds(str(path)) == {"L22_V001": 34890}


class TestExport:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        load_bounds.cache_clear()
        yield
        load_bounds.cache_clear()

    async def test_exports_when_every_frame_is_in_range(self, tmp_path) -> None:
        path = write_bounds(tmp_path, {"L22_V001": 34890, "L26_V190": 1000})
        service = LocalSubmissionService(bounds_manifest=str(path))

        result = await service.export(kis_request())

        assert result.content.decode() == "L22_V001,1200\nL26_V190,40\n"
        assert result.filename == "submission-kis.csv"

    async def test_unknown_video_is_rejected(self, tmp_path) -> None:
        path = write_bounds(tmp_path, {"L22_V001": 34890})
        service = LocalSubmissionService(bounds_manifest=str(path))

        with pytest.raises(VideoNotFoundError, match="L26_V190"):
            await service.export(kis_request())

    async def test_frame_past_the_end_is_rejected(self, tmp_path) -> None:
        path = write_bounds(tmp_path, {"L22_V001": 1200, "L26_V190": 1000})
        service = LocalSubmissionService(bounds_manifest=str(path))

        with pytest.raises(FrameOutOfBoundsError, match="frame 1200"):
            await service.export(kis_request())

    async def test_the_last_valid_frame_is_accepted(self, tmp_path) -> None:
        """The bound is exclusive; off-by-one here silently drops answers."""
        path = write_bounds(tmp_path, {"L22_V001": 1201, "L26_V190": 1000})
        service = LocalSubmissionService(bounds_manifest=str(path))

        assert await service.export(kis_request())

    async def test_every_trake_event_is_checked(self, tmp_path) -> None:
        path = write_bounds(tmp_path, {"L22_V001": 100})
        service = LocalSubmissionService(bounds_manifest=str(path))

        with pytest.raises(FrameOutOfBoundsError, match="frame 500"):
            await service.export(trake_request(10, 20, 500))

    async def test_the_rejected_row_is_named_by_position(self, tmp_path) -> None:
        """With up to 100 rows, "some frame is wrong" is not actionable."""
        path = write_bounds(tmp_path, {"L22_V001": 34890, "L26_V190": 20})
        service = LocalSubmissionService(bounds_manifest=str(path))

        with pytest.raises(FrameOutOfBoundsError, match="candidate 2"):
            await service.export(kis_request())

    async def test_absent_manifest_skips_the_check_instead_of_failing(
        self, tmp_path
    ) -> None:
        """No manifest means bounds are unknown, not that no video exists."""
        service = LocalSubmissionService(
            bounds_manifest=str(tmp_path / "absent.parquet")
        )

        result = await service.export(kis_request())

        assert result.content.decode() == "L22_V001,1200\nL26_V190,40\n"


class TestQaAnswerNormalisation:
    """The answer is typed by a person, so it arrives as a paste does."""

    def test_padding_and_newlines_collapse_to_one_line(self) -> None:
        rendered = formats.to_csv(qa_request("  Hà Nội\n hai  triệu \n")).decode()

        assert rendered == "L22_V001,1200,Hà Nội hai triệu\n"

    def test_a_blank_answer_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            qa_request("   \n ")
