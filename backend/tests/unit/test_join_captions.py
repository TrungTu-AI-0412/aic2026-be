"""Tests for pulling on-screen text back out of VLM captions.

Run from the repository root; `scripts` is not importable from `backend/`.
"""

from scripts.join_captions import quoted_spans, read_captions


class TestQuotedSpans:
    def test_pulls_the_text_the_model_transcribed(self) -> None:
        caption = (
            'Bức ảnh chụp một biển báo với dòng chữ "CẢNH BÁO SẠT LỞ NGUY HIỂM" '
            'kèm theo "TẠM DỪNG LƯU THÔNG".'
        )

        assert quoted_spans(caption) == [
            "CẢNH BÁO SẠT LỞ NGUY HIỂM",
            "TẠM DỪNG LƯU THÔNG",
        ]

    def test_curly_quotes_count_too(self) -> None:
        assert quoted_spans("logo “HTV9” ở góc phải") == ["HTV9"]

    def test_a_caption_with_no_quotes_yields_nothing(self) -> None:
        assert quoted_spans("Cảnh quay toàn cảnh thành phố lúc hoàng hôn.") == []

    def test_the_models_hedging_is_dropped(self) -> None:
        """It hedges inside the quotes, so the hedge would be indexed as text.

        Every unreadable sign in the corpus would then answer the same query.
        """
        caption = 'Có dòng chữ "(không rõ nội dung cụ thể)" và "Sơn La: Bản bị ngập"'

        assert quoted_spans(caption) == ["Sơn La: Bản bị ngập"]

    def test_repeated_text_is_kept_once(self) -> None:
        """The ticker scrolls, so the model often quotes one line twice."""
        caption = 'thấy "HTV9" ở trên và "HTV9" lặp lại phía dưới'

        assert quoted_spans(caption) == ["HTV9"]

    def test_single_characters_are_not_text(self) -> None:
        assert quoted_spans('ký hiệu "A" nhỏ') == []

    def test_order_of_appearance_is_preserved(self) -> None:
        """Reading order carries the headline's structure; sorting loses it."""
        caption = '"Dự án cầu Rạch Miễu 2" rồi "Sơn La: Bản miền núi bị ngập"'

        assert quoted_spans(caption) == [
            "Dự án cầu Rạch Miễu 2",
            "Sơn La: Bản miền núi bị ngập",
        ]


class TestReadCaptions:
    def test_reads_utf16_and_keys_on_the_keyframe_ordinal(self, tmp_path) -> None:
        """The source files are UTF-16.

        Read as UTF-8 they decode to nulls interleaved with text and every row
        is silently dropped rather than raising, so the encoding is load-bearing.
        """
        path = tmp_path / "L21_V001.csv"
        path.write_text(
            'keyframe,response\n001.jpg,"Cảnh quay thành phố"\n002.jpg,"Biển báo"\n',
            encoding="utf-16",
        )

        assert read_captions(path) == {1: "Cảnh quay thành phố", 2: "Biển báo"}

    def test_utf8_files_still_load(self, tmp_path) -> None:
        path = tmp_path / "L21_V002.csv"
        path.write_text('keyframe,response\n003.jpg,"Xin chào"\n', encoding="utf-8")

        assert read_captions(path) == {3: "Xin chào"}

    def test_rows_with_no_caption_are_skipped(self, tmp_path) -> None:
        path = tmp_path / "L21_V003.csv"
        path.write_text(
            'keyframe,response\n001.jpg,""\n002.jpg,"có nội dung"\n', encoding="utf-16"
        )

        assert read_captions(path) == {2: "có nội dung"}

    def test_a_non_numeric_name_does_not_break_the_file(self, tmp_path) -> None:
        path = tmp_path / "L21_V004.csv"
        path.write_text(
            'keyframe,response\nthumb.jpg,"bỏ qua"\n004.jpg,"giữ lại"\n',
            encoding="utf-16",
        )

        assert read_captions(path) == {4: "giữ lại"}
