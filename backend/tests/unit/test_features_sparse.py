from app.features import sparse


class TestTokenize:
    def test_splits_on_punctuation_and_lowercases(self) -> None:
        assert sparse.tokenize("Đồng bằng sông Cửu Long, 31/12/2025!") == [
            "đồng",
            "bằng",
            "sông",
            "cửu",
            "long",
            "31",
            "12",
            "2025",
        ]

    def test_drops_overlong_tokens(self) -> None:
        """Burned-in timecodes and ID strings match nothing anyone queries."""
        assert sparse.tokenize("a" * 30) == []

    def test_normalises_composed_and_decomposed_forms(self) -> None:
        """The same word typed two ways must land on one token."""
        composed = "đường"  # đường, NFC
        decomposed = "đường"  # đường, NFD
        assert sparse.tokenize(composed) == sparse.tokenize(decomposed)


class TestStripDiacritics:
    def test_folds_tone_marks_and_the_d_stroke(self) -> None:
        assert sparse.strip_diacritics("đường sụt lún") == "duong sut lun"

    def test_ocr_confusion_folds_together(self) -> None:
        """OCR reads `đ` as `d`; both spellings must reach one token."""
        assert sparse.strip_diacritics("dường") == sparse.strip_diacritics("đường")

    def test_asr_confusion_does_not_fold_together(self) -> None:
        """ASR errors are consonant-level, and folding must not hide that.

        `sục` for `sụt` is a different word, not a different spelling; merging
        them would invent matches that are simply wrong.
        """
        assert sparse.strip_diacritics("sục") != sparse.strip_diacritics("sụt")
        assert sparse.strip_diacritics("lúng") != sparse.strip_diacritics("lún")


class TestTokenIndex:
    def test_is_stable_across_calls(self) -> None:
        """Ingest and query run in different processes; a salted hash would
        give them different vocabularies and silently kill every match."""
        assert sparse.token_index("đường") == sparse.token_index("đường")

    def test_fits_the_sparse_index_range(self) -> None:
        for word in ("đường", "Israel", "2025", "hezbollah"):
            assert 0 <= sparse.token_index(word) <= 0x7FFFFFFF


class TestEncode:
    def test_empty_input_is_falsy(self) -> None:
        assert not sparse.encode("")
        assert not sparse.encode("", "")

    def test_repeated_terms_accumulate(self) -> None:
        once = sparse.encode("bão")
        twice = sparse.encode("bão bão")
        slot = sparse.token_index("bão")

        assert twice.values[twice.indices.index(slot)] == 2.0
        assert once.values[once.indices.index(slot)] == 1.0

    def test_folded_variant_is_weighted_below_the_exact_token(self) -> None:
        vector = sparse.encode("đường")
        exact = vector.values[vector.indices.index(sparse.token_index("đường"))]
        folded = vector.values[vector.indices.index(sparse.token_index("duong"))]

        assert exact > folded

    def test_indices_are_sorted_and_unique(self) -> None:
        vector = sparse.encode("Đồng bằng sông Cửu Long sụt lún")

        assert vector.indices == sorted(vector.indices)
        assert len(vector.indices) == len(set(vector.indices))
        assert len(vector.indices) == len(vector.values)

    def test_ocr_damaged_query_overlaps_the_correct_text(self) -> None:
        damaged = sparse.encode("dường cao tốc")
        correct = sparse.encode("đường cao tốc")

        assert set(damaged.indices) & set(correct.indices)

    def test_asr_damaged_query_does_not_overlap(self) -> None:
        assert not set(sparse.encode("sục lúng").indices) & set(
            sparse.encode("sụt lún").indices
        )

    def test_folding_can_be_disabled(self) -> None:
        assert len(sparse.encode("đường", fold_diacritics=False).indices) == 1

    def test_pools_several_fields(self) -> None:
        pooled = sparse.encode("bão", "lụt")

        assert sparse.token_index("bão") in pooled.indices
        assert sparse.token_index("lụt") in pooled.indices
