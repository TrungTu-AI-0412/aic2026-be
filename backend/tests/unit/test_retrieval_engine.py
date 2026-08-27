"""How the engine wires the OCR channel into a query.

The searches themselves are covered against a real Qdrant in
`tests/integration/test_vector_store_search.py`. What is checked here is the
wiring: which slots each branch queries, how many round trips happen, and
whether the boost is reachable at all.
"""

import pytest

from app.features.sparse import SparseVector
from app.retrieval import engine
from app.retrieval.engine import RetrievalConfig, Timings
from app.vector_store import collections
from app.vector_store.search import ScoredFrame

CONFIG = RetrievalConfig(
    frames_collection="frames-v1",
    feature_profile="clip-b32-v1",
)

QUERY = SparseVector(indices=[1, 2], values=[1.0, 1.0])


def frame(score: float, shot_id: int, ocr_text: str | None = None) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id="L01_V001",
        shot_id=shot_id,
        original_frame_id=shot_id * 10,
        start_frame=None,
        end_frame=None,
        path=None,
        ocr_text=ocr_text,
    )


@pytest.fixture
def spy(monkeypatch):
    """Record every Qdrant call the engine makes, and answer them canned."""
    dense: list[dict] = []
    lexical: list[dict] = []
    hits = {"dense": [], "lexical": []}

    def _search(client, collection_name, vector, **kwargs):
        dense.append({"collection": collection_name, **kwargs})
        return list(hits["dense"])

    def _search_sparse(client, collection_name, sparse_query, **kwargs):
        lexical.append({"collection": collection_name, **kwargs})
        return list(hits["lexical"])

    monkeypatch.setattr(engine, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(engine, "search", _search)
    monkeypatch.setattr(engine, "search_sparse", _search_sparse)
    return dense, lexical, hits


class TestSparseNames:
    def test_ocr_is_removed_from_the_server_side_fusion_when_boosted(self):
        """Double counting is the failure this guards: on-screen text once at
        RRF's fixed weight inside Qdrant and again at the configured weight on
        top would make the weight dial mean nothing."""
        names = engine.fused_sparse_names(CONFIG)

        assert collections.SPARSE_OCR not in names
        assert collections.SPARSE_SPEECH in names
        assert collections.SPARSE_CAPTION in names

    def test_all_three_slots_are_fused_when_the_boost_is_off(self):
        config = RetrievalConfig(
            frames_collection="frames-v1",
            feature_profile="clip-b32-v1",
            ocr_boost_enabled=False,
        )

        assert engine.fused_sparse_names(config) == collections.SPARSE_VECTOR_NAMES


class TestSearchVector:
    def test_the_ocr_channel_is_its_own_query(self, spy):
        dense, lexical, _ = spy

        engine.search_vector([0.1], 10, CONFIG, Timings(), sparse_query=QUERY)

        assert len(dense) == 1
        assert len(lexical) == 1
        assert lexical[0]["using"] == collections.SPARSE_OCR

    def test_the_ocr_channel_searches_frames_not_clips(self, spy):
        """A clip point knows a shot's frame range but not which frame inside
        it carries the text."""
        _, lexical, _ = spy
        config = RetrievalConfig(
            frames_collection="frames-v1",
            clips_collection="clips-v1",
            feature_profile="clip-b32-v1",
        )

        engine.search_vector([0.1], 10, config, Timings(), sparse_query=QUERY)

        assert [call["collection"] for call in lexical] == ["frames-v1"]

    def test_a_dense_only_query_makes_no_lexical_call(self, spy):
        """Nothing to match on, so the extra round trip would be pure latency."""
        _, lexical, _ = spy

        engine.search_vector([0.1], 10, CONFIG, Timings(), sparse_query=None)

        assert lexical == []

    def test_the_boost_can_be_turned_off(self, spy):
        _, lexical, _ = spy
        config = RetrievalConfig(
            frames_collection="frames-v1",
            feature_profile="clip-b32-v1",
            ocr_boost_enabled=False,
        )

        engine.search_vector([0.1], 10, config, Timings(), sparse_query=QUERY)

        assert lexical == []

    def test_on_screen_text_lifts_a_shot_the_image_ranked_low(self, spy):
        _, _, hits = spy
        hits["dense"] = [frame(0.9, shot_id=1), frame(0.8, shot_id=2)]
        hits["lexical"] = [frame(11.0, shot_id=2)]

        fused = engine.search_vector(
            [0.1], 10, CONFIG, Timings(), sparse_query=QUERY
        )

        assert [hit.shot_id for hit in fused] == [2, 1]

    def test_the_channel_is_timed_separately(self, spy):
        """It is an extra round trip, and on TRAKE it happens once per event.
        Folding it into "qdrant" would hide that."""
        timings = Timings()

        engine.search_vector([0.1], 10, CONFIG, timings, sparse_query=QUERY)

        assert "ocr" in timings.as_dict()


class TestRetrieveByOcr:
    def test_no_image_encoder_runs(self, spy, monkeypatch):
        def _explode(*args, **kwargs):
            raise AssertionError("the lexical path must not embed anything")

        monkeypatch.setattr(engine, "embed_text", _explode)

        engine.retrieve_by_ocr("tạm dừng lưu thông", 10, CONFIG, Timings())

    def test_no_reranker_runs(self, spy, monkeypatch):
        """BLIP ITM scores how well an image depicts a caption. It cannot read
        a ticker, so letting it reorder these would demote exactly the frames
        the query asked for."""
        _, _, hits = spy
        hits["lexical"] = [frame(9.0, shot_id=1)]
        monkeypatch.setattr(
            engine.rerank,
            "rerank",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("reranked")),
        )

        engine.retrieve_by_ocr("sạt lở", 10, CONFIG, Timings())

    def test_hits_are_collapsed_to_one_per_shot(self, spy):
        _, _, hits = spy
        hits["lexical"] = [frame(9.0, shot_id=1), frame(8.0, shot_id=1)]

        result = engine.retrieve_by_ocr("x", 10, CONFIG, Timings())

        assert len(result) == 1

    def test_a_query_with_no_indexable_tokens_makes_no_call(self, spy):
        """Punctuation alone tokenises to nothing; querying an empty sparse
        vector would return an arbitrary page of the collection."""
        _, lexical, _ = spy

        assert engine.retrieve_by_ocr("...", 10, CONFIG, Timings()) == []
        assert lexical == []

    def test_the_boost_flag_does_not_disable_the_dedicated_path(self, spy):
        """The flag governs how the *visual* ranking is assembled. Turning it
        off must not take /search/ocr down with it."""
        _, _, hits = spy
        hits["lexical"] = [frame(9.0, shot_id=1)]
        config = RetrievalConfig(
            frames_collection="frames-v1",
            feature_profile="clip-b32-v1",
            ocr_boost_enabled=False,
        )

        assert engine.retrieve_by_ocr("sạt lở", 10, config, Timings())
