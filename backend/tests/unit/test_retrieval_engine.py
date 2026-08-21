"""`retrieve_per_video`: the second TRAKE stage, without a model or Qdrant.

What matters here is not that a search happens but *how* it is split. One
filtered query per video rather than one query over all of them, because a
single query returns the global top-N across the videos and starves a correct
video that ranks low overall. And no shot collapse, because two events can
happen inside one shot.
"""

from dataclasses import replace

import pytest

from app.retrieval import engine
from app.retrieval.engine import RetrievalConfig, Timings
from app.vector_store.search import AsrSegment, ScoredFrame

CONFIG = RetrievalConfig(
    frames_collection="frames-v2",
    feature_profile="clip-b32-v1",
    asr_collection="asr-v1",
    asr_weight=0.5,
)


def frame(
    score: float,
    video_id: str = "L01_V001",
    shot_id: int = 0,
    frame_id: int = 0,
    pts_sec: float | None = None,
) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=frame_id,
        start_frame=None,
        end_frame=None,
        path=None,
        pts_sec=pts_sec,
    )


@pytest.fixture
def fake_search(monkeypatch):
    """Capture what the vector store was asked, and answer per video."""
    calls: list[tuple[str, list[str] | None, int]] = []
    hits: dict[str, list[ScoredFrame]] = {}
    segments: list[AsrSegment] = []
    encodes: list[str] = []

    def _encode_query(text, config, timings):
        encodes.append(text)
        return [0.0]

    def _search_vector(vector, top_k, config, timings, video_ids=None, sparse=None):
        calls.append(("frames", video_ids, top_k))
        return list(hits.get(video_ids[0] if video_ids else "", []))

    def _search_speech(text, top_k, config, timings, video_ids=None):
        calls.append(("speech", video_ids, top_k))
        return list(segments)

    monkeypatch.setattr(engine, "encode_query", _encode_query)
    monkeypatch.setattr(engine, "encode_query_sparse", lambda text, config: None)
    monkeypatch.setattr(engine, "search_vector", _search_vector)
    monkeypatch.setattr(engine, "search_speech", _search_speech)
    return calls, hits, segments, encodes


def test_one_filtered_query_per_video(fake_search):
    calls, hits, _, encodes = fake_search
    hits["A"] = [frame(0.9, video_id="A")]
    hits["B"] = [frame(0.8, video_id="B")]

    result = engine.retrieve_per_video("run", ["A", "B"], 5, CONFIG, Timings())

    assert [call for call in calls if call[0] == "frames"] == [
        ("frames", ["A"], 5),
        ("frames", ["B"], 5),
    ]
    # One encode for the whole fan-out, not one per video.
    assert encodes == ["run"]
    assert sorted(result) == ["A", "B"]


def test_a_video_with_no_hits_is_still_a_key(fake_search):
    _, hits, _, _ = fake_search
    hits["A"] = [frame(0.9, video_id="A")]

    result = engine.retrieve_per_video("run", ["A", "B"], 5, CONFIG, Timings())

    # The caller drops videos that cannot fill an event; it needs to see the
    # empty slot rather than a missing key.
    assert result["B"] == []


def test_shots_are_not_collapsed(fake_search):
    _, hits, _, _ = fake_search
    hits["A"] = [
        frame(0.9, video_id="A", shot_id=5, frame_id=100),
        frame(0.8, video_id="A", shot_id=5, frame_id=130),
    ]

    result = engine.retrieve_per_video("run", ["A"], 5, CONFIG, Timings())

    assert [hit.representative_frame for hit in result["A"]] == [100, 130]


def test_hits_are_score_ordered_and_truncated(fake_search):
    _, hits, _, _ = fake_search
    hits["A"] = [
        frame(0.5, video_id="A", frame_id=1),
        frame(0.9, video_id="A", frame_id=2),
        frame(0.7, video_id="A", frame_id=3),
    ]

    result = engine.retrieve_per_video("run", ["A"], 2, CONFIG, Timings())

    assert [hit.representative_frame for hit in result["A"]] == [2, 3]


def test_speech_is_searched_once_for_the_whole_candidate_set(fake_search):
    calls, hits, segments, _ = fake_search
    hits["A"] = [frame(0.5, video_id="A", pts_sec=5.0)]
    hits["B"] = [frame(0.5, video_id="B", pts_sec=5.0)]
    segments.append(AsrSegment(score=1.0, video_id="A", start_sec=0.0, end_sec=10.0))
    segments.append(AsrSegment(score=0.0, video_id="B", start_sec=0.0, end_sec=10.0))

    result = engine.retrieve_per_video("run", ["A", "B"], 5, CONFIG, Timings())

    assert [call for call in calls if call[0] == "speech"] == [
        ("speech", ["A", "B"], 5)
    ]
    # Segment scores are min-max normalised over the list they arrive in, so
    # boosting each video from its own list would give both a 1.0 bonus and
    # make the two videos' scores incomparable - which is the one thing the
    # caller ranks on.
    assert result["A"][0].score == pytest.approx(1.0)
    assert result["B"][0].score == pytest.approx(0.5)


def test_no_videos_means_no_queries(fake_search):
    calls, _, _, encodes = fake_search

    assert engine.retrieve_per_video("run", [], 5, CONFIG, Timings()) == {}
    assert calls == []
    assert encodes == []


def test_speech_searches_the_original_while_the_image_space_gets_the_rewrite(
    monkeypatch,
):
    """Rewriting is for the image space only.

    The rewrite is English, because SigLIP2 and the reranker are; the speech
    collection holds Vietnamese transcripts and is searched dense *and* by term
    overlap, so handing it the translation would drop the lexical half to
    nothing. Both entry points have to route the two strings the same way.
    """
    encoded: list[str] = []
    spoken: list[str] = []

    def _encode_query(text, config, timings):
        encoded.append(text)
        return [0.0]

    def _search_speech(text, top_k, config, timings, video_ids=None):
        spoken.append(text)
        return []

    monkeypatch.setattr(engine, "encode_query", _encode_query)
    monkeypatch.setattr(engine, "encode_query_sparse", lambda text, config: None)
    monkeypatch.setattr(
        engine, "search_vector", lambda *args, **kwargs: [frame(0.5)]
    )
    monkeypatch.setattr(engine, "search_speech", _search_speech)

    # `retrieve` reranks by default, which would load a cross-encoder.
    config = replace(CONFIG, rerank_enabled=False)
    engine.retrieve(
        "a man running", 5, config, Timings(), speech_text="người đàn ông đang chạy"
    )
    engine.retrieve_per_video(
        "a man running",
        ["A"],
        5,
        config,
        Timings(),
        speech_text="người đàn ông đang chạy",
    )

    assert encoded == ["a man running", "a man running"]
    assert spoken == ["người đàn ông đang chạy", "người đàn ông đang chạy"]
