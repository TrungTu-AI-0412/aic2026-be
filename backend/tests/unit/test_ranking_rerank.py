import pytest

from app.features.errors import FeatureExtractionError
from app.ranking import rerank
from app.vector_store.search import ScoredFrame


def frame(score: float, shot_id: int, path: str | None = "/data/kf.jpg") -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id="L01_V001",
        shot_id=shot_id,
        original_frame_id=shot_id * 100,
        start_frame=None,
        end_frame=None,
        path=path,
    )


def clip_hit(score: float, shot_id: int) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id="L01_V001",
        shot_id=shot_id,
        original_frame_id=None,
        start_frame=90,
        end_frame=140,
        path="/data/videos/L01_V001.mp4",
        start_sec=3.0,
        end_sec=5.0,
    )


@pytest.fixture
def fake_scores(monkeypatch):
    """Score hits by shot id from a table, so no model has to load."""
    table: dict[int, float] = {}

    def _score_hits(text, hits, model_id=rerank.DEFAULT_MODEL):
        return [table.get(hit.shot_id, 0.0) for hit in hits]

    monkeypatch.setattr(rerank, "score_hits", _score_hits)
    return table


class TestRerank:
    def test_the_head_is_reordered_by_the_cross_encoder(self, fake_scores):
        fake_scores.update({1: 0.2, 2: 0.9, 3: 0.5})
        hits = [frame(0.90, 1), frame(0.80, 2), frame(0.70, 3)]

        result = rerank.rerank("a red car", hits, top_n=3)

        assert [hit.shot_id for hit in result] == [2, 3, 1]
        assert [hit.score for hit in result] == [0.9, 0.5, 0.2]

    def test_the_tail_keeps_its_order_below_the_head(self, fake_scores):
        # Rerank scores are matching probabilities and the tail keeps cosine
        # scores, so the two blocks must never be sorted against each other.
        fake_scores.update({1: 0.01, 2: 0.02})
        hits = [frame(0.90, 1), frame(0.80, 2), frame(0.70, 3), frame(0.60, 4)]

        result = rerank.rerank("a red car", hits, top_n=2)

        assert [hit.shot_id for hit in result] == [2, 1, 3, 4]
        assert [hit.score for hit in result][2:] == [0.70, 0.60]

    def test_nothing_to_reorder_is_left_alone(self, fake_scores):
        hits = [frame(0.9, 1)]

        assert rerank.rerank("a red car", hits, top_n=30) == hits
        assert rerank.rerank("a red car", [], top_n=30) == []

    def test_identity_of_the_hit_survives_rescoring(self, fake_scores):
        fake_scores.update({1: 0.3, 2: 0.4})
        hits = [frame(0.9, 1), frame(0.8, 2)]

        result = rerank.rerank("a red car", hits, top_n=2)

        assert [hit.original_frame_id for hit in result] == [200, 100]


class TestHitImages:
    def test_a_keyframe_hit_reads_its_jpeg(self, monkeypatch):
        monkeypatch.setattr(rerank.media, "read_image", lambda path: f"img:{path}")

        assert rerank._hit_images(frame(0.9, 1)) == ["img:/data/kf.jpg"]

    def test_a_clip_hit_samples_the_shot(self, monkeypatch):
        seen = {}

        def _sample(segment, count):
            seen["segment"] = segment
            return ["a", "b", "c"][:count]

        monkeypatch.setattr(rerank.media, "sample_clip_frames", _sample)

        images = rerank._hit_images(clip_hit(0.9, 4))

        assert len(images) == rerank.CLIP_FRAMES
        assert seen["segment"].start_sec == 3.0
        assert seen["segment"].end_sec == 5.0

    def test_a_missing_source_scores_zero_instead_of_failing_the_query(
        self, monkeypatch
    ):
        def _boom(path):
            raise FeatureExtractionError("keyframe not found")

        monkeypatch.setattr(rerank.media, "read_image", _boom)

        assert rerank._hit_images(frame(0.9, 1)) == []

    def test_a_hit_without_a_path_is_skipped(self):
        assert rerank._hit_images(frame(0.9, 1, path=None)) == []
