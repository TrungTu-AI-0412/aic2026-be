from app.ranking import dedupe
from app.vector_store.search import ScoredFrame


def frame(
    score: float, video_id: str = "L01_V001", shot_id: int = 0, frame_id: int = 0
) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=frame_id,
        start_frame=None,
        end_frame=None,
        path=None,
    )


class TestOverfetchLimit:
    def test_scales_with_top_k(self):
        assert dedupe.overfetch_limit(10, factor=5) == 50

    def test_is_capped(self):
        assert dedupe.overfetch_limit(100, factor=5, cap=200) == 200

    def test_never_returns_zero(self):
        assert dedupe.overfetch_limit(0) >= 1


class TestDedupeByShot:
    def test_one_shot_contributes_one_result(self):
        # Sampling puts ~1 keyframe per second in the index, so a single shot
        # arrives as several near-identical hits.
        frames = [
            frame(0.90, shot_id=3, frame_id=100),
            frame(0.95, shot_id=3, frame_id=125),
            frame(0.88, shot_id=3, frame_id=150),
        ]

        result = dedupe.dedupe_by_shot(frames, top_k=10)

        assert len(result) == 1
        assert result[0].score == 0.95
        assert result[0].original_frame_id == 125

    def test_different_shots_are_kept_apart(self):
        frames = [frame(0.9, shot_id=1), frame(0.8, shot_id=2)]

        assert len(dedupe.dedupe_by_shot(frames, top_k=10)) == 2

    def test_same_shot_id_in_different_videos_is_not_merged(self):
        frames = [
            frame(0.9, video_id="L01_V001", shot_id=0),
            frame(0.8, video_id="L01_V002", shot_id=0),
        ]

        assert len(dedupe.dedupe_by_shot(frames, top_k=10)) == 2

    def test_results_are_ordered_by_score(self):
        frames = [frame(0.5, shot_id=1), frame(0.9, shot_id=2), frame(0.7, shot_id=3)]

        scores = [hit.score for hit in dedupe.dedupe_by_shot(frames, top_k=10)]

        assert scores == [0.9, 0.7, 0.5]

    def test_top_k_is_applied_after_collapsing(self):
        frames = [frame(0.9 - index / 100, shot_id=index) for index in range(10)]

        assert len(dedupe.dedupe_by_shot(frames, top_k=3)) == 3

    def test_empty_input(self):
        assert dedupe.dedupe_by_shot([], top_k=5) == []


class TestDedupeByVideo:
    def test_one_video_contributes_one_result(self):
        frames = [
            frame(0.7, video_id="L01_V001", shot_id=1),
            frame(0.9, video_id="L01_V001", shot_id=2),
            frame(0.8, video_id="L01_V002", shot_id=1),
        ]

        result = dedupe.dedupe_by_video(frames, top_k=10)

        assert [(hit.video_id, hit.score) for hit in result] == [
            ("L01_V001", 0.9),
            ("L01_V002", 0.8),
        ]
