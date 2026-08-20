import pytest

from app.ranking import asr
from app.vector_store.search import AsrSegment, ScoredFrame


def segment(score: float, start: float, end: float, index: int = 0, video: str = "L01_V001"):
    return AsrSegment(
        score=score, video_id=video, start_sec=start, end_sec=end, segment=index
    )


def frame(score: float, shot_start: float, shot_end: float, video: str = "L01_V001"):
    return ScoredFrame(
        score=score,
        video_id=video,
        shot_id=0,
        original_frame_id=1,
        start_frame=None,
        end_frame=None,
        path="k.jpg",
        shot_start_sec=shot_start,
        shot_end_sec=shot_end,
    )


class TestNormalizeScores:
    def test_scores_are_rescaled_to_the_unit_range(self):
        out = asr.normalize_scores(
            [segment(0.2, 0, 1, 0), segment(0.7, 1, 2, 1), segment(0.45, 2, 3, 2)]
        )

        assert [round(item.score, 4) for item in out] == [0.0, 1.0, 0.5]

    def test_a_single_hit_normalises_to_one(self):
        """With no spread there is no ranking information to preserve, and
        mapping the only hit to 0 would discard the branch entirely."""
        assert asr.normalize_scores([segment(0.42, 0, 1)])[0].score == 1.0

    def test_tied_scores_all_normalise_to_one(self):
        out = asr.normalize_scores([segment(0.5, 0, 1, 0), segment(0.5, 1, 2, 1)])

        assert [item.score for item in out] == [1.0, 1.0]

    def test_empty_stays_empty(self):
        assert asr.normalize_scores([]) == []


class TestFuseAsr:
    def test_dense_outweighs_sparse_at_the_configured_ratio(self):
        """The whole reason this is not Qdrant RRF: RRF fuses ranks and cannot
        express that dense should count for more than lexical."""
        dense = [segment(1.0, 0, 5, 0), segment(0.0, 5, 10, 1)]
        sparse = [segment(0.0, 0, 5, 0), segment(1.0, 5, 10, 1)]

        fused = {item.segment: item.score for item in asr.fuse_asr(dense, sparse, 0.7, 0.3)}

        # Segment 0 wins the dense branch, segment 1 wins the lexical one.
        assert fused[0] == pytest.approx(0.7)
        assert fused[1] == pytest.approx(0.3)

    def test_a_segment_found_by_both_branches_sums_them(self):
        dense = [segment(1.0, 0, 5, 0), segment(0.0, 9, 10, 9)]
        sparse = [segment(1.0, 0, 5, 0), segment(0.0, 9, 10, 9)]

        fused = {item.segment: item.score for item in asr.fuse_asr(dense, sparse, 0.7, 0.3)}

        assert fused[0] == pytest.approx(1.0)

    def test_ranking_is_by_descending_fused_score(self):
        dense = [segment(1.0, 0, 5, 0), segment(0.5, 5, 10, 1), segment(0.0, 10, 15, 2)]

        ranked = asr.fuse_asr(dense, [], 1.0, 0.0)

        assert [item.segment for item in ranked] == [0, 1, 2]

    def test_zero_sparse_weight_drops_the_lexical_branch(self):
        dense = [segment(1.0, 0, 5, 0)]
        sparse = [segment(1.0, 90, 95, 7)]

        fused = asr.fuse_asr(dense, sparse, 1.0, 0.0)

        assert [item.segment for item in fused] == [0]

    def test_one_branch_alone_still_ranks(self):
        fused = asr.fuse_asr([], [segment(1.0, 0, 5, 3)], 0.7, 0.3)

        assert [item.segment for item in fused] == [3]


class TestApplyAsrBonus:
    def test_overlapping_speech_raises_the_frame_score(self):
        frames = [frame(0.50, 0.0, 4.0)]

        out = asr.apply_asr_bonus(frames, [segment(1.0, 1.0, 3.0)], 0.3, pad_sec=0.0)

        assert out[0].score == pytest.approx(0.50 + 0.3 * 1.0)

    def test_non_overlapping_speech_leaves_the_score_untouched(self):
        """Absence of speech is never evidence against a frame: 4.5% of video
        time has no segment and 22 of 873 videos have no transcript at all."""
        frames = [frame(0.50, 0.0, 4.0)]

        out = asr.apply_asr_bonus(frames, [segment(1.0, 60.0, 65.0)], 0.3, pad_sec=0.0)

        assert out[0].score == 0.50

    def test_speech_from_another_video_never_leaks(self):
        frames = [frame(0.50, 0.0, 4.0, video="L01_V001")]
        segments = [segment(1.0, 1.0, 3.0, video="L01_V002")]

        out = asr.apply_asr_bonus(frames, segments, 0.3, pad_sec=0.0)

        assert out[0].score == 0.50

    def test_pad_widens_the_overlap_window(self):
        """Segment bounds are rounded to whole seconds in the source, so an
        exact test misses real overlaps at the edges."""
        frames = [frame(0.50, 10.0, 12.0)]
        just_outside = [segment(1.0, 12.5, 14.0)]

        assert asr.apply_asr_bonus(frames, just_outside, 0.3, pad_sec=0.0)[0].score == 0.50
        assert asr.apply_asr_bonus(frames, just_outside, 0.3, pad_sec=1.0)[0].score > 0.50

    def test_the_best_overlapping_segment_wins_rather_than_the_sum(self):
        """A long shot spans several segments; summing would reward shot length
        instead of relevance."""
        frames = [frame(0.0, 0.0, 100.0)]
        segments = [
            segment(1.0, 1.0, 2.0, 0),
            segment(0.5, 3.0, 4.0, 1),
            segment(0.0, 5.0, 6.0, 2),
        ]

        out = asr.apply_asr_bonus(frames, segments, 1.0, pad_sec=0.0)

        assert out[0].score == pytest.approx(1.0)

    def test_zero_weight_is_a_no_op(self):
        frames = [frame(0.50, 0.0, 4.0), frame(0.40, 5.0, 9.0)]

        out = asr.apply_asr_bonus(frames, [segment(1.0, 1.0, 3.0)], 0.0)

        assert out == frames

    def test_no_segments_is_a_no_op(self):
        frames = [frame(0.50, 0.0, 4.0)]

        assert asr.apply_asr_bonus(frames, [], 0.3) == frames

    def test_the_bonus_can_reorder_results(self):
        """The point of the feature: a visually weaker frame whose speech
        matches should be able to overtake a closer image that says nothing."""
        frames = [frame(0.60, 0.0, 4.0), frame(0.55, 10.0, 14.0)]
        segments = [segment(1.0, 11.0, 13.0)]

        out = asr.apply_asr_bonus(frames, segments, 0.3, pad_sec=0.0)

        assert out[0].shot_start_sec == 10.0
        assert out[0].score == pytest.approx(0.55 + 0.3)

    def test_a_frame_without_a_shot_range_falls_back_to_its_timestamp(self):
        bare = ScoredFrame(
            score=0.5,
            video_id="L01_V001",
            shot_id=0,
            original_frame_id=1,
            start_frame=None,
            end_frame=None,
            path="k.jpg",
            pts_sec=12.0,
        )

        out = asr.apply_asr_bonus([bare], [segment(1.0, 11.0, 13.0)], 0.3, pad_sec=0.0)

        assert out[0].score == pytest.approx(0.8)
