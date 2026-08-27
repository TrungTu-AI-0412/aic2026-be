import pytest

from app.ranking import boost
from app.vector_store.search import ScoredFrame


def frame(
    score: float,
    video_id: str = "L01_V001",
    shot_id: int = 0,
    frame_id: int = 0,
    ocr_text: str | None = None,
) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=frame_id,
        start_frame=None,
        end_frame=None,
        path=None,
        ocr_text=ocr_text,
    )


def rrf(*ranks_and_weights: tuple[int, float]) -> float:
    return sum(w / (boost.RRF_K + r) for r, w in ranks_and_weights)


class TestFusion:
    def test_a_shot_both_channels_found_outranks_one_only_seen_visually(self):
        visual = [frame(0.9, shot_id=1), frame(0.8, shot_id=2)]
        # The visually weaker shot is the one whose ticker matches.
        lexical = [frame(12.0, shot_id=2)]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert [hit.shot_id for hit in fused] == [2, 1]

    def test_a_lexical_only_shot_enters_the_list(self):
        """The whole point of the channel: a name on a chyron has no visual
        signature, so the image index never proposes that frame at all."""
        visual = [frame(0.9, shot_id=1)]
        lexical = [frame(12.0, shot_id=7)]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert {hit.shot_id for hit in fused} == {1, 7}

    def test_a_lexical_only_shot_does_not_displace_a_confident_visual_hit(self):
        """At the default weight it should land below the visual head, not on
        top of it. This is the guard on the weight itself."""
        visual = [frame(0.9, shot_id=1)]
        lexical = [frame(12.0, shot_id=7)]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert fused[0].shot_id == 1

    def test_scores_are_reciprocal_ranks_not_similarities(self):
        visual = [frame(0.9, shot_id=1)]
        lexical = [frame(12.0, shot_id=1)]

        fused = boost.reciprocal_rank_fuse(visual, lexical, weight=0.5)

        assert fused[0].score == pytest.approx(rrf((1, 1.0), (1, 0.5)))

    def test_magnitude_of_the_lexical_score_does_not_matter(self):
        """IDF sums have no upper bound; an arithmetic fusion would let one
        rare token in one query decide the whole ranking."""
        visual = [frame(0.9, shot_id=1), frame(0.8, shot_id=2)]

        modest = boost.reciprocal_rank_fuse(visual, [frame(0.4, shot_id=2)])
        enormous = boost.reciprocal_rank_fuse(visual, [frame(9000.0, shot_id=2)])

        assert [h.score for h in modest] == [h.score for h in enormous]

    def test_the_visual_hit_is_the_carrier(self):
        """Only the frame point knows which exact frame to put on a submission;
        a lexical hit on the same shot must not overwrite it."""
        visual = [frame(0.9, shot_id=2, frame_id=1234)]
        lexical = [frame(12.0, shot_id=2, frame_id=9999)]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert fused[0].original_frame_id == 1234

    def test_on_screen_text_is_carried_across_when_the_visual_hit_lacks_it(self):
        """A clip point carries a shot's range but not its keyframe text; the
        evidence for the boost has to survive onto the result."""
        visual = [frame(0.9, shot_id=2, ocr_text=None)]
        lexical = [frame(12.0, shot_id=2, ocr_text="TẠM DỪNG LƯU THÔNG")]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert fused[0].ocr_text == "TẠM DỪNG LƯU THÔNG"

    def test_existing_text_on_the_visual_hit_is_kept(self):
        visual = [frame(0.9, shot_id=2, ocr_text="from the frame")]
        lexical = [frame(12.0, shot_id=2, ocr_text="from the lexical hit")]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert fused[0].ocr_text == "from the frame"

    def test_duplicate_keyframes_of_one_shot_count_once(self):
        """~1 keyframe/sec means a shot appears many times in a raw list. Left
        uncollapsed, a long shot would take every rank slot and inflate its own
        reciprocal rank."""
        visual = [frame(0.9 - i / 100, shot_id=1, frame_id=i) for i in range(5)]
        lexical = [frame(12.0, shot_id=2)]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert len(fused) == 2
        # One reciprocal rank, not five: the shot occupies rank 1 and nothing
        # else. Its four other keyframes would otherwise have taken ranks 2-5
        # and pushed every real competitor down.
        assert fused[0].shot_id == 1
        assert fused[0].score == pytest.approx(rrf((1, 1.0)))

    def test_shots_are_keyed_by_video_as_well(self):
        visual = [frame(0.9, video_id="L01_V001", shot_id=3)]
        lexical = [frame(12.0, video_id="L02_V001", shot_id=3)]

        fused = boost.reciprocal_rank_fuse(visual, lexical)

        assert len(fused) == 2


class TestDisabled:
    def test_no_lexical_hits_leaves_the_visual_list_untouched(self):
        visual = [frame(0.9, shot_id=1), frame(0.8, shot_id=2)]

        assert boost.reciprocal_rank_fuse(visual, []) == visual

    def test_zero_weight_leaves_the_visual_list_untouched(self):
        visual = [frame(0.9, shot_id=1)]
        lexical = [frame(12.0, shot_id=7)]

        assert boost.reciprocal_rank_fuse(visual, lexical, weight=0.0) == visual

    def test_similarities_survive_when_the_boost_is_off(self):
        """Turning the channel off must not silently change the score scale
        the rest of the pipeline reports."""
        visual = [frame(0.9, shot_id=1)]

        assert boost.reciprocal_rank_fuse(visual, [])[0].score == 0.9


class TestOrdering:
    def test_ties_are_broken_deterministically(self):
        """Two shots each seen by one channel at rank 1 score identically;
        without a stable tie-break the output would follow set iteration."""
        visual = [frame(0.9, video_id="L01_V002", shot_id=0)]
        lexical = [frame(12.0, video_id="L01_V001", shot_id=0)]

        runs = {
            tuple(
                (h.video_id, h.shot_id)
                for h in boost.reciprocal_rank_fuse(visual, lexical, weight=1.0)
            )
            for _ in range(20)
        }

        assert len(runs) == 1

    def test_weight_moves_a_lexical_only_shot_up(self):
        visual = [frame(0.9 - i / 100, shot_id=i) for i in range(1, 6)]
        lexical = [frame(12.0, shot_id=99)]

        low = boost.reciprocal_rank_fuse(visual, lexical, weight=0.1)
        high = boost.reciprocal_rank_fuse(visual, lexical, weight=2.0)

        position = lambda hits: [h.shot_id for h in hits].index(99)  # noqa: E731
        assert position(high) < position(low)
