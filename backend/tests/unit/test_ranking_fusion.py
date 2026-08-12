from app.ranking import fusion
from app.vector_store.search import ScoredFrame


def frame(
    score: float, video_id: str = "L01_V001", shot_id: int = 0, frame_id: int = 100
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


def clip(
    score: float, video_id: str = "L01_V001", shot_id: int = 0, start_frame: int = 90
) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=None,
        start_frame=start_frame,
        end_frame=start_frame + 50,
        path="/data/videos/L01_V001.mp4",
        start_sec=3.0,
        end_sec=5.0,
    )


class TestFuseFramesAndClips:
    def test_a_shot_both_indexes_found_gets_both_scores(self):
        fused = fusion.fuse_frames_and_clips(
            [frame(0.60, shot_id=1)], [clip(0.40, shot_id=1)], clip_weight=0.5
        )

        assert len(fused) == 1
        assert fused[0].score == 0.60 + 0.5 * 0.40

    def test_agreement_outranks_a_stronger_one_sided_hit(self):
        frames = [frame(0.70, shot_id=1), frame(0.60, shot_id=2)]
        clips = [clip(0.90, shot_id=2), clip(0.10, shot_id=7)]

        fused = fusion.fuse_frames_and_clips(frames, clips, clip_weight=0.5)

        # Shot 2 is weaker on frames, but the clip index strongly agrees with
        # it: 0.60 + 0.45 beats shot 1's 0.70 + the imputed floor of 0.05.
        assert [hit.shot_id for hit in fused] == [2, 1, 7]

    def test_a_shot_missing_from_one_index_is_imputed_not_zeroed(self):
        frames = [frame(0.70, shot_id=1), frame(0.30, shot_id=2)]
        clips = [clip(0.50, shot_id=2)]

        fused = fusion.fuse_frames_and_clips(frames, clips, clip_weight=0.5)
        by_shot = {hit.shot_id: hit.score for hit in fused}

        # Shot 1 has no clip hit, so it borrows the worst clip score seen
        # (0.50) rather than being punished with a zero.
        assert by_shot[1] == 0.70 + 0.5 * 0.50

    def test_a_clip_only_shot_still_reaches_the_results(self):
        fused = fusion.fuse_frames_and_clips(
            [frame(0.80, shot_id=1)], [clip(0.75, shot_id=9)], clip_weight=0.5
        )

        assert {hit.shot_id for hit in fused} == {1, 9}

    def test_the_frame_hit_carries_the_result(self):
        # Only the frame point knows an exact original_frame_id, and that is
        # what goes on a submission.
        fused = fusion.fuse_frames_and_clips(
            [frame(0.6, shot_id=1, frame_id=415)], [clip(0.9, shot_id=1)]
        )

        assert fused[0].original_frame_id == 415

    def test_a_clip_only_shot_falls_back_to_its_start_frame(self):
        fused = fusion.fuse_frames_and_clips([], [clip(0.9, shot_id=1, start_frame=90)])

        assert fused[0].representative_frame == 90

    def test_same_shot_id_in_different_videos_is_not_fused(self):
        fused = fusion.fuse_frames_and_clips(
            [frame(0.6, video_id="L01_V001", shot_id=0)],
            [clip(0.9, video_id="L01_V002", shot_id=0)],
        )

        assert len(fused) == 2

    def test_duplicate_hits_per_shot_collapse_to_the_best(self):
        frames = [frame(0.50, shot_id=1), frame(0.80, shot_id=1)]

        fused = fusion.fuse_frames_and_clips(frames, [clip(0.20, shot_id=1)])

        assert len(fused) == 1
        assert fused[0].score == 0.80 + 0.5 * 0.20

    def test_results_are_ordered_by_fused_score(self):
        frames = [frame(0.5, shot_id=1), frame(0.9, shot_id=2), frame(0.7, shot_id=3)]

        fused = fusion.fuse_frames_and_clips(frames, [clip(0.4, shot_id=1)])

        assert [hit.shot_id for hit in fused] == [2, 3, 1]

    def test_zero_weight_and_no_clips_are_pass_through(self):
        frames = [frame(0.9, shot_id=1), frame(0.5, shot_id=2)]

        assert fusion.fuse_frames_and_clips(frames, [clip(0.9)], clip_weight=0) == frames
        assert fusion.fuse_frames_and_clips(frames, []) == frames

    def test_empty_input(self):
        assert fusion.fuse_frames_and_clips([], []) == []
