import numpy as np
import pytest

from app.ingestion.video import probe, shot_detect, transnet
from tests.unit.test_shot_detect import write_segmented_video


class TestScenesFromScores:
    def test_no_transition_is_one_shot(self):
        scores = np.zeros(50)

        assert transnet.scenes_from_scores(scores) == [(0, 49)]

    def test_a_shot_ends_on_the_flagged_frame(self):
        # The flagged frame still shows the outgoing content, so it belongs to
        # the shot that is ending, not the one starting.
        scores = np.zeros(100)
        scores[59] = 0.9

        assert transnet.scenes_from_scores(scores) == [(0, 59), (60, 99)]

    def test_multi_frame_transitions_produce_one_boundary(self):
        scores = np.zeros(100)
        scores[40:46] = 0.9

        scenes = transnet.scenes_from_scores(scores)

        assert scenes == [(0, 40), (41, 99)]

    def test_leftover_transition_frames_join_the_incoming_shot(self):
        scores = np.zeros(60)
        scores[20:25] = 0.8

        scenes = transnet.scenes_from_scores(scores)

        # Contiguous: every frame belongs to exactly one shot.
        assert scenes[0][1] + 1 == scenes[1][0]
        assert scenes[0][0] == 0
        assert scenes[-1][1] == 59

    def test_several_transitions(self):
        scores = np.zeros(120)
        scores[30] = 0.9
        scores[70] = 0.9

        assert transnet.scenes_from_scores(scores) == [(0, 30), (31, 70), (71, 119)]

    def test_threshold_is_respected(self):
        scores = np.zeros(50)
        scores[25] = 0.4

        assert transnet.scenes_from_scores(scores, threshold=0.5) == [(0, 49)]
        assert transnet.scenes_from_scores(scores, threshold=0.3) == [(0, 25), (26, 49)]

    def test_empty_input(self):
        assert transnet.scenes_from_scores(np.array([])) == []

    def test_transition_on_the_first_frame_does_not_open_an_empty_shot(self):
        scores = np.zeros(40)
        scores[0] = 0.9

        assert transnet.scenes_from_scores(scores) == [(0, 39)]


class TestMergeShortScenes:
    def test_short_scene_folds_into_its_predecessor(self):
        scenes = [(0, 49), (50, 54), (55, 99)]

        merged = transnet.merge_short_scenes(scenes, min_shot_frames=15)

        assert merged == [(0, 54), (55, 99)]

    def test_a_short_opening_scene_absorbs_the_next(self):
        scenes = [(0, 4), (5, 99)]

        assert transnet.merge_short_scenes(scenes, min_shot_frames=15) == [(0, 99)]

    def test_long_scenes_are_untouched(self):
        scenes = [(0, 49), (50, 99)]

        assert transnet.merge_short_scenes(scenes, 15) == scenes

    def test_merging_keeps_coverage_contiguous(self):
        scenes = [(0, 20), (21, 23), (24, 26), (27, 99)]

        merged = transnet.merge_short_scenes(scenes, min_shot_frames=15)

        assert merged[0][0] == 0
        assert merged[-1][1] == 99
        for earlier, later in zip(merged, merged[1:], strict=False):
            assert later[0] == earlier[1] + 1

    def test_empty_input(self):
        assert transnet.merge_short_scenes([], 15) == []


class TestModelInference:
    def test_detects_known_boundaries(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [60, 60, 60])

        ranges = transnet.detect_ranges(str(source))

        assert ranges == [(0, 59), (60, 119), (120, 179)]

    def test_scores_are_probabilities_one_per_frame(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [40, 40])

        scores = transnet.predict_transition_scores(str(source))

        assert scores.shape == (80,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_analysis_frames_use_the_model_input_size(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [30])

        frames = transnet.decode_analysis_frames(str(source))

        assert frames.shape == (30, transnet.INPUT_HEIGHT, transnet.INPUT_WIDTH, 3)
        assert frames.dtype == np.uint8

    def test_unreadable_video_raises(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")

        with pytest.raises(transnet.TransNetInferenceError, match="cannot decode"):
            transnet.decode_analysis_frames(str(broken))


class TestDetectorSelection:
    def test_transnetv2_is_the_default(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [60, 60])
        video = probe.probe_video(source)

        shots = shot_detect.detect_shots(video)

        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [
            (0, 59),
            (60, 119),
        ]

    def test_both_detectors_find_the_same_hard_cut(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [60, 60])
        video = probe.probe_video(source)

        learned = shot_detect.detect_shots(video, detector="transnetv2")
        content = shot_detect.detect_shots(video, detector="content")

        assert len(learned) == len(content) == 2
        assert learned[1].start_frame == content[1].start_frame

    def test_unknown_detector_is_rejected(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [30])
        video = probe.probe_video(source)

        with pytest.raises(shot_detect.ShotDetectionError, match="unknown detector"):
            shot_detect.detect_shots(video, detector="magic")

    def test_shot_ids_and_timestamps_come_from_the_probe(self, tmp_path):
        source = write_segmented_video(tmp_path / "L01_V001.mp4", [60, 60], rate=25)
        video = probe.probe_video(source)

        shots = shot_detect.detect_shots(video, detector="transnetv2")

        assert [shot.shot_id for shot in shots] == [0, 1]
        assert shots[1].start_sec == pytest.approx(60 / 25)
        assert all(shot.video_id == "L01_V001" for shot in shots)
