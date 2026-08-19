"""The speech collection and the overlap bonus, against a real Qdrant.

The scenario is the one the feature exists for: a studio shot looks the same
whatever is being said, so the image vectors barely separate the frames and only
the words distinguish them. Frames carry no speech vector at all here -- the
transcript lives in its own collection and is joined back on time.
"""

import uuid

import pytest
from qdrant_client import QdrantClient

from app.features import sparse
from app.ranking import asr as asr_ranking
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, search, upsert
from app.vector_store.client import build_client

IMAGE_SIZE = 8
TEXT_SIZE = 4


@pytest.fixture(scope="module")
def client():
    qdrant_client = build_client()
    try:
        qdrant_client.get_collections()
        yield qdrant_client
        return
    except Exception:
        pass
    yield QdrantClient(location=":memory:")


@pytest.fixture
def names(client):
    suffix = uuid.uuid4().hex[:8]
    created = [f"test_frames_{suffix}", f"test_asr_{suffix}"]
    yield created
    for name in created:
        if collections.collection_exists(client, name):
            collections.delete_collection(client, name)


def image_vector(index: int) -> list[float]:
    """Near-identical images: frame 0 is marginally the closest to the query."""
    vector = [0.0] * IMAGE_SIZE
    vector[0] = 1.0
    vector[1] = index / 100.0
    return vector


def text_vector(index: int) -> list[float]:
    vector = [0.0] * TEXT_SIZE
    vector[index % TEXT_SIZE] = 1.0
    return vector


# Three shots, three seconds apart, each with its own speech.
SHOTS = [(0.0, 4.0), (10.0, 14.0), (20.0, 24.0)]
TEXTS = [
    "Đồng bằng sông Cửu Long sụt lún gấp hai mươi lần",
    "Nghỉ lễ Quốc khánh năm 2024 từ ngày 31/8",
    "leo thang giữa Israel và Hezbollah",
]


@pytest.fixture
def populated(client, names):
    frames_name, asr_name = names

    collections.create_collection(
        client,
        frames_name,
        dense_vectors={
            collections.DENSE_VECTOR_NAME: IMAGE_SIZE,
            collections.DENSE_TEXT_NAME: TEXT_SIZE,
        },
        sparse_vectors=collections.FRAME_SPARSE_VECTORS,
    )
    payload_indexes.create_payload_indexes(
        client, frames_name, IngestionEntity.FRAMES
    )
    client.upsert(
        collection_name=frames_name,
        points=[
            upsert.make_point(
                point_id=index,
                vector=image_vector(index),
                payload={
                    "video_id": "L01_V001",
                    "shot_id": index,
                    "keyframe_n": index + 1,
                    "original_frame_id": index * 100,
                    "pts_sec": (start + end) / 2,
                    "shot_start_sec": start,
                    "shot_end_sec": end,
                    "path": f"k{index}.jpg",
                },
            )
            for index, (start, end) in enumerate(SHOTS)
        ],
    )

    collections.create_collection(
        client,
        asr_name,
        dense_vectors={collections.DENSE_TEXT_NAME: TEXT_SIZE},
        sparse_vectors=collections.ASR_SPARSE_VECTORS,
    )
    payload_indexes.create_payload_indexes(
        client, asr_name, IngestionEntity.ASR_SEGMENTS
    )
    client.upsert(
        collection_name=asr_name,
        points=[
            upsert.make_point(
                point_id=index,
                vector=text_vector(index),
                payload={
                    "video_id": "L01_V001",
                    "segment": index + 1,
                    "start_sec": start + 1.0,
                    "end_sec": end - 1.0,
                    "text_corrected": TEXTS[index],
                },
                sparse_vectors={collections.SPARSE_SPEECH: sparse.encode(TEXTS[index])},
                dense_name=collections.DENSE_TEXT_NAME,
            )
            for index, (start, end) in enumerate(SHOTS)
        ],
    )
    return frames_name, asr_name


class TestFrameCollectionShape:
    def test_frames_hold_two_dense_spaces(self, client, populated):
        frames_name, _ = populated
        info = client.get_collection(frames_name)
        vectors = info.config.params.vectors

        assert set(vectors) == {
            collections.DENSE_VECTOR_NAME,
            collections.DENSE_TEXT_NAME,
        }

    def test_frames_declare_no_speech_slot(self, client, populated):
        frames_name, _ = populated
        declared = client.get_collection(frames_name).config.params.sparse_vectors

        assert collections.SPARSE_SPEECH not in (declared or {})


class TestSearchAsr:
    def test_the_lexical_branch_finds_the_segment_that_says_it(self, client, populated):
        _, asr_name = populated

        _, lexical = search.search_asr(
            client, asr_name, None, sparse.encode("Israel Hezbollah leo thang"), limit=3
        )

        assert lexical[0].segment == 3
        assert "Hezbollah" in lexical[0].text

    def test_the_dense_branch_is_queried_on_the_text_slot(self, client, populated):
        _, asr_name = populated

        dense, _ = search.search_asr(client, asr_name, text_vector(1), None, limit=3)

        assert dense[0].segment == 2

    def test_branches_come_back_separately_for_weighted_fusion(
        self, client, populated
    ):
        """Not fused server-side: Qdrant's RRF combines ranks and has no weight
        to express that dense should count for more than lexical."""
        _, asr_name = populated

        dense, lexical = search.search_asr(
            client, asr_name, text_vector(0), sparse.encode("Israel Hezbollah"), limit=3
        )

        assert dense and lexical
        assert dense[0].segment != lexical[0].segment

    def test_segment_timings_survive_the_round_trip(self, client, populated):
        _, asr_name = populated

        dense, _ = search.search_asr(client, asr_name, text_vector(0), None, limit=1)

        assert (dense[0].start_sec, dense[0].end_sec) == (1.0, 3.0)


class TestOverlapBonus:
    def _frames(self, client, frames_name):
        return search.search(client, frames_name, image_vector(0), limit=3)

    def test_frames_carry_the_shot_range_needed_for_overlap(self, client, populated):
        frames_name, _ = populated

        hits = self._frames(client, frames_name)

        assert hits[0].shot_start_sec is not None
        assert hits[0].time_window(0.0) is not None

    def test_dense_order_puts_frame_zero_first_without_the_bonus(
        self, client, populated
    ):
        frames_name, _ = populated

        hits = self._frames(client, frames_name)

        assert hits[0].shot_id == 0

    def test_matching_speech_lifts_its_own_frame_to_the_top(self, client, populated):
        """The whole point: shot 2's image is the *worst* match, but its speech
        is what the query asks about, so it must overtake."""
        frames_name, asr_name = populated

        frames = self._frames(client, frames_name)
        _, lexical = search.search_asr(
            client, asr_name, None, sparse.encode("Israel Hezbollah leo thang"), limit=3
        )
        segments = asr_ranking.fuse_asr([], lexical, 0.0, 1.0)
        boosted = asr_ranking.apply_asr_bonus(frames, segments, weight=0.5)

        assert boosted[0].shot_id == 2

    def test_the_bonus_is_confined_to_the_overlapping_shot(self, client, populated):
        frames_name, asr_name = populated

        frames = self._frames(client, frames_name)
        before = {hit.shot_id: hit.score for hit in frames}
        _, lexical = search.search_asr(
            client, asr_name, None, sparse.encode("Israel Hezbollah leo thang"), limit=3
        )
        segments = [item for item in lexical if item.segment == 3]
        boosted = asr_ranking.apply_asr_bonus(
            frames, asr_ranking.fuse_asr([], segments, 0.0, 1.0), weight=0.5
        )
        after = {hit.shot_id: hit.score for hit in boosted}

        assert after[2] > before[2]
        assert after[0] == before[0]
        assert after[1] == before[1]

    def test_zero_weight_reproduces_the_unboosted_ranking(self, client, populated):
        frames_name, asr_name = populated

        frames = self._frames(client, frames_name)
        _, lexical = search.search_asr(
            client, asr_name, None, sparse.encode("Israel Hezbollah"), limit=3
        )
        boosted = asr_ranking.apply_asr_bonus(
            frames, asr_ranking.fuse_asr([], lexical, 0.0, 1.0), weight=0.0
        )

        assert [hit.shot_id for hit in boosted] == [hit.shot_id for hit in frames]
