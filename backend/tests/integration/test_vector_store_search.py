import uuid

import pytest
from qdrant_client import QdrantClient

from app.features import sparse
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, search, upsert
from app.vector_store.client import build_client

VECTOR_SIZE = 8


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
def collection_name(client):
    name = f"test_search_{uuid.uuid4().hex[:8]}"
    yield name
    if collections.collection_exists(client, name):
        collections.delete_collection(client, name)


def _vector(index: int) -> list[float]:
    """Vectors that get progressively less similar to `_vector(0)`."""
    vector = [0.0] * VECTOR_SIZE
    vector[0] = 1.0
    vector[1] = index / 10.0
    return vector


@pytest.fixture
def populated(client, collection_name):
    collections.create_collection(client, collection_name, VECTOR_SIZE)
    payload_indexes.create_payload_indexes(
        client, collection_name, IngestionEntity.FRAMES
    )

    points = []
    for index in range(9):
        video_id = "L01_V001" if index < 6 else "L01_V002"
        points.append(
            upsert.make_point(
                point_id=index,
                vector=_vector(index),
                payload={
                    "video_id": video_id,
                    "shot_id": index // 3,
                    "original_frame_id": index * 25,
                    "path": f"/kf/{video_id}_{index:06d}.jpg",
                },
            )
        )
    upsert.upsert_points(client, collection_name, points)
    return collection_name


class TestSearch:
    def test_returns_hits_ordered_by_similarity(self, client, populated):
        hits = search.search(client, populated, _vector(0), limit=3)

        assert len(hits) == 3
        assert hits[0].original_frame_id == 0
        assert hits[0].score >= hits[1].score >= hits[2].score

    def test_payload_is_mapped_onto_the_result(self, client, populated):
        hits = search.search(client, populated, _vector(0), limit=1)

        assert hits[0].video_id == "L01_V001"
        assert hits[0].shot_id == 0
        assert hits[0].path == "/kf/L01_V001_000000.jpg"

    def test_representative_frame_uses_the_exact_frame_id(self, client, populated):
        hits = search.search(client, populated, _vector(0), limit=1)

        assert hits[0].representative_frame == hits[0].original_frame_id

    def test_limit_is_respected(self, client, populated):
        assert len(search.search(client, populated, _vector(0), limit=2)) == 2

    def test_video_filter_restricts_results(self, client, populated):
        hits = search.search(
            client,
            populated,
            _vector(0),
            limit=10,
            query_filter=search.build_filter(video_ids=["L01_V002"]),
        )

        assert hits
        assert {hit.video_id for hit in hits} == {"L01_V002"}

    def test_shot_filter_restricts_results(self, client, populated):
        hits = search.search(
            client,
            populated,
            _vector(0),
            limit=10,
            query_filter=search.build_filter(shot_ids=[2]),
        )

        assert hits
        assert {hit.shot_id for hit in hits} == {2}

    def test_no_constraints_means_no_filter(self):
        assert search.build_filter() is None


@pytest.fixture
def hybrid(client, collection_name):
    """Three frames whose speech differs but whose images are near-identical.

    This is the situation the lexical vectors exist for: a news studio shot
    looks the same whatever is being said, so the dense vectors are almost
    indistinguishable and only the words separate the frames.
    """
    texts = [
        "Đồng bằng sông Cửu Long sụt lún gấp hai mươi lần",
        "Nghỉ lễ Quốc khánh năm 2024 từ ngày 31/8",
        "leo thang giữa Israel và Hezbollah",
    ]
    # As the recogniser returns it: all caps, no diacritics, low confidence.
    # Frame 1's ticker says something its speech never mentions, which is the
    # case the OCR slot exists for.
    ocr_texts = [
        "TIN CHINH SUT LUN O DBSCL",
        "Tam DUnG LuU Thong doi Voi Xe 3 BaNH",
        "",
    ]
    # What the VLM read off the same frames. It gets the diacritics right
    # where EasyOCR does not, so both readings feed the one `ocr` slot and
    # both come back on the hit.
    vlm_texts = [
        "",
        "TẠM DỪNG LƯU THÔNG ĐỐI VỚI XE 3 BÁNH",
        "",
    ]
    # Prose descriptions of the same frames. Frame 2's caption names something
    # neither its speech nor its ticker mentions, which is the case the caption
    # slot exists for.
    captions = [
        "Cảnh quay từ trên cao một vùng đất ven biển với dòng sông uốn lượn.",
        "Hai người dẫn chương trình ngồi sau bàn trong trường quay.",
        "Một đàn chim bồ câu bay lên từ quảng trường lát đá.",
    ]
    collections.create_collection(client, collection_name, VECTOR_SIZE)

    points = []
    for index, text in enumerate(texts):
        sparse_vectors = {collections.SPARSE_SPEECH: sparse.encode(text)}
        if ocr_texts[index] or vlm_texts[index]:
            sparse_vectors[collections.SPARSE_OCR] = sparse.encode(
                ocr_texts[index], vlm_texts[index]
            )
        sparse_vectors[collections.SPARSE_CAPTION] = sparse.encode(captions[index])
        points.append(
            upsert.make_point(
                point_id=index,
                # Deliberately ordered so the dense ranking is the reverse of
                # the lexical one; a hit that wins must have won on words.
                vector=_vector(index),
                payload={
                    "video_id": "L01_V001",
                    "shot_id": index,
                    "original_frame_id": index,
                    "asr_text": text,
                    # As `enrichment_payload` writes them: an empty reading is
                    # left out of the payload entirely rather than stored as "".
                    **({"ocr_text": ocr_texts[index]} if ocr_texts[index] else {}),
                    **({"ocr_text_vlm": vlm_texts[index]} if vlm_texts[index] else {}),
                    "caption_vi": captions[index],
                },
                sparse_vectors=sparse_vectors,
            )
        )
    upsert.upsert_points(client, collection_name, points)
    return collection_name


class TestHybridSearch:
    def test_lexical_match_outranks_a_closer_image(self, client, hybrid):
        """The dense-nearest point is frame 0; the words point at frame 2."""
        hits = search.search(
            client,
            hybrid,
            _vector(0),
            limit=3,
            sparse_query=sparse.encode("Israel Hezbollah leo thang"),
        )

        assert hits[0].original_frame_id == 2

    def test_dense_only_query_ignores_the_lexical_vectors(self, client, hybrid):
        hits = search.search(client, hybrid, _vector(0), limit=3)

        assert hits[0].original_frame_id == 0

    def test_diacritic_damaged_text_still_matches(self, client, hybrid):
        """OCR reads `đ` as `d`; the folded token has to bridge that."""
        hits = search.search(
            client,
            hybrid,
            _vector(0),
            limit=3,
            sparse_query=sparse.encode("dồng bằng sông cửu long"),
        )

        assert hits[0].original_frame_id == 0

    def test_frames_without_speech_are_still_reachable(self, client, hybrid):
        """25 videos in this corpus are music only and carry no speech vector.

        Their frames have to keep surfacing through the dense branch of a
        hybrid query rather than dropping out of the result set entirely.
        """
        upsert.upsert_points(
            client,
            hybrid,
            [
                upsert.make_point(
                    99,
                    _vector(0),
                    {"video_id": "L01_V009", "shot_id": 0, "original_frame_id": 99},
                    sparse_vectors={collections.SPARSE_SPEECH: sparse.encode("")},
                )
            ],
        )

        hits = search.search(
            client,
            hybrid,
            _vector(0),
            limit=4,
            sparse_query=sparse.encode("Israel Hezbollah"),
        )

        assert 99 in [hit.original_frame_id for hit in hits]

    def test_on_screen_text_is_reachable_when_speech_never_says_it(
        self, client, hybrid
    ):
        """Frame 1's ticker reads "Tam DUnG LuU Thong"; its speech is about a
        public holiday. Only the OCR slot can answer this."""
        hits = search.search(
            client,
            hybrid,
            _vector(0),
            limit=3,
            sparse_query=sparse.encode("tạm dừng lưu thông"),
        )

        assert hits[0].original_frame_id == 1

    def test_a_frame_carrying_no_ocr_still_survives_the_fusion(
        self, client, hybrid
    ):
        """Frame 2 has no on-screen text at all and must not drop out."""
        hits = search.search(
            client,
            hybrid,
            _vector(2),
            limit=3,
            sparse_query=sparse.encode("Israel Hezbollah"),
        )

        assert 2 in [hit.original_frame_id for hit in hits]

    def test_a_scene_only_the_caption_describes_is_reachable(self, client, hybrid):
        """Nobody says "chim bồ câu" and no ticker writes it; only the VLM
        description of frame 2 contains it."""
        hits = search.search(
            client,
            hybrid,
            _vector(0),
            limit=3,
            sparse_query=sparse.encode("đàn chim bồ câu bay lên"),
        )

        assert hits[0].original_frame_id == 2

    def test_the_three_lexical_slots_stay_independent(self, client, hybrid):
        """Speech, on-screen text and caption each win their own query.

        Pooling them into one vector would let the 465-character caption swamp
        a short headline, and this is what would catch that regression.
        """
        by_channel = {
            "Israel Hezbollah leo thang": 2,   # speech
            "tạm dừng lưu thông": 1,           # on-screen text
            "chim bồ câu quảng trường": 2,     # caption
            "sụt lún đồng bằng": 0,            # speech + on-screen text agree
        }
        for query, expected in by_channel.items():
            hits = search.search(
                client, hybrid, _vector(0), limit=3, sparse_query=sparse.encode(query)
            )
            assert hits[0].original_frame_id == expected, query


class TestOcrOnlySearch:
    """`search_sparse` against the `ocr` slot, with no dense branch at all."""

    def test_only_frames_carrying_the_words_come_back(self, client, hybrid):
        """The fused query has to return `limit` hits and pads with whatever
        the image index liked. This one returns nothing it cannot justify."""
        hits = search.search_sparse(
            client, hybrid, sparse.encode("tạm dừng lưu thông"), limit=3
        )

        assert [hit.original_frame_id for hit in hits] == [1]

    def test_the_image_has_no_say(self, hybrid, client):
        """Frame 0 is the dense-nearest point to `_vector(0)` and wins the
        fused query outright; here it must not appear at all."""
        hits = search.search_sparse(
            client, hybrid, sparse.encode("xe 3 bánh trở lên"), limit=3
        )

        assert 0 not in [hit.original_frame_id for hit in hits]

    def test_speech_does_not_leak_into_the_slot(self, client, hybrid):
        """"Hezbollah" is said on frame 2 and printed nowhere. Matching it
        here would mean the slots were pooled at ingest."""
        assert (
            search.search_sparse(
                client, hybrid, sparse.encode("Israel Hezbollah"), limit=3
            )
            == []
        )

    def test_a_caption_does_not_leak_into_the_slot(self, client, hybrid):
        assert (
            search.search_sparse(
                client, hybrid, sparse.encode("chim bồ câu quảng trường"), limit=3
            )
            == []
        )

    def test_diacritic_damaged_text_is_still_reachable(self, client, hybrid):
        """The ticker was read as "SUT LUN"; the query is typed correctly."""
        hits = search.search_sparse(
            client, hybrid, sparse.encode("sụt lún"), limit=3
        )

        assert [hit.original_frame_id for hit in hits] == [0]

    def test_both_readings_come_back_on_the_hit(self, client, hybrid):
        """The VLM reading is correct Vietnamese and the EasyOCR one is what a
        folded token may have matched. Showing only one can display text that
        does not contain the query that found it."""
        hits = search.search_sparse(
            client, hybrid, sparse.encode("tạm dừng lưu thông"), limit=1
        )

        assert hits[0].ocr_text == (
            "TẠM DỪNG LƯU THÔNG ĐỐI VỚI XE 3 BÁNH"
            " · Tam DUnG LuU Thong doi Voi Xe 3 BaNH"
        )

    def test_a_frame_with_no_on_screen_text_reports_none(self, client, hybrid):
        hits = search.search(client, hybrid, _vector(2), limit=3)

        assert next(h for h in hits if h.original_frame_id == 2).ocr_text is None

    def test_the_video_filter_still_applies(self, client, hybrid):
        hits = search.search_sparse(
            client,
            hybrid,
            sparse.encode("tạm dừng lưu thông"),
            limit=3,
            query_filter=search.build_filter(video_ids=["L09_V999"]),
        )

        assert hits == []


class TestOptimizeCollection:
    def test_collection_reaches_green(self, client, collection_name):
        collections.create_collection(client, collection_name, VECTOR_SIZE)
        upsert.upsert_points(
            client,
            collection_name,
            [upsert.make_point(0, _vector(0), {"video_id": "L01_V001", "shot_id": 0})],
        )

        collections.optimize_collection(client, collection_name, timeout_sec=60)

        assert str(client.get_collection(collection_name).status).endswith("green")

    def test_searchable_after_optimizing(self, client, populated):
        collections.optimize_collection(client, populated, timeout_sec=60)

        assert search.search(client, populated, _vector(0), limit=1)
