import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_DISTANCE = qmodels.Distance.COSINE

# Named vectors, so a collection can hold the image embedding and the lexical
# signals side by side on one point and a single query can fuse them.
DENSE_VECTOR_NAME = "dense"

# All three lexical slots are declared at creation time even though only
# `speech` is populated today. Qdrant cannot add a vector to an existing
# collection, so declaring `ocr` and `caption` now is what makes those a
# re-upsert later instead of a full re-ingest of 177k images.
SPARSE_SPEECH = "speech"
SPARSE_OCR = "ocr"
SPARSE_CAPTION = "caption"
SPARSE_VECTOR_NAMES = (SPARSE_SPEECH, SPARSE_OCR, SPARSE_CAPTION)

# Qdrant's own default. Restored after a bulk load to let the HNSW index build.
DEFAULT_INDEXING_THRESHOLD = 20_000

# 0 disables indexing entirely, which is the documented way to load a large
# collection quickly: building HNSW incrementally during upsert costs far more
# than building it once at the end.
INDEXING_DISABLED = 0

DEFAULT_OPTIMIZE_TIMEOUT_SEC = 1800.0
_POLL_INTERVAL_SEC = 2.0


class CollectionOptimizeTimeout(Exception):
    pass


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return client.collection_exists(collection_name)


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = DEFAULT_DISTANCE,
    indexing_threshold: int | None = INDEXING_DISABLED,
) -> None:
    """Create a collection, by default with indexing deferred for bulk load.

    `optimize_collection` restores the threshold once every point is in.

    The dense vector is named rather than anonymous, and the sparse slots use
    Qdrant's IDF modifier so lexical scoring is BM25-equivalent with the
    corpus statistics computed server-side.
    """
    optimizers_config = (
        None
        if indexing_threshold is None
        else qmodels.OptimizersConfigDiff(indexing_threshold=indexing_threshold)
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=vector_size, distance=distance
            )
        },
        sparse_vectors_config={
            name: qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF,
            )
            for name in SPARSE_VECTOR_NAMES
        },
        optimizers_config=optimizers_config,
    )


def optimize_collection(
    client: QdrantClient,
    collection_name: str,
    indexing_threshold: int = DEFAULT_INDEXING_THRESHOLD,
    timeout_sec: float = DEFAULT_OPTIMIZE_TIMEOUT_SEC,
) -> None:
    """Re-enable indexing and block until the collection finishes optimizing.

    Waiting matters: returning while the collection is still yellow would let
    a job report success while the first queries still run unindexed and slow.
    """
    client.update_collection(
        collection_name=collection_name,
        optimizers_config=qmodels.OptimizersConfigDiff(
            indexing_threshold=indexing_threshold
        ),
    )

    deadline = time.monotonic() + timeout_sec
    while True:
        status = str(client.get_collection(collection_name).status)
        if status.endswith("green"):
            return
        if time.monotonic() >= deadline:
            raise CollectionOptimizeTimeout(
                f"'{collection_name}' still {status} after {timeout_sec:.0f}s"
            )
        time.sleep(_POLL_INTERVAL_SEC)


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    client.delete_collection(collection_name=collection_name)
