from collections.abc import Sequence
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_DISTANCE = qmodels.Distance.COSINE

# Named vectors, so one point can hold several embedding spaces and the lexical
# signals side by side and a single query can fuse them.
DENSE_VECTOR_NAME = "dense_video"
DENSE_TEXT_NAME = "dense_text"

SPARSE_SPEECH = "speech"
SPARSE_OCR = "ocr"

# Declared slots per collection. Qdrant cannot add a vector to an existing
# collection, so a slot declared now is a re-upsert later instead of a full
# re-ingest of 293k images. A slot only earns that cost if it names the signal
# that will fill it: `dense_text` holds VLM caption text, and `ocr` holds the
# on-screen text whose extraction run already exists off-machine.
#
# Frames deliberately declare no `speech`: segment-level ASR lives in its own
# collection and reaches frames through the overlap bonus, so pooling it onto
# frames as well would score the same speech twice.
FRAME_DENSE_VECTORS = (DENSE_VECTOR_NAME, DENSE_TEXT_NAME)
FRAME_SPARSE_VECTORS = (SPARSE_OCR,)

# ASR points are text throughout: one dense text space, one lexical vector, both
# populated. Nothing is reserved — a slot for a SigLIP2 text-tower embedding
# would hold room for an approach this pipeline rejected, that encoder being
# caption-trained and weak on long speech.
ASR_DENSE_VECTORS = (DENSE_TEXT_NAME,)
ASR_SPARSE_VECTORS = (SPARSE_SPEECH,)

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
    dense_vectors: dict[str, int],
    sparse_vectors: Sequence[str] = (),
    distance: qmodels.Distance = DEFAULT_DISTANCE,
    indexing_threshold: int | None = INDEXING_DISABLED,
) -> None:
    """Create a collection, by default with indexing deferred for bulk load.

    `optimize_collection` restores the threshold once every point is in.

    `dense_vectors` maps each named dense slot to its dimension, so one
    collection can declare several embedding spaces; every dense vector is
    named rather than anonymous. The sparse slots use Qdrant's IDF modifier so
    lexical scoring is BM25-equivalent with the corpus statistics computed
    server-side.

    Both arguments list what the collection *declares*, not what ingestion
    fills. A declared slot costs nothing until a point uses it, and it is the
    only way to add a signal later without re-ingesting, because Qdrant cannot
    add a vector to a collection that already exists.
    """
    if not dense_vectors:
        raise ValueError(f"'{collection_name}' needs at least one dense vector")

    optimizers_config = (
        None
        if indexing_threshold is None
        else qmodels.OptimizersConfigDiff(indexing_threshold=indexing_threshold)
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            name: qmodels.VectorParams(size=size, distance=distance)
            for name, size in dense_vectors.items()
        },
        sparse_vectors_config={
            name: qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF,
            )
            for name in sparse_vectors
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
