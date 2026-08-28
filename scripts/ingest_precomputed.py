"""Ingest keyframes using the organiser's precomputed CLIP features.

The normal path (`app.ingestion.pipeline`) embeds each keyframe as it goes,
which needs the 28.7 GiB of images and a GPU. The organiser also ships the
features themselves — 168 MB, one `.npy` per video — so a working, measurable
index can exist today instead of after a download and an embedding run.

Everything except the dense vector is shared with the normal path: point ids,
payloads, sparse vectors, collection setup. Only the source of the dense
vector differs, so nothing about ranking behaviour is special-cased here.

⚠ THIS IS A BASELINE, NOT THE TARGET

CLIP ViT-B/32 at 512 dimensions is the weakest profile in `profiles.py`.
SigLIP2 is expected to retrieve better. The point of ingesting these first is
that no retrieval number exists at all yet, and every design decision in this
repo is currently unvalidated. Measure with these, then decide whether the
SigLIP2 embedding run is worth its cost — rather than assuming.

ALIGNMENT

Row `i` of `<video_id>.npy` is `keyframe_n == i + 1`. Verified across the whole
corpus before writing this: all 873 videos are present, and every file's row
count equals that video's highest keyframe_n exactly. The vectors also arrive
L2-normalised (norm 1.0000), which is what the rest of the pipeline assumes.

    docker compose up -d qdrant
    python scripts/ingest_precomputed.py \\
        --frames data/frames.parquet \\
        --features data/clipfeat/clip-features-32 \\
        --collection aic2026-frames-clipb32
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ingestion import manifest as manifest_module  # noqa: E402
from app.ingestion.pipeline import _sparse_vectors  # noqa: E402
from app.schemas.ingestions import IngestionEntity  # noqa: E402
from app.vector_store import collections, payload_indexes, upsert  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.core.config import settings  # noqa: E402

EXPECTED_DIM = 512
# The default client timeout is fine for a query and too short for a bulk
# upsert: Qdrant is indexing while it accepts writes, and a batch that lands
# during a segment merge can sit well past it. Measured failure at 154,368
# points on this corpus with the default.
UPSERT_TIMEOUT_SEC = 300


class IngestError(RuntimeError):
    pass


class FeatureStore:
    """Per-video feature matrices, opened one at a time.

    The manifest is ordered by video, so holding a single open array is enough
    and keeps 336 MB of float16 off the heap.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._video_id: str | None = None
        self._array: np.ndarray | None = None

    def vector(self, video_id: str, keyframe_n: int) -> list[float] | None:
        if video_id != self._video_id:
            path = self._dir / f"{video_id}.npy"
            if not path.is_file():
                self._video_id, self._array = video_id, None
            else:
                self._video_id = video_id
                self._array = np.load(path)
        if self._array is None:
            return None
        index = keyframe_n - 1
        if not 0 <= index < len(self._array):
            return None
        return self._array[index].astype(np.float32).tolist()


def ingest(args: argparse.Namespace) -> int:
    features = Path(args.features)
    if not features.is_dir():
        raise IngestError(f"no feature directory at {features}")

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        timeout=UPSERT_TIMEOUT_SEC,
    )

    exists = collections.collection_exists(client, args.collection)
    if exists and args.recreate:
        collections.delete_collection(client, args.collection)
        exists = False
    elif exists and not args.resume:
        raise IngestError(
            f"collection '{args.collection}' exists; pass --resume to continue "
            f"filling it or --recreate to replace it"
        )

    if not exists:
        collections.create_collection(client, args.collection, EXPECTED_DIM)
        payload_indexes.create_payload_indexes(
            client, args.collection, IngestionEntity.FRAMES
        )

    store = FeatureStore(features)
    batch: list = []
    written = skipped = 0

    for row in manifest_module.iter_rows(args.frames, IngestionEntity.FRAMES):
        vector = store.vector(row.video_id, row.keyframe_n)
        if vector is None:
            # A keyframe with no feature row cannot be searched, and writing a
            # zero vector would make it a false neighbour of everything.
            skipped += 1
            continue
        batch.append(
            upsert.make_point(
                point_id=upsert.deterministic_point_id(*row.point_parts()),
                vector=vector,
                payload=row.payload(),
                sparse_vectors=_sparse_vectors(row),
            )
        )
        if len(batch) >= args.batch_size:
            written += upsert.upsert_points(client, args.collection, batch)
            batch = []
            print(f"  {written:,} points", end="\r", flush=True)

    if batch:
        written += upsert.upsert_points(client, args.collection, batch)

    print(f"  {written:,} points written, {skipped:,} skipped for no feature row")
    print("optimising (re-enables indexing, blocks until green)...")
    collections.optimize_collection(client, args.collection)
    info = client.get_collection(args.collection)
    print(f"collection '{args.collection}': {info.points_count:,} points, {info.status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/frames.parquet")
    parser.add_argument("--features", default="data/clipfeat/clip-features-32")
    parser.add_argument("--collection", default="aic2026-frames-clipb32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep an existing collection and re-upsert every row; point ids "
        "are deterministic, so rewriting a row that is already there is a "
        "no-op rather than a duplicate",
    )
    args = parser.parse_args(argv)
    try:
        return ingest(args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
