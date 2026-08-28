"""Re-embed an existing collection's frames with a different image encoder.

The point of this is comparison, not replacement. `aic2-frames-v1` holds
SigLIP2-giant vectors with three lexical channels bolted on; this builds the
same collection with the same payload and the same lexical vectors, changing
only the dense half. Any difference in a retrieval number between the two is
then attributable to the image encoder alone.

WHY JINA CLIP V2 IS WORTH AN HOUR OF GPU

SigLIP2's text tower handles Vietnamese poorly. Measured symptom on the real
index: the query `sat lo bo song` — Vietnamese typed without diacritics,
which is how people actually type — returned a close-up of a turtle as its
first hit. Jina CLIP v2 lists Vietnamese among its 89 training languages.
Whether that translates into better retrieval here is exactly what this is
for, and it is unknown until both are scored on the same query set.

THE LEXICAL VECTORS ARE COPIED, NOT RECOMPUTED

They are the output of a tokeniser, not of a model, so recomputing them would
produce identical numbers and add a second source of truth for no gain.
Reading them across also means a bug in this script cannot quietly change
what the lexical channels contain.

    nohup python scripts/ingest_jina.py --source aic2-frames-v1 \\
        --collection aic2-frames-jinaclip2 --keyframe-root DIR &
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.features.multimodal import _load_runtime  # noqa: E402
from app.features.profiles import get_profile  # noqa: E402
from app.schemas.ingestions import IngestionEntity  # noqa: E402
from app.vector_store import collections, payload_indexes  # noqa: E402

PROFILE = "jina-clip-v2"
SCROLL_BATCH = 256
TIMEOUT_SEC = 600


class IngestError(RuntimeError):
    pass


def encode_batch(runtime, paths: list[str]) -> np.ndarray | None:
    images = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except OSError:
            return None
    with runtime.torch.inference_mode():
        vectors = np.asarray(runtime.model.encode_image(images), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build(args: argparse.Namespace) -> int:
    profile = get_profile(PROFILE)
    client = QdrantClient(
        url=args.url, api_key=args.api_key or None, timeout=TIMEOUT_SEC
    )
    if not client.collection_exists(args.source):
        raise IngestError(f"no source collection '{args.source}'")

    started = time.time()
    runtime = _load_runtime(profile)
    print(f"model loaded in {time.time() - started:.0f}s, dim={profile.dimension}",
          flush=True)

    exists = client.collection_exists(args.collection)
    if exists and args.recreate:
        client.delete_collection(args.collection)
        exists = False
    elif exists and not args.resume:
        raise IngestError(
            f"'{args.collection}' exists; pass --resume or --recreate"
        )
    if not exists:
        collections.create_collection(client, args.collection, profile.dimension)
        payload_indexes.create_payload_indexes(
            client, args.collection, IngestionEntity.FRAMES
        )

    total = client.count(args.source, exact=True).count
    print(f"source    : {args.source}, {total:,} points", flush=True)

    written = skipped = 0
    offset = None
    started = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=args.source,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            # The lexical vectors travel with the point; only `dense` is
            # replaced. Naming them explicitly keeps the 1536-float SigLIP2
            # vector off the wire, which is most of the bytes.
            with_vectors=list(collections.SPARSE_VECTOR_NAMES),
        )
        if not points:
            break

        usable = [
            point for point in points if (point.payload or {}).get("path")
        ]
        skipped += len(points) - len(usable)

        for start in range(0, len(usable), args.batch_size):
            chunk = usable[start : start + args.batch_size]
            paths = [str(point.payload["path"]) for point in chunk]
            vectors = encode_batch(runtime, paths)
            if vectors is None:
                # One unreadable file poisons its batch; fall back to one at a
                # time so the rest of the batch is not lost with it.
                for point in chunk:
                    single = encode_batch(runtime, [str(point.payload["path"])])
                    if single is None:
                        skipped += 1
                        continue
                    client.upsert(
                        collection_name=args.collection,
                        points=[_to_point(point, single[0])],
                    )
                    written += 1
                continue

            client.upsert(
                collection_name=args.collection,
                points=[
                    _to_point(point, vector)
                    for point, vector in zip(chunk, vectors)
                ],
            )
            written += len(chunk)

        if written % args.log_every < args.batch_size:
            rate = written / max(time.time() - started, 1e-9)
            left = (total - written) / max(rate, 1e-9) / 3600
            print(
                f"  {written:,}/{total:,}  {rate:.1f}/s  còn ~{left:.1f}h  "
                f"bỏ qua={skipped}",
                flush=True,
            )
        if offset is None:
            break

    print(f"{written:,} written, {skipped:,} skipped", flush=True)
    print("optimising...", flush=True)
    collections.optimize_collection(client, args.collection)
    info = client.get_collection(args.collection)
    print(f"'{args.collection}': {info.points_count:,} points, {info.status}",
          flush=True)
    return 0


def _to_point(point, dense: np.ndarray) -> qmodels.PointStruct:
    vectors: dict = {collections.DENSE_VECTOR_NAME: dense.tolist()}
    for name, value in (point.vector or {}).items():
        if value is not None:
            vectors[name] = value
    return qmodels.PointStruct(id=point.id, vector=vectors, payload=point.payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="aic2-frames-v1")
    parser.add_argument("--collection", default="aic2-frames-jinaclip2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        return build(args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
