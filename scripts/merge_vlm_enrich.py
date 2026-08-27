"""Fold a VLM enrichment run into an existing collection, in place.

`vlm_enrich.py` produces JSONL keyed by `(video_id, keyframe_n)`; the points
it describes are already in the collection, carrying their dense vector and
nothing else. This attaches the text and the two lexical vectors it feeds.

WHY NOT REBUILD

Rebuilding would re-read 289,881 dense vectors to change 30,750 of them, and
would take the collection down while it happened. Qdrant can set payload and
add named vectors on existing points, so this touches only the rows that
changed and the collection stays queryable throughout.

IDEMPOTENT ON PURPOSE

Both operations are writes to a known point id, so running this twice is a
no-op rather than a duplicate, and running it against a partial JSONL and
then again against the finished one is a valid way to work. That matters
because the producing job is hours long and resumable — there is no reason
its consumer should demand it be complete.

THE KEY IS NOT THE POINT ID

The JSONL knows `(video_id, keyframe_n)`, not Qdrant's ids, so the mapping is
rebuilt by scrolling the collection's payload once. `keyframe_n` here means
the *target* manifest's numbering, which is not this repo's — the two sample
differently and share no numbering. See `join_server_frames.py`.

    python scripts/merge_vlm_enrich.py \\
        --enrich data/vlm-enrich.jsonl --collection aic2-frames-v1
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.features import sparse  # noqa: E402
from app.vector_store import collections  # noqa: E402

SCROLL_BATCH = 4000
WRITE_BATCH = 500
TIMEOUT_SEC = 300


class MergeError(RuntimeError):
    pass


def load_enrichment(path: str) -> dict[tuple[str, int], dict]:
    """Last write wins, so a resumed run's later answer replaces an earlier."""
    rows: dict[tuple[str, int], dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves one truncated line. Skipping it
                # is right; refusing to start because of it is not.
                continue
            text = (row.get("ocr_text_vlm") or "").strip()
            caption = (row.get("caption_vi") or "").strip()
            if not text and not caption:
                continue
            rows[(row["video_id"], int(row["keyframe_n"]))] = {
                "ocr_text_vlm": text,
                "caption_vi": caption,
            }
    return rows


def point_ids(client: QdrantClient, collection: str) -> dict[tuple[str, int], int]:
    mapping: dict[tuple[str, int], int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["video_id", "keyframe_n"],
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            mapping[(str(payload.get("video_id", "")), int(payload.get("keyframe_n", 0)))] = point.id
        if offset is None:
            break
    return mapping


def merge(args: argparse.Namespace) -> int:
    client = QdrantClient(
        url=args.url, api_key=args.api_key or None, timeout=TIMEOUT_SEC
    )
    if not client.collection_exists(args.collection):
        raise MergeError(f"no collection '{args.collection}'")

    enrichment = load_enrichment(args.enrich)
    print(f"enrichment rows : {len(enrichment):,}")
    if not enrichment:
        return 0

    mapping = point_ids(client, args.collection)
    print(f"points in index : {len(mapping):,}")

    payload_batch: list[tuple[int, dict]] = []
    vector_batch: list[qmodels.PointVectors] = []
    written = orphaned = 0
    with_text = with_caption = 0

    def flush() -> None:
        nonlocal payload_batch, vector_batch
        for point_id, payload in payload_batch:
            client.set_payload(
                collection_name=args.collection, payload=payload, points=[point_id]
            )
        if vector_batch:
            client.update_vectors(
                collection_name=args.collection, points=vector_batch
            )
        payload_batch, vector_batch = [], []

    for key, fields in enrichment.items():
        point_id = mapping.get(key)
        if point_id is None:
            orphaned += 1
            continue

        payload_batch.append((point_id, dict(fields)))

        vectors: dict[str, qmodels.SparseVector] = {}
        on_screen = sparse.encode_document(fields["ocr_text_vlm"])
        if on_screen:
            vectors[collections.SPARSE_OCR] = qmodels.SparseVector(
                indices=on_screen.indices, values=on_screen.values
            )
            with_text += 1
        caption = sparse.encode_document(fields["caption_vi"])
        if caption:
            vectors[collections.SPARSE_CAPTION] = qmodels.SparseVector(
                indices=caption.indices, values=caption.values
            )
            with_caption += 1
        if vectors:
            vector_batch.append(qmodels.PointVectors(id=point_id, vector=vectors))

        written += 1
        if len(payload_batch) >= WRITE_BATCH:
            flush()
            print(f"  {written:,}", end="\r", flush=True)

    flush()
    print(f"  updated       : {written:,}")
    print(f"  gained ocr    : {with_text:,}")
    print(f"  gained caption: {with_caption:,}")
    print(f"  not in index  : {orphaned:,}")

    info = client.get_collection(args.collection)
    print(f"'{args.collection}': {info.points_count:,} points, {info.status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrich", required=True)
    parser.add_argument("--collection", default="aic2-frames-v1")
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    args = parser.parse_args(argv)
    try:
        return merge(args)
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
