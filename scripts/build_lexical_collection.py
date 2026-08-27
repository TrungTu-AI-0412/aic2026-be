"""Build a lexical-plus-visual collection by reusing an existing dense index.

The team's `aic2026-frames-v2` already holds a SigLIP2-giant vector for all
289,881 keyframes. Re-embedding them to add text would cost about an hour of
L40S time (77 images/sec measured, batch 64) and would produce numerically
identical vectors, so this reads them back out instead and only adds what is
missing: the three lexical slots and the payload they are evidence for.

Reusing the vectors also makes the two collections directly comparable. Any
difference in a retrieval number between them is then attributable to the
lexical channels alone, because the visual half is not merely equivalent but
literally the same floats.

POINT IDENTITY

Source point ids are preserved. They already key `(video_id, keyframe_n)` in
the source collection, and keeping them means a hit here can be traced back
to the point it came from without a translation table. Note that this is the
*target* manifest's `keyframe_n`, which is not the same numbering as this
repo's — see `join_server_frames.py` for why that distinction matters.

    python scripts/build_lexical_collection.py \\
        --frames data/frames-joined.parquet \\
        --source-collection aic2026-frames-v2 \\
        --collection aic2-frames-v1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pyarrow.parquet as pq  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qmodels  # noqa: E402

from app.features import sparse  # noqa: E402
from app.vector_store import collections  # noqa: E402

SOURCE_DENSE = "dense_video"
SCROLL_BATCH = 1000
UPSERT_BATCH = 500
# Bulk upsert against a collection that is indexing can sit well past the
# client default; measured failure at 154k points with it.
TIMEOUT_SEC = 300

TEXT_COLUMNS = (
    "ocr_text",
    "ocr_text_vlm",
    "caption_vi",
    "asr_text",
    "asr_text_corrected",
    "ocr_regions",
)


class BuildError(RuntimeError):
    pass


def load_enrichment(path: str) -> dict[tuple[str, int], dict]:
    """Text for each `(video_id, keyframe_n)` of the target manifest."""
    table = pq.read_table(path)
    missing = [name for name in TEXT_COLUMNS if name not in table.column_names]
    if missing:
        raise BuildError(f"{path} is missing {', '.join(missing)}")

    videos = table.column("video_id").to_pylist()
    numbers = [int(n) for n in table.column("keyframe_n").to_pylist()]
    columns = {name: table.column(name).to_pylist() for name in TEXT_COLUMNS}

    out: dict[tuple[str, int], dict] = {}
    for row, key in enumerate(zip(videos, numbers)):
        fields = {
            name: columns[name][row]
            for name in TEXT_COLUMNS
            if columns[name][row]
        }
        if fields:
            out[key] = fields
    return out


def sparse_vectors(fields: dict) -> dict[str, qmodels.SparseVector]:
    """The three lexical slots, each only when its source has something.

    `encode_document` rather than `encode`: raw counts give a point no length
    normalisation, and these three fields differ in length by an order of
    magnitude, so the frames carrying the most text would win queries they
    have no business winning.
    """
    built: dict[str, sparse.SparseVector] = {}

    speech = sparse.encode_document(
        fields.get("asr_text", ""), fields.get("asr_text_corrected", "")
    )
    if speech:
        built[collections.SPARSE_SPEECH] = speech

    on_screen = sparse.encode_document(
        fields.get("ocr_text", ""), fields.get("ocr_text_vlm", "")
    )
    if on_screen:
        built[collections.SPARSE_OCR] = on_screen

    caption = sparse.encode_document(fields.get("caption_vi", ""))
    if caption:
        built[collections.SPARSE_CAPTION] = caption

    return {
        name: qmodels.SparseVector(indices=vector.indices, values=vector.values)
        for name, vector in built.items()
    }


def build(args: argparse.Namespace) -> int:
    client = QdrantClient(
        url=args.url, api_key=args.api_key or None, timeout=TIMEOUT_SEC
    )

    if not client.collection_exists(args.source_collection):
        raise BuildError(f"no source collection '{args.source_collection}'")
    info = client.get_collection(args.source_collection)
    vectors = info.config.params.vectors
    if SOURCE_DENSE not in vectors:
        raise BuildError(
            f"'{args.source_collection}' has no '{SOURCE_DENSE}' vector; "
            f"found {list(vectors)}"
        )
    dimension = vectors[SOURCE_DENSE].size
    print(f"source: {args.source_collection}, {info.points_count:,} points, "
          f"{SOURCE_DENSE} dim={dimension}")

    enrichment = load_enrichment(args.frames)
    print(f"enrichment rows: {len(enrichment):,}")

    if client.collection_exists(args.collection):
        if not args.recreate:
            raise BuildError(
                f"'{args.collection}' exists; pass --recreate to replace it"
            )
        client.delete_collection(args.collection)
    collections.create_collection(client, args.collection, dimension)
    # Without these every filtered query falls back to a full scan of the
    # collection. The first build of this collection omitted them, and the
    # symptom is not an error - just a `video_id` filter reading 289,881
    # points to return a few hundred.
    payload_indexes.create_payload_indexes(
        client, args.collection, IngestionEntity.FRAMES
    )

    written = no_text = 0
    batch: list[qmodels.PointStruct] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.source_collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=[SOURCE_DENSE],
        )
        if not points:
            break

        for point in points:
            payload = dict(point.payload or {})
            key = (str(payload.get("video_id", "")), int(payload.get("keyframe_n", 0)))
            fields = enrichment.get(key)
            if fields:
                payload.update(fields)
            else:
                no_text += 1
            vector = point.vector[SOURCE_DENSE]
            batch.append(
                qmodels.PointStruct(
                    id=point.id,
                    vector={
                        collections.DENSE_VECTOR_NAME: vector,
                        **sparse_vectors(fields or {}),
                    },
                    payload=payload,
                )
            )

        if len(batch) >= UPSERT_BATCH:
            client.upsert(collection_name=args.collection, points=batch)
            written += len(batch)
            batch = []
            print(f"  {written:,} points", end="\r", flush=True)

        if offset is None:
            break

    if batch:
        client.upsert(collection_name=args.collection, points=batch)
        written += len(batch)

    print(f"  {written:,} points written, {no_text:,} carry no text")
    print("optimising (re-enables indexing, blocks until green)...")
    collections.optimize_collection(client, args.collection)
    final = client.get_collection(args.collection)
    print(f"'{args.collection}': {final.points_count:,} points, {final.status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--source-collection", default="aic2026-frames-v2")
    parser.add_argument("--collection", default="aic2-frames-v1")
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)
    try:
        return build(args)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
