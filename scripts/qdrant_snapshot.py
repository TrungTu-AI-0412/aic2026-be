"""Create/download or verify/upload portable Qdrant collection snapshots.

This is an operations tool, intentionally separate from the ingestion API.
Parquet manifests remain the rebuild/audit source of truth; snapshots are the
fast hand-off path when another machine should not recompute embeddings.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

CHUNK_SIZE = 8 * 1024 * 1024
MANIFEST_NAME = "snapshot-manifest.json"


class SnapshotToolError(RuntimeError):
    pass


def _headers(api_key: str | None) -> dict[str, str]:
    return {"api-key": api_key} if api_key else {}


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _request_json(
    method: str,
    url: str,
    api_key: str | None,
    data: bytes | None = None,
) -> dict:
    headers = _headers(api_key)
    if data is not None:
        headers["content-type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=3600) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SnapshotToolError(f"Qdrant returned HTTP {exc.code}: {detail}") from exc


def _server_version(base_url: str, api_key: str | None) -> str:
    response = _request_json("GET", _url(base_url, "/"), api_key)
    version = response.get("version")
    if not version:
        raise SnapshotToolError("Qdrant root response did not include a version")
    return str(version)


def _minor_version(version: str) -> tuple[int, int]:
    return _version_tuple(version)[:2]


def _version_tuple(version: str) -> tuple[int, int, int]:
    try:
        match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", version)
        if match is None:
            raise ValueError
        return tuple(int(part) for part in match.groups())
    except (ValueError, TypeError) as exc:
        raise SnapshotToolError(f"cannot parse Qdrant version '{version}'") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_filename(collection: str) -> str:
    """Return a stable filename without trusting a collection as a path."""
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", collection).strip("._")
    readable = readable[:80] or "collection"
    suffix = hashlib.sha256(collection.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{suffix}.snapshot"


def _manifest_artifact_path(base_dir: Path, filename: str) -> Path:
    """Resolve a manifest artifact while preventing directory traversal."""
    relative_path = Path(filename)
    if relative_path.is_absolute():
        raise SnapshotToolError(
            f"snapshot manifest contains an absolute artifact path: {filename}"
        )
    resolved_base = base_dir.resolve()
    resolved_path = (resolved_base / relative_path).resolve()
    if resolved_path.parent != resolved_base:
        raise SnapshotToolError(
            f"snapshot artifact must be next to the manifest: {filename}"
        )
    return resolved_path


def _download(url: str, destination: Path, api_key: str | None) -> str:
    request = Request(url, headers=_headers(api_key), method="GET")
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=3600) as response, destination.open("wb") as out:
            while chunk := response.read(CHUNK_SIZE):
                out.write(chunk)
                digest.update(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def create_snapshots(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise SnapshotToolError(
            f"output already contains {MANIFEST_NAME}; use a new versioned directory"
        )
    server_version = _server_version(args.url, args.api_key)
    artifacts = []

    for collection in args.collection:
        encoded = quote(collection, safe="")
        collection_response = _request_json(
            "GET", _url(args.url, f"/collections/{encoded}"), args.api_key
        )
        collection_info = collection_response.get("result") or {}

        response = _request_json(
            "POST",
            _url(args.url, f"/collections/{encoded}/snapshots?wait=true"),
            args.api_key,
        )
        snapshot = response.get("result") or {}
        snapshot_name = snapshot.get("name")
        if not snapshot_name:
            raise SnapshotToolError(
                f"snapshot response for '{collection}' did not include a name"
            )

        filename = _snapshot_filename(collection)
        destination = output_dir / filename
        if destination.exists():
            raise SnapshotToolError(
                f"snapshot destination already exists: {destination}"
            )
        checksum = _download(
            _url(
                args.url,
                f"/collections/{encoded}/snapshots/{quote(snapshot_name, safe='')}",
            ),
            destination,
            args.api_key,
        )
        artifacts.append(
            {
                "collection": collection,
                "file": filename,
                "sha256": checksum,
                "bytes": destination.stat().st_size,
                "points_count": collection_info.get("points_count"),
                "source_snapshot_name": snapshot_name,
            }
        )
        print(f"created {destination} ({destination.stat().st_size} bytes)")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_version": server_version,
        "feature_profile": args.feature_profile,
        "collections": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")


def _parse_mappings(values: list[str]) -> dict[str, str]:
    mappings = {}
    for value in values:
        if "=" not in value:
            raise SnapshotToolError(
                f"invalid mapping '{value}'; expected SOURCE=TARGET"
            )
        source, target = value.split("=", 1)
        if not source or not target:
            raise SnapshotToolError(
                f"invalid mapping '{value}'; expected SOURCE=TARGET"
            )
        mappings[source] = target
    return mappings


def _collection_exists(base_url: str, collection: str, api_key: str | None) -> bool:
    url = _url(base_url, f"/collections/{quote(collection, safe='')}")
    request = Request(url, headers=_headers(api_key), method="GET")
    try:
        with urlopen(request, timeout=30):
            return True
    except HTTPError as exc:
        if exc.code == 404:
            return False
        detail = exc.read().decode("utf-8", errors="replace")
        raise SnapshotToolError(f"Qdrant returned HTTP {exc.code}: {detail}") from exc


def _upload_snapshot(
    base_url: str,
    collection: str,
    snapshot_path: Path,
    api_key: str | None,
) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SnapshotToolError(f"invalid Qdrant URL '{base_url}'")

    boundary = f"----aic2026-{uuid.uuid4().hex}"
    preamble = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="snapshot"; '
        f'filename="{snapshot_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    epilogue = f"\r\n--{boundary}--\r\n".encode()
    content_length = len(preamble) + snapshot_path.stat().st_size + len(epilogue)
    query = urlencode({"priority": "snapshot", "wait": "true"})
    base_path = parsed.path.rstrip("/")
    path = (
        f"{base_path}/collections/{quote(collection, safe='')}"
        f"/snapshots/upload?{query}"
    )

    connection_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_cls(parsed.hostname, parsed.port, timeout=3600)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        if api_key:
            connection.putheader("api-key", api_key)
        connection.endheaders()
        connection.send(preamble)
        with snapshot_path.open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                connection.send(chunk)
        connection.send(epilogue)

        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise SnapshotToolError(
                f"Qdrant returned HTTP {response.status}: {body}"
            )
        payload = json.loads(body)
        if payload.get("result") is not True:
            raise SnapshotToolError(f"unexpected restore response: {body}")
    finally:
        connection.close()


def restore_snapshots(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SnapshotToolError("unsupported snapshot manifest schema")

    source_version = str(manifest.get("qdrant_version", ""))
    target_version = _server_version(args.url, args.api_key)
    source_version_tuple = _version_tuple(source_version)
    target_version_tuple = _version_tuple(target_version)
    incompatible = (
        source_version_tuple[:2] != target_version_tuple[:2]
        or target_version_tuple[2] < source_version_tuple[2]
    )
    if not args.allow_version_mismatch and incompatible:
        raise SnapshotToolError(
            "incompatible Qdrant versions: "
            f"snapshot={source_version}, target={target_version}. "
            "The target must have the same minor version and an equal or "
            "newer patch. Run the pinned Compose image or explicitly pass "
            "--allow-version-mismatch after reviewing Qdrant compatibility."
        )

    artifacts = manifest.get("collections")
    if not isinstance(artifacts, list) or not artifacts:
        raise SnapshotToolError("snapshot manifest contains no collections")

    mappings = _parse_mappings(args.map)
    try:
        source_collections = {artifact["collection"] for artifact in artifacts}
    except (KeyError, TypeError) as exc:
        raise SnapshotToolError("snapshot manifest has an invalid collection entry") from exc
    unknown_mappings = set(mappings) - source_collections
    if unknown_mappings:
        unknown = ", ".join(sorted(unknown_mappings))
        raise SnapshotToolError(f"collection mapping not found in manifest: {unknown}")

    base_dir = manifest_path.parent
    prepared = []
    for artifact in artifacts:
        try:
            source_collection = artifact["collection"]
            filename = artifact["file"]
            expected_checksum = artifact["sha256"]
        except (KeyError, TypeError) as exc:
            raise SnapshotToolError(
                "snapshot manifest has an invalid collection entry"
            ) from exc
        target_collection = mappings.get(source_collection, source_collection)
        snapshot_path = _manifest_artifact_path(base_dir, filename)
        if not snapshot_path.is_file():
            raise SnapshotToolError(f"snapshot not found: {snapshot_path}")

        actual_checksum = _sha256(snapshot_path)
        if actual_checksum != expected_checksum:
            raise SnapshotToolError(
                f"checksum mismatch for {snapshot_path.name}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
        if _collection_exists(args.url, target_collection, args.api_key):
            raise SnapshotToolError(
                f"target collection '{target_collection}' already exists; "
                "restore to a new versioned name instead of overwriting it"
            )
        prepared.append((snapshot_path, target_collection))

    target_collections = [target for _, target in prepared]
    if len(set(target_collections)) != len(target_collections):
        raise SnapshotToolError("multiple snapshots map to the same target collection")

    for snapshot_path, target_collection in prepared:
        print(f"restoring {snapshot_path.name} -> {target_collection}")
        _upload_snapshot(args.url, target_collection, snapshot_path, args.api_key)
        print(f"restored {target_collection}")

    print(
        "restore complete; configure QDRANT_*_COLLECTION and FEATURE_PROFILE="
        f"{manifest.get('feature_profile')} before starting the API"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create and download snapshots")
    create.add_argument("--url", default=parser.get_default("url"))
    create.add_argument("--api-key", default=parser.get_default("api_key"))
    create.add_argument("--collection", action="append", required=True)
    create.add_argument("--feature-profile", required=True)
    create.add_argument("--output-dir", default="artifacts/qdrant-snapshots")
    create.set_defaults(handler=create_snapshots)

    restore = subparsers.add_parser("restore", help="verify and upload snapshots")
    restore.add_argument("--url", default=parser.get_default("url"))
    restore.add_argument("--api-key", default=parser.get_default("api_key"))
    restore.add_argument("--manifest", required=True)
    restore.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="restore a source collection under a different versioned name",
    )
    restore.add_argument("--allow-version-mismatch", action="store_true")
    restore.set_defaults(handler=restore_snapshots)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (SnapshotToolError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
