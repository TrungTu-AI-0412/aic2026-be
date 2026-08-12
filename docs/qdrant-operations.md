# Qdrant deployment and snapshot hand-off

For a complete Windows workstation or Ubuntu server setup, API startup,
ingestion, and validation flow, see the
[fresh-machine setup runbook](runbook-ubuntu.md).

The repository Compose file runs one pinned Qdrant server with persistent
storage. Collection snapshots are the fast deployment artifact: a recipient
restores the already-computed vectors, payloads, and payload indexes instead
of embedding the media again. Parquet manifests remain the rebuild and audit
source of truth.

## Start Qdrant on a configured machine

From the repository root:

```bash
cp .env.example .env
# Replace QDRANT_API_KEY in .env with: openssl rand -hex 32
mkdir -p data/qdrant/storage data/qdrant/snapshots
docker compose up -d qdrant
set -a && . ./.env && set +a
curl --fail -H "api-key: ${QDRANT_API_KEY}" http://127.0.0.1:6333/readyz
```

The default bind address is `127.0.0.1`, so Qdrant is not exposed directly to
the internet. If the API is later put in another Compose container, connect it
to `http://qdrant:6333` on the Compose network rather than publishing Qdrant
publicly. Compose requires `QDRANT_API_KEY` and passes it to Qdrant. Add TLS at
a private proxy/firewall boundary before changing `QDRANT_BIND_ADDRESS` to a
public interface.

The bind-mounted directories under `data/qdrant/` survive container restarts
and image recreation. Keep the exact Qdrant image tag pinned: collection
snapshots require the same minor version, and the restore target patch must be
equal to or newer than the source patch.

## Create a portable hand-off

Finish ingestion into new versioned collections first. Then create and
download snapshots for both entities:

```bash
python3 scripts/qdrant_snapshot.py create \
  --collection aic2026-frames-siglip2-so400m-v1 \
  --collection aic2026-clips-siglip2-so400m-v1 \
  --feature-profile siglip2-so400m-patch14-384-v1 \
  --output-dir artifacts/qdrant-snapshots/2026-08-12
```

Transfer the entire output directory. It contains one `.snapshot` per
collection plus `snapshot-manifest.json`. The manifest records the Qdrant
version, feature profile, point counts, byte sizes, and SHA-256 checksums.

Do not distribute a live copy of `data/qdrant/storage` as the portable format.
Use collection snapshots so Qdrant creates a consistent artifact while it is
running.

## Restore without embedding again

On the receiving machine, check out the same revision and start its pinned
Qdrant image. Place all transferred files in one directory, then run:

```bash
python3 scripts/qdrant_snapshot.py restore \
  --manifest /path/to/qdrant-snapshots/2026-08-12/snapshot-manifest.json
```

The restore command verifies every checksum and Qdrant version compatibility.
It refuses to overwrite an existing collection. To deploy under new versioned
collection names, map each name explicitly:

```bash
python3 scripts/qdrant_snapshot.py restore \
  --manifest /path/to/snapshot-manifest.json \
  --map aic2026-frames-siglip2-so400m-v1=aic2026-frames-release-002 \
  --map aic2026-clips-siglip2-so400m-v1=aic2026-clips-release-002
```

After restore, set `QDRANT_FRAMES_COLLECTION`, `QDRANT_CLIPS_COLLECTION`, and
`FEATURE_PROFILE` in `.env` to the restored names and the profile recorded in
the snapshot manifest. Restart the API after changing those values.

## What still has to accompany a snapshot

A snapshot eliminates media embedding on the receiving server, but it does
not make the deployment entirely self-contained:

- The API still needs the exact feature profile/model to embed incoming text
  queries into the same vector space. Pre-download or transfer the Hugging
  Face model cache if the server must start without internet access.
- Qdrant snapshots contain point vectors, payload metadata, payload indexes,
  and collection configuration. They do not contain image/video files named
  by payload paths. Transfer the media dataset separately if result rendering
  or playback reads those paths.
- Keep the original Parquet manifests and model/profile revision with the
  release so the collections can be audited or rebuilt.
- Treat a frame snapshot and its clip snapshot as one release. Restore both
  before switching the API configuration to their names.

The snapshot tool uses only the Python standard library; it adds no production
dependency to the backend.
