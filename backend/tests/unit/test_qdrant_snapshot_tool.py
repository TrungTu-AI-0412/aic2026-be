from pathlib import Path

import pytest

from scripts.qdrant_snapshot import (
    SnapshotToolError,
    _manifest_artifact_path,
    _minor_version,
    _parse_mappings,
    _snapshot_filename,
    _version_tuple,
)


def test_snapshot_filename_does_not_trust_collection_as_path() -> None:
    filename = _snapshot_filename("../../frames/release-1")

    assert "/" not in filename
    assert "\\" not in filename
    assert filename.endswith(".snapshot")


def test_manifest_artifact_must_be_next_to_manifest(tmp_path: Path) -> None:
    assert _manifest_artifact_path(tmp_path, "frames.snapshot").parent == tmp_path

    with pytest.raises(SnapshotToolError, match="next to the manifest"):
        _manifest_artifact_path(tmp_path, "../frames.snapshot")


def test_qdrant_version_compatibility_uses_major_and_minor() -> None:
    assert _minor_version("v1.12.4") == (1, 12)
    assert _minor_version("1.12.9") == (1, 12)
    assert _version_tuple("1.12.9-dev") == (1, 12, 9)


def test_collection_name_mapping() -> None:
    assert _parse_mappings(["frames-v1=frames-v2", "clips-v1=clips-v2"]) == {
        "frames-v1": "frames-v2",
        "clips-v1": "clips-v2",
    }

    with pytest.raises(SnapshotToolError, match="SOURCE=TARGET"):
        _parse_mappings(["frames-v1"])
