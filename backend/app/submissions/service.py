"""Turn an operator's chosen candidates into a gradeable submission file.

The competition allows a small number of attempts per query, so the expensive
mistake is spending one on a row that could never have scored: a video id that
is not in the corpus, or a frame index past the end of its video. Both are
cheap to catch here and impossible to take back afterwards.

The frame bound is deliberately generous. Rejecting a valid answer costs a
real point; accepting one frame too many costs nothing, because the grader is
the authority either way. See `load_bounds` for why the two are not symmetric.
"""

from asyncio import to_thread
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.schemas.submissions import ExportRequest, TrakeCandidate
from app.services.submissions import (
    ExportFile,
    FrameOutOfBoundsError,
    VideoNotFoundError,
)
from app.submissions import formats


@lru_cache(maxsize=1)
def load_bounds(manifest_path: str) -> dict[str, int]:
    """Map each video id to one past its last submittable frame index.

    Returns an empty map when the manifest is absent, which the service reads
    as "bounds unknown" and skips the check rather than rejecting everything.

    Loaded once per process, like the video manifest in the media service:
    restart the API after rebuilding it.
    """
    if not Path(manifest_path).is_file():
        return {}

    import pyarrow.parquet as pq

    table = pq.read_table(manifest_path, columns=["video_id", "frame_upper_bound"])
    return {
        video_id: int(bound)
        for video_id, bound in zip(
            table.column("video_id").to_pylist(),
            table.column("frame_upper_bound").to_pylist(),
        )
    }


@dataclass
class LocalSubmissionService:
    """Validates against a local bounds manifest, then renders the file."""

    bounds_manifest: str

    async def export(self, request: ExportRequest) -> ExportFile:
        bounds = await to_thread(load_bounds, self.bounds_manifest)
        self._validate(request, bounds)
        content, media_type, filename = formats.render(request)
        return ExportFile(content=content, media_type=media_type, filename=filename)

    def _validate(self, request: ExportRequest, bounds: dict[str, int]) -> None:
        # An empty map means no manifest was built, not that the corpus is
        # empty. Treating it as the latter would fail every export.
        if not bounds:
            return

        for position, candidate in enumerate(request.candidates, start=1):
            upper = bounds.get(candidate.video_id)
            if upper is None:
                raise VideoNotFoundError(
                    f"candidate {position}: video '{candidate.video_id}' is not "
                    f"in {self.bounds_manifest}"
                )

            if isinstance(candidate, TrakeCandidate):
                frame_ids = candidate.event_frame_ids
            else:
                frame_ids = [candidate.frame_id]

            for frame_id in frame_ids:
                if frame_id >= upper:
                    raise FrameOutOfBoundsError(
                        f"candidate {position}: frame {frame_id} is past the end "
                        f"of '{candidate.video_id}' ({upper} frames)"
                    )
