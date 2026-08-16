"""Render an accepted candidate list into the file the organiser grades.

Pure functions over the request: no IO, no validation. Everything that could
reject a submission happens in the service before these are called, so a
failure here is a formatting bug and nothing else.

The row shapes are the contract:

    kis     video_id, frame_id
    qa      video_id, frame_id, answer
    trake   video_id, frame_id_1, ..., frame_id_n

No header row. A header would be read as a candidate by a grader that counts
lines, which costs the first — and best — answer of the run.
"""

import csv
import json
from io import StringIO

from app.schemas.submissions import (
    Candidate,
    ExportFormat,
    ExportRequest,
    KisCandidate,
    QaCandidate,
    TrakeCandidate,
)

CSV_MEDIA_TYPE = "text/csv"
JSON_MEDIA_TYPE = "application/json"


def candidate_fields(candidate: Candidate) -> list[str]:
    """One candidate as the ordered strings its row is built from."""
    if isinstance(candidate, KisCandidate):
        return [candidate.video_id, str(candidate.frame_id)]
    if isinstance(candidate, QaCandidate):
        return [candidate.video_id, str(candidate.frame_id), candidate.answer]
    if isinstance(candidate, TrakeCandidate):
        return [candidate.video_id, *(str(f) for f in candidate.event_frame_ids)]
    raise TypeError(f"unsupported candidate type {type(candidate).__name__}")


def to_csv(request: ExportRequest) -> bytes:
    """UTF-8 CSV, LF-terminated, no byte-order mark.

    A BOM would ride along on the first field of the first row, so the best
    answer of the submission would arrive with an invisible prefix on its
    video id and score zero. Excel readability is not worth that.
    """
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for candidate in request.candidates:
        writer.writerow(candidate_fields(candidate))
    return buffer.getvalue().encode("utf-8")


def to_json(request: ExportRequest) -> bytes:
    """Same information as the CSV, addressed by name instead of position.

    Kept field-for-field identical to `candidate_fields` so the two formats
    can never disagree about what was submitted.
    """
    payload = [
        {"task": request.task, "fields": candidate_fields(candidate)}
        for candidate in request.candidates
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def render(request: ExportRequest) -> tuple[bytes, str, str]:
    """Return (content, media_type, filename) for the requested format."""
    if request.format is ExportFormat.json:
        return to_json(request), JSON_MEDIA_TYPE, f"submission-{request.task}.json"
    return to_csv(request), CSV_MEDIA_TYPE, f"submission-{request.task}.csv"
