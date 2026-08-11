from fastapi import APIRouter, Depends, HTTPException, Response

from app.api.deps import get_submission_service
from app.schemas.submissions import ExportRequest
from app.services.submissions import (
    FrameOutOfBoundsError,
    SubmissionService,
    VideoNotFoundError,
)

router = APIRouter()


@router.post("/export")
async def export_submission(
    request: ExportRequest,
    submission_service: SubmissionService = Depends(get_submission_service),
) -> Response:
    try:
        export_file = await submission_service.export(request)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FrameOutOfBoundsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(
        content=export_file.content,
        media_type=export_file.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{export_file.filename}"'
        },
    )
