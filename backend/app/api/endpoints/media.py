from pydantic import ValidationError

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.responses import FileResponse

from app.api.deps import get_media_service
from app.schemas.media import ClipRequest, FrameContext, VideoTimeline
from app.services.media import FrameNotFoundError, MediaService, VideoNotFoundError

router = APIRouter()


@router.get("/{video_id}/frames/{frame_id}")
async def get_frame(
    video_id: str,
    frame_id: int,
    media_service: MediaService = Depends(get_media_service),
) -> Response:
    try:
        frame = await media_service.get_frame(video_id, frame_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FrameNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Keyframes are content-addressed by (video_id, frame_id) and never
    # rewritten, so a timeline hover-scrub re-shows them from cache.
    return Response(
        content=frame.content,
        media_type=frame.media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{video_id}/frames/{frame_id}/context")
async def get_frame_context(
    video_id: str,
    frame_id: int,
    radius: int = Query(default=5, ge=0, le=25, description="Neighbouring keyframes each side."),
    media_service: MediaService = Depends(get_media_service),
) -> FrameContext:
    """Metadata for one keyframe plus its neighbouring keyframes.

    One request per verify panel: the neighbour thumbnails are then plain
    `/frames/{frame_id}` reads and the clip preview is `/stream#t=pts_sec`.
    """
    try:
        return await media_service.get_frame_context(video_id, frame_id, radius)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FrameNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str,
    media_service: MediaService = Depends(get_media_service),
) -> FileResponse:
    """Stream the source video for in-place playback.

    Seek on the client: `<video src=".../stream#t=12.5">`. FileResponse honours
    Range requests, so the browser pulls only the bytes around that timestamp
    instead of the file, and nothing is decoded or re-encoded server side.
    """
    try:
        path = await media_service.get_video_path(video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(path)


@router.get("/{video_id}/timeline")
async def get_video_timeline(
    video_id: str,
    media_service: MediaService = Depends(get_media_service),
) -> VideoTimeline:
    """Metadata and sampled-frame markers for the full verification studio."""
    try:
        return await media_service.get_video_timeline(video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/{video_id}/source-frames/{frame_id}",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def get_source_frame(
    video_id: str,
    frame_id: int = Path(ge=0),
    media_service: MediaService = Depends(get_media_service),
) -> Response:
    """Decode one exact original-video frame for submission verification."""
    try:
        frame = await media_service.get_source_frame(video_id, frame_id)
    except (VideoNotFoundError, FrameNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=frame.content,
        media_type=frame.media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{video_id}/clip")
async def get_clip(
    video_id: str,
    start_frame: int | None = Query(default=None, ge=0),
    end_frame: int | None = Query(default=None, ge=0),
    center_frame: int | None = Query(default=None, ge=0),
    radius: int | None = Query(default=None, ge=0),
    media_service: MediaService = Depends(get_media_service),
) -> Response:
    try:
        clip_request = ClipRequest(
            start_frame=start_frame,
            end_frame=end_frame,
            center_frame=center_frame,
            radius=radius,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    start, end = clip_request.frame_range

    try:
        clip = await media_service.get_clip(video_id, start, end)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FrameNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(content=clip.content, media_type=clip.media_type)
