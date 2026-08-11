from fastapi import APIRouter

from app.api.endpoints import (
    health,
    internal_ingestions,
    media,
    search,
    submissions,
)

router = APIRouter()

router.include_router(health.router, tags=["Health"])
router.include_router(search.router, tags=["Search"])
router.include_router(media.router, prefix="/videos", tags=["Media"])
router.include_router(
    submissions.router,
    prefix="/submissions",
    tags=["Submissions"],
)
router.include_router(
    internal_ingestions.router,
    prefix="/ingestions",
    tags=["Ingestion"],
)