from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, str]:
    container = request.app.state.container

    return {
        "status": "ready",
        "search_service": type(container.search_service).__name__,
    }