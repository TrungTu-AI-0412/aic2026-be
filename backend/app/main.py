from fastapi import FastAPI

from app.api.router import router as api_router
from app.core.lifespan import lifespan
from app.core.config import settings

def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.include_router(api_router, prefix=settings.API_V1_STR)

    return application


app = create_app()
