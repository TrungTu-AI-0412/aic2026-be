from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.runtime.container import build_container


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    container = await build_container(settings)

    application.state.container = container

    try:
        yield
    finally:
        await container.close()