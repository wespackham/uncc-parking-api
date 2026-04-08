from contextlib import asynccontextmanager

from fastapi import FastAPI

from .models import ModelRegistry
from .router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry()
    yield


app = FastAPI(title="UNCC Parking Prediction API", lifespan=lifespan)
app.include_router(router)
