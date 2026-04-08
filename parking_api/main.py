from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import MODELS_V2_DIR
from .models import ModelRegistry
from .router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry()
    app.state.registry_v2 = ModelRegistry(MODELS_V2_DIR)
    yield


app = FastAPI(title="UNCC Parking Prediction API", lifespan=lifespan)
app.include_router(router)
