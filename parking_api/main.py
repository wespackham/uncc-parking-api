from fastapi import FastAPI

from .router import router

app = FastAPI(title="UNCC Parking Prediction API")
app.include_router(router)

# CI/CD push
