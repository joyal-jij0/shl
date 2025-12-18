import logging 
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from app.schemas import ApiError
from app.api.v1.endpoints import health 

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__) 

app = FastAPI(
    title="SHL Assignment Backend API Server",
    description="BackEnd API server",
    version="0.0.1",
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error = ApiError(
        status_code=500,
        message=str(exc)
    )
    return error.to_response()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
