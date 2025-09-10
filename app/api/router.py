from fastapi import APIRouter
from app.api.v1.endpoints import competitors, health, chat

api_router = APIRouter()

# Health check
api_router.include_router(health.router, prefix="/health", tags=["health"])

# API v1 routes
api_router.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
api_router.include_router(
    competitors.router, prefix="/v1/competitors", tags=["competitors"]
)
