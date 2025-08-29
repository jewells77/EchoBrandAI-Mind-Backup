from fastapi import APIRouter
from app.api.v1.endpoints import content, creators, competitors, health

api_router = APIRouter()

# Health check
api_router.include_router(health.router, prefix="/health", tags=["health"])

# API v1 routes
api_router.include_router(content.router, prefix="/v1/content", tags=["content"])
# api_router.include_router(creators.router, prefix="/v1/creators", tags=["creators"])
# api_router.include_router(
#     competitors.router, prefix="/v1/competitors", tags=["competitors"]
# )
