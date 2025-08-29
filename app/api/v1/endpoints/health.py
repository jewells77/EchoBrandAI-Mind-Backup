from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorClient

from app.infrastructure.db.mongodb import get_database
from app.domain.llm_providers.base import BaseLLMProvider
from app.api.deps import get_llm_provider

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Basic health check endpoint.
    Returns status OK if the API is running.
    """
    return {"status": "ok", "message": "API is running"}


@router.get("/db")
async def db_health():
    """
    Check database connection health.
    Returns database status and information.
    """
    try:
        db = get_database()
        server_info = await db.command("serverStatus")
        return {
            "status": "ok",
            "message": "Database connection successful",
            "version": server_info.get("version", "unknown"),
        }
    except Exception as e:
        return {"status": "error", "message": f"Database connection failed: {str(e)}"}


@router.get("/llm")
async def llm_health(llm_provider: BaseLLMProvider = Depends(get_llm_provider)):
    """
    Check LLM provider health.
    Returns LLM provider status and information.
    """
    try:
        provider_info = llm_provider.get_info()
        return {
            "status": "ok",
            "message": "LLM provider connection successful",
            "provider": provider_info,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"LLM provider connection failed: {str(e)}",
        }
