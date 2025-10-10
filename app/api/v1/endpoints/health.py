from fastapi import APIRouter

from app.infrastructure.db.mongodb import get_database

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
