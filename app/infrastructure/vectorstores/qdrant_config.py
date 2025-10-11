from qdrant_client import AsyncQdrantClient
from app.config import settings


QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_URL = settings.QDRANT_URL

# Module-level global for convenience.
# NOTE: This is **per-process**. In a multi-worker setup, each Gunicorn/Uvicorn worker
# (separate Python process) will have its own _qdrant_client.
_qdrant_client: AsyncQdrantClient | None = None


def init_qdrant():
    """
    Initialize the Qdrant client for this FastAPI worker.

    How it works:
    1. Each worker runs its own startup event, so each process will call this function.
    2. _qdrant_client is set per worker, and stored in app.state.qdrant_client for helpers/routes.
    3. AsyncQdrantClient is just an HTTP client, so having one per worker is safe and expected.

    Why it works:
    - Within a single worker, both _qdrant_client and app.state.qdrant_client point to the same client.
    - Routes and helper functions can safely access app.state.qdrant_client without passing it explicitly.

    Why it can fail / won’t work if misused:
    - **Globals are not shared across workers.** Worker 1’s _qdrant_client is invisible to Worker 2.
    - If you forget to call this function in the worker’s startup/lifespan, app.state.qdrant_client remains None.
      Any route or helper that tries to use it will fail at runtime.
    - Any code assuming a single shared _qdrant_client across workers will break.

    ✅ Best practice:
    - Always call init_qdrant() during FastAPI startup/lifespan events.
    - Access the client via app.state.qdrant_client inside routes or helpers.
    """
    global _qdrant_client
    _qdrant_client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    return _qdrant_client


def get_qdrant_client() -> AsyncQdrantClient:
    if not _qdrant_client:
        raise RuntimeError("Qdrant client not initialized")
    return _qdrant_client
