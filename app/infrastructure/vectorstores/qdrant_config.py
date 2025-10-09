import time
from qdrant_client import QdrantClient
from app.config import settings

QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_URL = settings.QDRANT_URL

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
