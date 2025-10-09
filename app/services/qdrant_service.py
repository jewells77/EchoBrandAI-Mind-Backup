from app.infrastructure.vectorstores import qdrant_store
from app.api.exceptions import APIError


class QdrantService:
    @staticmethod
    def create_user_collection(user_id: str):
        return qdrant_store.create_collection(user_id)

    @staticmethod
    def check_user_collection_exists(user_id: str) -> bool:
        return qdrant_store.collection_exists(user_id)
