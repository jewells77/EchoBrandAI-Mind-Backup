from app.infrastructure.vectorstores.pinecone_store import query_embeddings
from app.domain.llm_providers.embedding_factory import (
    get_embedding_provider,
)


class ContentRetrievalPipeline:
    def __init__(
        self,
        embedding_provider_name: str = "huggingface",
        namespace: str = "",
        include_values: bool = False,
    ):
        self.embedding_provider_name = embedding_provider_name
        self.namespace = namespace
        self.include_values = include_values
        self.embedding_provider = get_embedding_provider(embedding_provider_name)

    def retrieve_context(
        self,
        user_query: str,
        top_k: int = 5,
        filter: dict = None,
        fields: list = None,
        include_values: bool = False,
    ):
        """
        Retrieve relevant context chunks from Pinecone for a user query, with optional metadata filtering and custom fields.
        Assumes embeddings and metadata are already upserted.
        """
        # Allow override of provider, namespace, include_values
        provider = self.embedding_provider
        query_emb = provider.get_embedding(user_query)
        query_args = {
            "vector": query_emb,
            "top_k": top_k,
            "include_metadata": True,
            "include_values": (
                include_values if include_values is not None else self.include_values
            ),
        }
        query_args["namespace"] = self.namespace
        if filter:
            query_args["filter"] = filter
        results = query_embeddings(**query_args)
        # Build context from specified fields, default to 'text'
        if fields is None:
            fields = ["text"]
        context_chunks = []
        for r in results.get("matches", []):
            if "metadata" in r:
                chunk = " ".join([str(r["metadata"].get(f, "")) for f in fields])
                context_chunks.append(chunk)
        context = " ".join(context_chunks)
        return context
