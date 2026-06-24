from langchain_mistralai import MistralAIEmbeddings

from app.common.logging import logger


class TrackedMistralEmbeddings(MistralAIEmbeddings):
    """MistralAIEmbeddings subclass that captures token usage and publishes it to Redis Streams"""

    async def aembed_with_usage(
        self, texts: list[str]
    ) -> tuple[list[list[float]], int]:
        """Embed texts and return vectors alongside input token count

        :param:
            texts: list of strings to embed

        :returns:
            vectors: list of embedding vectors
            input_tokens: number of tokens consumed
        """
        # async_client is an internal httpx.AsyncClient of MistralAIEmbeddings —
        # accessing it directly to capture usage metadata not exposed via public API
        prefixed = [f"search_document: {t}" for t in texts]
        all_vectors: list[list[float]] = []
        total_tokens: int = 0
        for batch in self._get_batches(prefixed):
            response = await self.async_client.post(
                url="/embeddings",
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            data = response.json()
            all_vectors.extend(
                list(map(float, item["embedding"])) for item in data["data"]
            )
            usage = data.get("usage") or {}
            total_tokens += usage.get("prompt_tokens", 0)
        if not total_tokens:
            logger.warning(
                "[W4] embedding response missing usage metadata, tokens not tracked"
            )
        return all_vectors, total_tokens
