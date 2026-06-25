from typing import Literal

from langchain_mistralai import MistralAIEmbeddings
from langsmith import traceable

from app.common.logging import logger


class TrackedMistralEmbeddings(MistralAIEmbeddings):
    """MistralAIEmbeddings subclass that captures token usage and publishes it to Redis Streams"""

    async def aembed_with_usage(
        self,
        texts: list[str],
        input_type: Literal["search_document", "search_query"] = "search_document",
    ) -> tuple[list[list[float]], int]:
        """Embed texts and return vectors alongside input token count

        :param:
            texts: list of strings to embed
            input_type: E5-style prefix strategy — 'search_document' for W4, 'search_query' for T1

        :returns:
            vectors: list of embedding vectors
            input_tokens: number of tokens consumed
        """
        prefixed = [f"{input_type}: {t}" for t in texts]
        batches = list(self._get_batches(prefixed))
        vectors: list[list[float]] = []
        result = await self._traced_embed(batches, vectors)
        total_tokens: int = result["llm_output"]["token_usage"]["prompt_tokens"]
        if not total_tokens:
            logger.warning(
                "[embed] embedding response missing usage metadata, tokens not tracked"
            )
        return vectors, total_tokens

    @traceable(run_type="llm", name="mistral_embed")
    async def _traced_embed(
        self,
        batches: list[list[str]],
        vectors_out: list[list[float]],
    ) -> dict[str, dict]:
        """Call Mistral embedding API and collect vectors and token usage.

        Vectors are written into vectors_out (not returned) so that @traceable
        logs only the token_usage dict to LangSmith, not the raw embedding floats.

        :param:
            batches: pre-split lists of prefixed texts
            vectors_out: accumulator filled in place with embedding vectors

        :returns:
            LangSmith-compatible llm_output dict with token_usage
        """
        # async_client is an internal httpx.AsyncClient of MistralAIEmbeddings —
        # accessing it directly to capture usage metadata not exposed via public API
        total_tokens: int = 0
        for batch in batches:
            response = await self.async_client.post(
                url="/embeddings",
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            data = response.json()
            vectors_out.extend(
                list(map(float, item["embedding"])) for item in data["data"]
            )
            usage = data.get("usage") or {}
            total_tokens += usage.get("prompt_tokens", 0)
        return {
            "llm_output": {
                "token_usage": {
                    "prompt_tokens": total_tokens,
                    "completion_tokens": 0,
                    "total_tokens": total_tokens,
                }
            }
        }
