"""
Run T1 evaluation experiment in LangSmith.
Run: poetry run python -m evaluation.run_t1

Embeds all dataset observations in one batch, then searches Qdrant per example
via client.aevaluate(). Records retrieval_score (Qdrant similarity) to LangSmith.
"""

import asyncio

from langchain_mistralai import MistralAIEmbeddings
from langsmith.evaluation import aevaluate
from langsmith import Client
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, QueryRequest

from evaluation.config import DATASET_NAME_T1  # triggers load_dotenv()
from app.core.config import settings


async def main() -> None:
    client = Client()

    # 1. Load all examples
    dataset = client.read_dataset(dataset_name=DATASET_NAME_T1)
    examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"Loaded {len(examples)} examples from '{DATASET_NAME_T1}'")
    if not examples:
        print("Dataset is empty — run build_dataset_t1.py first.")
        return

    # 2. One batch embed call — all observations at once
    embeddings = MistralAIEmbeddings(
        model=settings.MISTRAL_EMBED_MODEL,
        api_key=SecretStr(settings.MISTRAL_API_KEY),
    )
    observations = [ex.inputs["observation"] for ex in examples]
    prefixed = [f"search_query: {obs}" for obs in observations]
    print(f"Embedding {len(prefixed)} observations in one batch...")
    vectors = await embeddings.aembed_documents(prefixed)
    vector_map = {obs: vec for obs, vec in zip(observations, vectors)}
    print("Done.")

    # 3. One batch Qdrant call for all examples
    qdrant = AsyncQdrantClient(
        url=settings.QDRANT_MAIN_URL,
        api_key=settings.QDRANT_MAIN_API_KEY,
        check_compatibility=False,
        timeout=settings.QDRANT_TIMEOUT,
    )
    requests = [
        QueryRequest(
            query=vector_map[ex.inputs["observation"]],
            filter=Filter(
                must=[
                    FieldCondition(
                        key="lang", match=MatchValue(value=ex.inputs["lang"])
                    ),
                    FieldCondition(key="tag", match=MatchValue(value=ex.inputs["tag"])),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vector=False,
        )
        for ex in examples
    ]
    print(f"Searching Qdrant for {len(requests)} observations in one batch...")
    responses = await qdrant.query_batch_points(
        collection_name=settings.VECTOR_DB_COLLECTION,
        requests=requests,
    )
    batch_results = [r.points for r in responses]
    await qdrant.close()

    results_map: dict[str, dict] = {}
    for ex, points in zip(examples, batch_results):
        obs = ex.inputs["observation"]
        if points:
            results_map[obs] = {
                "retrieved_phrase": points[0].payload.get("original", ""),
                "score": points[0].score,
            }
        else:
            results_map[obs] = {"retrieved_phrase": None, "score": 0.0}
    print("Done.")

    # 5. Target: pure dict lookup, no I/O
    async def target(inputs: dict) -> dict:
        return results_map[inputs["observation"]]

    # 6. Evaluator: reads Qdrant score, no LLM
    def retrieval_score(outputs: dict) -> dict:
        return {"key": "retrieval_score", "score": outputs.get("score", 0.0)}

    # 7. Run
    print("Running evaluation...")
    await aevaluate(
        target,
        data=DATASET_NAME_T1,
        evaluators=[retrieval_score],
        experiment_prefix=DATASET_NAME_T1,
        client=client,
    )
    print("Done. View results in LangSmith.")


if __name__ == "__main__":
    asyncio.run(main())
