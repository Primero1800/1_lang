"""
Run W2 evaluation experiment in LangSmith.
Run: poetry run python -m evaluation.run_w2

Pre-computes variants-quality scores for all examples via batched LLM calls,
then logs per-example results to LangSmith via aevaluate().
"""

import asyncio

from langsmith import Client
from langsmith.evaluation import aevaluate

from evaluation.config import (
    DATASET_NAME_W2,
    EVAL_BATCH_SIZE_W2,
)  # triggers load_dotenv()
from evaluation.evaluators.variants_quality_w2 import evaluate_batch


def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


async def main() -> None:
    client = Client()
    dataset = client.read_dataset(dataset_name=DATASET_NAME_W2)
    examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"Loaded {len(examples)} examples from '{DATASET_NAME_W2}'")
    if not examples:
        print("Dataset is empty — run build_dataset_w2.py first.")
        return

    # Pre-compute all scores in batches before aevaluate()
    scores_map: dict[str, dict] = {}
    for batch_num, batch in enumerate(_chunks(examples, EVAL_BATCH_SIZE_W2), start=1):
        items = [
            {
                "n": j + 1,
                "phrase": ex.inputs["phrase"],
                "tag": ex.inputs["tag"],
                "variants": ex.inputs["variants"],
            }
            for j, ex in enumerate(batch)
        ]
        print(
            f"  batch {batch_num}: scoring {len(items)} phrases...", end=" ", flush=True
        )
        scores_by_n = {s.n: s for s in evaluate_batch(items)}
        for j, ex in enumerate(batch):
            score = scores_by_n.get(j + 1)
            if score is None:
                print(f"\n  WARNING: no score returned for item {j + 1}")
                continue
            scores_map[ex.inputs["phrase"]] = {
                "gender_match": score.gender_match,
                "tone_gradient": score.tone_gradient,
                "coherence": score.coherence,
                "reasoning": score.reasoning,
            }
        print("done")

    async def target(inputs: dict) -> dict:
        return scores_map.get(inputs["phrase"], {})

    def gender_match(outputs: dict) -> dict:
        return {"key": "gender_match", "score": outputs.get("gender_match", 0)}

    def tone_gradient(outputs: dict) -> dict:
        return {"key": "tone_gradient", "score": outputs.get("tone_gradient", 0)}

    def coherence(outputs: dict) -> dict:
        return {"key": "coherence", "score": outputs.get("coherence", 0)}

    print("Running evaluation...")
    await aevaluate(
        target,
        data=examples,
        evaluators=[gender_match, tone_gradient, coherence],
        experiment_prefix=DATASET_NAME_W2,
        client=client,
    )
    print("Done. View results in LangSmith.")


if __name__ == "__main__":
    asyncio.run(main())
