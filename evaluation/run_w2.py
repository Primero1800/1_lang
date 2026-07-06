"""
Run W2 evaluation experiment in LangSmith.
Run: poetry run python -m evaluation.run_w2

Fetches examples from the dataset, batches them (EVAL_BATCH_SIZE_W2 per LLM call),
scores each batch via variants_quality_w2 evaluator, logs results to LangSmith.
"""

import uuid
from datetime import datetime, timezone

from langsmith import Client

from evaluation.config import DATASET_NAME_W2, EVAL_BATCH_SIZE_W2
from evaluation.evaluators.variants_quality_w2 import evaluate_batch

_SCORE_KEYS = ("gender_match", "tone_gradient", "coherence")


def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


if __name__ == "__main__":
    client = Client()
    dataset = client.read_dataset(dataset_name=DATASET_NAME_W2)
    examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"Loaded {len(examples)} examples from '{DATASET_NAME_W2}'")

    experiment_name = f"w2-variants-quality-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    client.create_project(experiment_name, reference_dataset_id=str(dataset.id))
    print(f"Experiment: '{experiment_name}'")

    total = 0
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
        scores = evaluate_batch(items)
        scores_by_n = {s.n: s for s in scores}

        now = datetime.now(timezone.utc)
        for j, ex in enumerate(batch):
            score = scores_by_n.get(j + 1)
            if score is None:
                print(f"\n  WARNING: no score for item {j + 1}")
                continue
            run_id = str(uuid.uuid4())
            client.create_run(
                id=run_id,
                name="target",
                run_type="chain",
                inputs=ex.inputs,
                outputs={"tag": ex.inputs["tag"], "reasoning": score.reasoning},
                project_name=experiment_name,
                reference_example_id=str(ex.id),
                start_time=now,
                end_time=now,
            )
            for key in _SCORE_KEYS:
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=getattr(score, key),
                    comment=score.reasoning,
                )
        total += len(batch)
        print(f"done ({total}/{len(examples)})")

    print("Complete. View at: https://smith.langchain.com")
