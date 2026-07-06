"""
Build W1 evaluation dataset in LangSmith.
Run: poetry run python -m evaluation.build_dataset_w1

Samples EVAL_SAMPLE_SIZE phrases from LOADING_DONE, uploads to LangSmith dataset.
Re-running appends; reset the dataset in LangSmith UI if you need a clean slate.

EVAL_SAMPLE_SIZE env var controls total sample size (default 96).
"""

import asyncio
import os

from evaluation.config import DATASET_NAME_W1, EVAL_SAMPLE_SIZE
from langsmith import Client
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.repositories.phrase_repository import PhraseRepository

_user = os.environ["POSTGRES_USER"]
_password = os.environ["POSTGRES_PASSWORD"]
_db = os.environ["POSTGRES_DB"]
_DATABASE_URL = f"postgresql+asyncpg://{_user}:{_password}@localhost:5433/{_db}"


async def _sample_phrases():
    engine = create_async_engine(_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            repo = PhraseRepository(session)
            return await repo.get_sample_per_tag(sample_size=EVAL_SAMPLE_SIZE)
    finally:
        await engine.dispose()


def _upload_to_langsmith(phrases, dataset_name: str) -> None:
    client = Client()

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Stratified RU sample of W1 vision phrases for LLM-as-judge evaluation",
        )
        print(f"Created dataset '{dataset_name}' (id={dataset.id})")
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset '{dataset_name}' (id={dataset.id})")

    inputs = [
        {
            "phrase_id": p.id,
            "phrase": p.original,
            "tag": p.tag,
            "lang": p.lang.value,
        }
        for p in phrases
    ]
    client.create_examples(inputs=inputs, dataset_id=dataset.id)
    print(f"Uploaded {len(inputs)} examples to '{dataset_name}'")


async def main() -> None:
    phrases = await _sample_phrases()
    n_tags = len(set(p.tag for p in phrases))
    print(f"Sampled {len(phrases)} phrases ({n_tags} tags)")
    if not phrases:
        print("No LOADING_DONE RU phrases found — run the full pipeline first.")
        return
    _upload_to_langsmith(phrases, DATASET_NAME_W1)


if __name__ == "__main__":
    asyncio.run(main())
