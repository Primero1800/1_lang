"""
Build W3 evaluation dataset in LangSmith.
Run: poetry run python -m evaluation.build_dataset_w3

Samples EVAL_SAMPLE_SIZE_W3 phrases per language (ru + en) with phrase_data,
uploads to LangSmith dataset for grammar correctness evaluation.

EVAL_SAMPLE_SIZE_W3 env var controls sample size per language (default 8).
"""

import asyncio
import os

from langsmith import Client
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.repositories.phrase_repository import PhraseRepository
from evaluation.config import DATASET_NAME_W3, EVAL_SAMPLE_SIZE_W3

_user = os.environ["POSTGRES_USER"]
_password = os.environ["POSTGRES_PASSWORD"]
_db = os.environ["POSTGRES_DB"]
_DATABASE_URL = f"postgresql+asyncpg://{_user}:{_password}@localhost:5433/{_db}"

_LANGS = ("ru", "en")


async def _sample_phrases():
    engine = create_async_engine(_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            repo = PhraseRepository(session)
            phrases = []
            for lang in _LANGS:
                batch = await repo.get_sample_per_tag(
                    sample_size=EVAL_SAMPLE_SIZE_W3 * 3, load_data=True, lang=lang
                )
                phrases += batch[:EVAL_SAMPLE_SIZE_W3]
            return phrases
    finally:
        await engine.dispose()


def _upload_to_langsmith(phrases, dataset_name: str) -> None:
    client = Client()

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Sample of W3 phrase variants per language for grammar evaluation",
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
            "variants": p.phrase_data.variants,
        }
        for p in phrases
    ]
    client.create_examples(inputs=inputs, dataset_id=dataset.id)
    print(f"Uploaded {len(inputs)} examples to '{dataset_name}'")


async def main() -> None:
    phrases = await _sample_phrases()
    by_lang = {}
    for p in phrases:
        lang = p.lang.value if hasattr(p.lang, "value") else p.lang
        by_lang[lang] = by_lang.get(lang, 0) + 1
    print(f"Sampled {len(phrases)} phrases: {by_lang}")
    if not phrases:
        print("No LOADING_DONE phrases found — run the full pipeline first.")
        return
    _upload_to_langsmith(phrases, DATASET_NAME_W3)


if __name__ == "__main__":
    asyncio.run(main())
