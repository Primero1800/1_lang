"""
Build T1 evaluation dataset in LangSmith.
Run: poetry run python -m evaluation.build_dataset_t1

Generates OBSERVATIONS_PER_TAG_T1 synthetic T1-style observations per tag
(6 tags × OBSERVATIONS_PER_TAG_T1 = 30 total by default) using Mistral in
a single request, then uploads them to LangSmith.

OBSERVATIONS_PER_TAG_T1 env var controls per-tag count (default 5).
EVAL_LANG_T1 env var selects language: 'ru' or 'en' (default 'ru').
"""

from langsmith import Client
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field, SecretStr

from evaluation.config import DATASET_NAME_T1, EVAL_LANG_T1, OBSERVATIONS_PER_TAG_T1
from app.core.config import settings


class T1ObservationSet(BaseModel):
    """Synthetic T1-style observations grouped by tag — one batch per Mistral call"""

    behavior: list[str] = Field(description="Observations about behaviour")
    appearance: list[str] = Field(description="Observations about appearance")
    age: list[str] = Field(description="Observations about age")
    mood: list[str] = Field(description="Observations about mood")
    posture: list[str] = Field(description="Observations about posture")
    hairstyle: list[str] = Field(description="Observations about hairstyle")


_PROMPT_TEMPLATES: dict[str, str] = {
    "ru": (
        "Ты генерируешь синтетические наблюдения о человеке для тестирования векторного поиска.\n"
        "Для каждой из 6 категорий сгенерируй ровно {n} наблюдений на русском языке.\n"
        "Каждое наблюдение — два фрагмента через точку и пробел:\n"
        "  1. конкретное: что именно видно (5-6 слов, нижний регистр, без знаков препинания в конце)\n"
        "  2. образное: метафора или обобщение (5-6 слов, нижний регистр, без знаков препинания в конце)\n"
        'Пример для posture: "горбится над экраном весь день. человек-вопросительный знак"\n'
        "Все {n} наблюдений внутри одного тега должны быть разными."
    ),
    "en": (
        "You generate synthetic person observations for vector retrieval testing.\n"
        "For each of the 6 categories generate exactly {n} observations in English.\n"
        "Each observation is two phrases joined by a period and space:\n"
        "  1. concrete: what is literally visible (5-6 words, lowercase, no trailing punctuation)\n"
        "  2. abstract: a metaphor or generalisation (5-6 words, lowercase, no trailing punctuation)\n"
        'Example for posture: "hunched over the laptop all day. a human question mark"\n'
        "All {n} observations within a tag must be distinct."
    ),
}


def _generate_observations(lang: str, per_tag: int) -> T1ObservationSet:
    llm = ChatMistralAI(
        model_name=settings.MISTRAL_MODEL,
        api_key=SecretStr(settings.MISTRAL_API_KEY),
        temperature=0,
    ).with_structured_output(T1ObservationSet)
    prompt = _PROMPT_TEMPLATES[lang].format(n=per_tag)
    result = llm.invoke(prompt)
    return result  # type: ignore[return-value]


def _upload_to_langsmith(
    observation_set: T1ObservationSet, lang: str, dataset_name: str
) -> None:
    client = Client()
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Synthetic T1-style observations per tag for retrieval relevancy evaluation",
        )
        print(f"Created dataset '{dataset_name}' (id={dataset.id})")
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset '{dataset_name}' (id={dataset.id})")

    data = observation_set.model_dump()
    inputs = [
        {"observation": obs, "tag": tag, "lang": lang}
        for tag, observations in data.items()
        for obs in observations
    ]
    client.create_examples(inputs=inputs, dataset_id=dataset.id)
    print(f"Uploaded {len(inputs)} examples to '{dataset_name}'")


def main() -> None:
    lang = EVAL_LANG_T1
    per_tag = OBSERVATIONS_PER_TAG_T1
    total = per_tag * 6
    print(f"Generating {per_tag} observations × 6 tags = {total} total (lang={lang})")
    observation_set = _generate_observations(lang=lang, per_tag=per_tag)
    for tag, observations in observation_set.model_dump().items():
        print(f"  {tag}: {len(observations)}")
    _upload_to_langsmith(observation_set, lang=lang, dataset_name=DATASET_NAME_T1)


if __name__ == "__main__":
    main()
