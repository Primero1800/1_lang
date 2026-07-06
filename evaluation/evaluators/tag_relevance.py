import os

from evaluation.config import JUDGE_TEMPERATURE_W1
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, SecretStr

_TAG_DESCRIPTIONS = {
    "behavior": "what the person is doing — actions, activities, habits",
    "appearance": "how the person looks — clothes, grooming, physical traits",
    "age": "age-related observations — young, old, youthful, aged",
    "mood": "emotional state — cheerful, gloomy, stressed, energetic",
    "posture": "body position and stance — sitting, slouching, upright, relaxed",
    "hairstyle": "hair or headwear — cut, style, colour, hat, or absence of hair",
}

_BATCH_PROMPT = """\
Evaluate each observation phrase for tag relevance. Score 1-5:
  1 — completely off-topic
  2 — loosely related
  3 — partially relevant
  4 — mostly on-topic
  5 — perfectly matches the tag

{items}

Return a score for every numbered item.\
"""


class _ScoreItem(BaseModel):
    n: int
    score: int
    reasoning: str


class _BatchScores(BaseModel):
    results: list[_ScoreItem]


_llm = ChatMistralAI(
    model_name=os.environ["MISTRAL_MODEL"],
    api_key=SecretStr(os.environ["MISTRAL_API_KEY"]),
    timeout=int(os.environ.get("MISTRAL_TIMEOUT_SEC", "60")),
    temperature=JUDGE_TEMPERATURE_W1,
).with_structured_output(_BatchScores)


def evaluate_batch(items: list[dict]) -> list[_ScoreItem]:
    """Score a batch of phrases in a single LLM call.

    :param:
        items: list of {"n": int, "phrase": str, "tag": str} — n is 1-based index

    :returns:
        scores: list of _ScoreItem sorted by n
    """
    lines = "\n".join(
        f'{item["n"]}. [{item["tag"]} — {_TAG_DESCRIPTIONS.get(item["tag"], item["tag"])}] "{item["phrase"]}"'
        for item in items
    )
    result: _BatchScores = _llm.invoke(_BATCH_PROMPT.format(items=lines))
    return sorted(result.results, key=lambda x: x.n)
