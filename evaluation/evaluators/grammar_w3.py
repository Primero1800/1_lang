import os

from evaluation.config import JUDGE_TEMPERATURE_W3
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, SecretStr

_TONES = ["A", "B", "C", "D", "E"]

_BATCH_PROMPT = """\
Evaluate the grammatical correctness and fluency of AI-generated comment variants.
Each entry includes an original observation, its language, and a sample of generated comments (one per tone/gender).
All comments address the person as "you" (second person).

Score grammar_quality 1-5:
  5 — all variants are grammatically correct and fluent in the given language
  4 — minor issues in 1-2 variants (awkward phrasing, small errors)
  3 — noticeable errors in several variants
  2 — significant grammar issues throughout
  1 — variants are mostly ungrammatical or in the wrong language

{items}

Return a score and one reasoning string per numbered entry.\
"""


class _PhraseResult(BaseModel):
    n: int
    grammar_quality: int
    reasoning: str


class _BatchScores(BaseModel):
    results: list[_PhraseResult]


_llm = ChatMistralAI(
    model_name=os.environ["MISTRAL_MODEL"],
    api_key=SecretStr(os.environ["MISTRAL_API_KEY"]),
    timeout=int(os.environ.get("MISTRAL_TIMEOUT_SEC", "60")),
    temperature=JUDGE_TEMPERATURE_W3,
).with_structured_output(_BatchScores)


def _format_item(n: int, phrase: str, tag: str, lang: str, variants: dict) -> str:
    lines = [
        f'{n}. Original: "{phrase}" [tag: {tag}, lang: {lang}]',
        "   Variants (first per tone/gender):",
    ]
    for tone in _TONES:
        tone_data = variants.get(tone, {})
        male = (tone_data.get("male") or ["—"])[0]
        female = (tone_data.get("female") or ["—"])[0]
        lines.append(f'   {tone}  male: "{male}"  |  female: "{female}"')
    return "\n".join(lines)


def evaluate_batch(items: list[dict]) -> list[_PhraseResult]:
    """Score grammar quality of a batch of phrase+variants in a single LLM call.

    :param:
        items: list of {"n": int, "phrase": str, "tag": str, "lang": str, "variants": dict}

    :returns:
        scores: list of _PhraseResult sorted by n
    """
    formatted = "\n\n".join(
        _format_item(
            item["n"], item["phrase"], item["tag"], item["lang"], item["variants"]
        )
        for item in items
    )
    result: _BatchScores = _llm.invoke(_BATCH_PROMPT.format(items=formatted))
    return sorted(result.results, key=lambda x: x.n)
