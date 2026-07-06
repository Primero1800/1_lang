import os

from evaluation.config import JUDGE_TEMPERATURE_W2
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, SecretStr

_TONES = ["A", "B", "C", "D", "E"]
_TONE_LABELS = {
    "A": "cynically (harsh)",
    "B": "bluntly (dry, honest)",
    "C": "normally (neutral)",
    "D": "complimentary (gentle praise)",
    "E": "enthusiastically (max positive)",
}

_BATCH_PROMPT = """\
Evaluate AI-generated comment variants for a person-observation system.
Each numbered entry has an original observation and a sample of generated comments (one per tone/gender).
Score 1-5 on each criterion:

  gender_match  — comments always address the person as "you" (second person), never third person.
                  In Russian, second-person verbs conjugate by gender (застыл/застыла) — check those endings.
                  In English, second-person "you/your" is gender-neutral by nature — identical male/female variants are correct, score 5.
                  Ignore any gender mentioned in the original observation — only judge the grammar of the variants themselves.
  tone_gradient — A is the harshest/most cynical, E is the most enthusiastic; gradient A→E is preserved.
                  Score 5 if the step from A to E is clearly progressive, 1 if tones are random or inverted.
  coherence     — comments logically follow from / comment on the original observation.
                  Do NOT penalise for gender mismatch between the variant and any person mentioned in the original.

{items}

Return a score and one reasoning string per numbered entry.\
"""


class _PhraseResult(BaseModel):
    n: int
    tone_gradient: int
    coherence: int
    gender_match: int
    reasoning: str


class _BatchScores(BaseModel):
    results: list[_PhraseResult]


_llm = ChatMistralAI(
    model_name=os.environ["MISTRAL_MODEL"],
    api_key=SecretStr(os.environ["MISTRAL_API_KEY"]),
    timeout=int(os.environ.get("MISTRAL_TIMEOUT_SEC", "60")),
    temperature=JUDGE_TEMPERATURE_W2,
).with_structured_output(_BatchScores)


def _format_item(n: int, phrase: str, tag: str, variants: dict) -> str:
    lines = [
        f'{n}. Original: "{phrase}" [tag: {tag}]',
        "   Variants (first per tone/gender):",
    ]
    for tone in _TONES:
        tone_data = variants.get(tone, {})
        male = (tone_data.get("male") or ["—"])[0]
        female = (tone_data.get("female") or ["—"])[0]
        lines.append(
            f'   {tone} [{_TONE_LABELS[tone]}]  male: "{male}"  |  female: "{female}"'
        )
    return "\n".join(lines)


def evaluate_batch(items: list[dict]) -> list[_PhraseResult]:
    """Score a batch of phrase+variants in a single LLM call.

    :param:
        items: list of {"n": int, "phrase": str, "tag": str, "variants": dict}

    :returns:
        scores: list of _PhraseResult sorted by n
    """
    formatted = "\n\n".join(
        _format_item(item["n"], item["phrase"], item["tag"], item["variants"])
        for item in items
    )
    result: _BatchScores = _llm.invoke(_BATCH_PROMPT.format(items=formatted))
    return sorted(result.results, key=lambda x: x.n)
