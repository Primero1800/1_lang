import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.runnables import RunnableLambda

from app.pyd.requests import SearchSettings, TagExclusionFilters
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.base import BaseDeps
from app.services.phrase_search_service import PhraseSearchService


@pytest.fixture
def test_service() -> PhraseSearchService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    base_deps.queue_client = AsyncMock()
    vector_repository = MagicMock(spec=PhraseVectorRepository)
    return PhraseSearchService(base_deps=base_deps, vector_repository=vector_repository)


def _make_variants() -> dict:
    tone = {"male": ["phrase_m_1", "phrase_m_2"], "female": ["phrase_f_1"]}
    return {"A": tone, "B": tone, "C": tone, "D": tone, "E": tone}


def _make_scored_point(
    phrase_id: int,
    original: str,
    tag: str,
    variants: dict | None = None,
    score: float = 0.9,
) -> MagicMock:
    point = MagicMock()
    point.id = phrase_id
    point.score = score
    point.payload = {
        "original": original,
        "tag": tag,
        "variants": variants if variants is not None else _make_variants(),
    }
    return point


def _fake_vision_response(phrases: dict, gender: str = "male"):
    mock_response = MagicMock()
    mock_response.usage_metadata = {}
    mock_response.content = json.dumps({"gender": gender, "phrases": phrases})
    return mock_response


# --- _parse_vision_phrases ---


def test_parse_vision_phrases_valid_json(test_service: PhraseSearchService) -> None:
    raw = json.dumps(
        {
            "gender": "female",
            "phrases": {"behavior": "typing fast", "appearance": "neat"},
        }
    )
    gender, tag_phrases = test_service._parse_vision_phrases(raw)
    assert gender == "female"
    assert tag_phrases == {"behavior": "typing fast", "appearance": "neat"}


def test_parse_vision_phrases_invalid_json_defaults_to_male(
    test_service: PhraseSearchService,
) -> None:
    gender, tag_phrases = test_service._parse_vision_phrases(
        "not json at all {{ broken"
    )
    assert gender == "male"
    assert tag_phrases == {}


def test_parse_vision_phrases_unknown_gender_defaults_to_male(
    test_service: PhraseSearchService,
) -> None:
    raw = json.dumps({"gender": "alien", "phrases": {"behavior": "something"}})
    gender, _ = test_service._parse_vision_phrases(raw)
    assert gender == "male"


def test_parse_vision_phrases_strips_code_fence(
    test_service: PhraseSearchService,
) -> None:
    inner = json.dumps({"gender": "male", "phrases": {"behavior": "looking focused"}})
    raw = f"```json\n{inner}\n```"
    gender, tag_phrases = test_service._parse_vision_phrases(raw)
    assert gender == "male"
    assert tag_phrases == {"behavior": "looking focused"}


# --- _t1_extract_variants ---


def test_t1_extract_variants_basic() -> None:
    point = _make_scored_point(1, "original_1", "behavior")
    output = PhraseSearchService._t1_extract_variants(
        [[point]], mood_key="C", gender="male"
    )
    assert "original_1" in output
    assert output["original_1"]["tag"] == "behavior"
    assert output["original_1"]["gender"] == "male"
    assert "phrase_m_1" in output["original_1"]["phrases"]


def test_t1_extract_variants_deduplicates_originals() -> None:
    point1 = _make_scored_point(1, "original_1", "behavior", score=0.9)
    point2 = _make_scored_point(2, "original_1", "mood", score=0.8)
    output = PhraseSearchService._t1_extract_variants(
        [[point1, point2]], mood_key="C", gender="male"
    )
    assert len(output) == 1
    assert output["original_1"]["tag"] == "behavior"


def test_t1_extract_variants_missing_mood_skips() -> None:
    tone = {"male": ["phrase_m"]}
    variants = {"A": tone}  # no "C"
    point = _make_scored_point(1, "original_1", "behavior", variants=variants)
    output = PhraseSearchService._t1_extract_variants(
        [[point]], mood_key="C", gender="male"
    )
    assert "original_1" not in output


def test_t1_extract_variants_empty_input() -> None:
    output = PhraseSearchService._t1_extract_variants([], mood_key="C", gender="male")
    assert output == {}


# --- _t1_embed_phrases ---


@pytest.mark.asyncio
async def test_t1_embed_phrases_empty_tag_phrases_returns_empty_vectors(
    test_service: PhraseSearchService,
) -> None:
    result = await test_service._t1_embed_phrases(
        {"tag_phrases": {}, "gender": "male", "lang": "ru", "mood_key": "C"}
    )
    assert result["tag_vectors"] == {}


@pytest.mark.asyncio
async def test_t1_embed_phrases_returns_tag_vectors(
    test_service: PhraseSearchService,
) -> None:
    test_service._llm_embeddings = MagicMock()
    test_service._llm_embeddings.aembed_with_usage = AsyncMock(
        return_value=([[0.1, 0.2], [0.3, 0.4]], 20)
    )
    result = await test_service._t1_embed_phrases(
        {
            "tag_phrases": {"behavior": "typing fast", "mood": "looking tired"},
            "gender": "male",
            "lang": "ru",
            "mood_key": "C",
        }
    )
    await asyncio.sleep(0)
    assert set(result["tag_vectors"].keys()) == {"behavior", "mood"}
    assert result["tag_vectors"]["behavior"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_t1_embed_phrases_queues_token_usage(
    test_service: PhraseSearchService,
) -> None:
    test_service._llm_embeddings = MagicMock()
    test_service._llm_embeddings.aembed_with_usage = AsyncMock(
        return_value=([[0.1, 0.2]], 15)
    )
    await test_service._t1_embed_phrases(
        {
            "tag_phrases": {"behavior": "typing fast"},
            "gender": "male",
            "lang": "ru",
            "mood_key": "C",
        }
    )
    await asyncio.sleep(0)
    test_service.queue_client.xadd.assert_called_once()


# --- _t1_search_extract ---


@pytest.mark.asyncio
async def test_t1_search_extract_empty_vectors_returns_empty(
    test_service: PhraseSearchService,
) -> None:
    result = await test_service._t1_search_extract(
        {"tag_vectors": {}, "gender": "male", "lang": "ru", "mood_key": "C"}
    )
    assert result == {}


@pytest.mark.asyncio
async def test_t1_search_extract_returns_matched_phrases(
    test_service: PhraseSearchService,
) -> None:
    point = _make_scored_point(1, "original_1", "behavior")
    test_service.vector_repository.search_batch = AsyncMock(return_value=[[point]])
    result = await test_service._t1_search_extract(
        {
            "tag_vectors": {"behavior": [0.1, 0.2]},
            "gender": "male",
            "lang": "ru",
            "mood_key": "C",
        }
    )
    assert "original_1" in result
    test_service.vector_repository.search_batch.assert_called_once()


@pytest.mark.asyncio
async def test_t1_search_extract_missing_mood_returns_empty(
    test_service: PhraseSearchService,
) -> None:
    tone = {"male": ["phrase_m"]}
    variants = {"A": tone}  # no "C"
    point = _make_scored_point(1, "original_1", "behavior", variants=variants)
    test_service.vector_repository.search_batch = AsyncMock(return_value=[[point]])
    result = await test_service._t1_search_extract(
        {
            "tag_vectors": {"behavior": [0.1, 0.2]},
            "gender": "male",
            "lang": "ru",
            "mood_key": "C",
        }
    )
    assert result == {}


# --- t1_search ---


@pytest.mark.asyncio
async def test_t1_search_all_tags_restricted_returns_message(
    test_service: PhraseSearchService,
) -> None:
    filters = TagExclusionFilters(
        not_behavior=True,
        not_appearance=True,
        not_age=True,
        not_mood=True,
        not_posture=True,
        not_hairstyle=True,
    )
    result = await test_service.t1_search(
        image_raw=b"img", filters=filters, search_settings=SearchSettings()
    )
    assert "message" in result


@pytest.mark.asyncio
async def test_t1_search_empty_phrases_returns_empty(
    test_service: PhraseSearchService,
) -> None:
    async def _fake_vision(_):
        return _fake_vision_response(phrases={})

    test_service._llm_vision = RunnableLambda(_fake_vision)
    result = await test_service.t1_search(
        image_raw=b"img",
        filters=TagExclusionFilters(),
        search_settings=SearchSettings(),
    )
    assert result == {}


@pytest.mark.asyncio
async def test_t1_search_empty_vectors_returns_empty(
    test_service: PhraseSearchService, mocker
) -> None:
    async def _fake_vision(_):
        return _fake_vision_response(phrases={"behavior": "typing fast"})

    test_service._llm_vision = RunnableLambda(_fake_vision)
    mocker.patch.object(
        test_service,
        "_t1_embed_phrases",
        new=AsyncMock(
            return_value={
                "gender": "male",
                "tag_vectors": {},
                "lang": "ru",
                "mood_key": "A",
            }
        ),
    )
    result = await test_service.t1_search(
        image_raw=b"img",
        filters=TagExclusionFilters(),
        search_settings=SearchSettings(),
    )
    assert result == {}


@pytest.mark.asyncio
async def test_t1_search_success(test_service: PhraseSearchService, mocker) -> None:
    point = _make_scored_point(1, "original_1", "behavior")

    async def _fake_vision(_):
        return _fake_vision_response(phrases={"behavior": "typing fast"})

    test_service._llm_vision = RunnableLambda(_fake_vision)
    mocker.patch.object(
        test_service,
        "_t1_embed_phrases",
        new=AsyncMock(
            return_value={
                "gender": "male",
                "tag_vectors": {"behavior": [0.1, 0.2]},
                "lang": "ru",
                "mood_key": "A",
            }
        ),
    )
    test_service.vector_repository.search_batch = AsyncMock(return_value=[[point]])

    result = await test_service.t1_search(
        image_raw=b"img",
        filters=TagExclusionFilters(),
        search_settings=SearchSettings(),
    )
    assert "original_1" in result
    test_service.vector_repository.search_batch.assert_called_once()
