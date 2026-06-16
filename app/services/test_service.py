import asyncio
import base64
import json
import logging
import re
from typing import Any

import aiohttp

from app.adapters.ai_client import GroqClient, MistralClient
from app.common.logging import log_decorator, logger
from app.services.base import BaseService


class TestService(BaseService):
    """Service for test/debug endpoints: vision, variant generation, and vector search"""

    @log_decorator(level=logging.DEBUG)
    async def vision(self, images_raw: list[bytes], prompt: str) -> str | None:
        """Send images to the Groq vision model and return raw text output

        :param:
            images_raw: list of raw image bytes
            prompt: the vision prompt to use

        :returns:
            raw_text: raw model response string, or None if ai_client2 is not GroqClient
        """
        images_b64 = [base64.b64encode(img).decode() for img in images_raw]
        if not isinstance(self.ai_client2, GroqClient):
            logger.error("ai_client2 is not GroqClient")
            return None
        return await self.ai_client2.vision_chat(images_b64=images_b64, prompt=prompt)

    @log_decorator(level=logging.DEBUG)
    async def pixtral_vision(
        self, images_raw: list[bytes], prompt: str, count: int = 5
    ) -> list[dict[str, Any]]:
        """Send images to Pixtral and parse the structured phrase list

        :param:
            images_raw: list of raw image bytes
            prompt: the Pixtral vision prompt
            count: number of phrase variants to request per tag

        :returns:
            batch: list of phrase dicts (phrase, tag, count), empty on failure
        """
        if not isinstance(self.ai_client, MistralClient):
            logger.error("ai_client is not MistralClient")
            return []
        images_b64 = [base64.b64encode(img).decode() for img in images_raw]
        raw = await self.ai_client.vision_chat(images_b64=images_b64, prompt=prompt)
        if not raw:
            return []
        return self.parse_pixtral_to_batch(raw, count=count)

    @staticmethod
    def parse_pixtral_to_batch(raw: str, count: int = 5) -> list[dict[str, Any]]:
        """Parse raw Pixtral output into a flat deduplicated phrase list

        :param:
            raw: raw text response from the Pixtral vision model
            count: variant count to attach to each phrase dict

        :returns:
            result: deduplicated list of dicts with 'phrase', 'tag', and 'count'
        """
        import ast

        text = raw.strip()
        match = re.search(r"```(?:python|json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            data = ast.literal_eval(text)
        except Exception as exc:
            logger.error("Failed to parse pixtral output: %s", text[:300], exc_info=exc)
            return []
        seen: set[tuple[str, str]] = set()
        result: list[dict[str, Any]] = []
        for photo in data:
            for tag_dict in photo:
                for tag_num, variants in tag_dict.items():
                    for phrase in variants:
                        key = (phrase.strip(), str(tag_num))
                        if key not in seen:
                            seen.add(key)
                            result.append(
                                {
                                    "phrase": phrase.strip(),
                                    "tag": str(tag_num),
                                    "count": count,
                                }
                            )
        return result

    @log_decorator(level=logging.DEBUG)
    async def generate_variants(
        self, phrase: str, tag: str, count: int = 5
    ) -> dict[str, list[str]] | None:
        """Generate mood/gender variants for a single phrase using Groq

        :param:
            phrase: the observation phrase to generate variants for
            tag: the observation category tag
            count: number of variants per mood/gender combination

        :returns:
            variants: dict of mood → gender → phrase list, or None on failure
        """
        # 1. Validate client type
        if not isinstance(self.ai_client2, GroqClient):
            logger.error("ai_client2 is not GroqClient")
            return None

        # 2. Build system instruction and prompt
        system = (
            "Ты генерируешь короткие комментарии о поведении человека на русском языке. "
            "Всегда отвечай строго в формате JSON без дополнительных пояснений."
        )
        prompt = (
            f'Наблюдение за человеком по категории "{tag}":\n'
            f'"{phrase}"\n\n'
            f"Сгенерируй по {count} вариантов короткого комментария (1 предложение) "
            f"для каждого из 5 настроений. Для каждого настроения нужны два набора: "
            f'для мужчины (male) и для женщины (female). Обращение на "ты".\n\n'
            f"Настроения:\n"
            f"A — цинично: очень грубо и жестоко оскорби, насколько способен, не литературно, с насмешкой, можно за гранью, "
            f"обругай, обзови, жести по-страшному, но без матов\n"
            f"B — прямолинейно: честно, цинично и сухо, говоришь как есть\n"
            f"C — нормально: нейтральный тон, обычная речь\n"
            f"D — комплиментарно: мягко, с лёгкой похвалой\n"
            f"E — хвалебно: восторженно, максимально позитивно\n\n"
            f"Формат ответа:\n"
            f'{{"A": {{"male": ["ф1","ф2","ф3","ф4","ф5"], "female": ["ф1","ф2","ф3","ф4","ф5"]}}, '
            f'"B": {{"male": [...], "female": [...]}}, '
            f'"C": {{"male": [...], "female": [...]}}, '
            f'"D": {{"male": [...], "female": [...]}}, '
            f'"E": {{"male": [...], "female": [...]}}}}'
        )
        # 3. Call the AI
        raw = await self.ai_client2.generate(
            prompt=prompt,
            system=system,
            options={"response_format": {"type": "json_object"}},
        )
        if not raw:
            return None

        # 4. Parse JSON response and return mood-keyed dict
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # fix trailing commas before closing braces/brackets
        text = re.sub(r",\s*([}\]])", r"\1", text)

        try:
            data = json.loads(text)
            return {k: v for k, v in data.items() if k in {"A", "B", "C", "D", "E"}}
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse variants JSON: %s", text[:500], exc_info=exc)
            return None

    async def _generate_variants_multi(
        self,
        phrases: list[tuple[str, str]],  # (phrase, tag), up to 8
        count: int = 5,
    ) -> list[dict[str, Any] | None]:
        """Generate variants for a small chunk of phrases in a single Groq request

        :param:
            phrases: list of (phrase, tag) tuples, up to 8 items
            count: number of variants per mood/gender combination

        :returns:
            results: list of variant dicts (original, tag, variants) or None per phrase
        """
        # 1. Validate client type
        if not isinstance(self.ai_client2, GroqClient):
            logger.error("ai_client2 is not GroqClient")
            return [None] * len(phrases)

        # 2. Build numbered items list and prompt
        items = [
            {"id": i, "phrase": phrase, "tag": tag}
            for i, (phrase, tag) in enumerate(phrases)
        ]
        system = (
            "Ты генерируешь короткие комментарии о поведении человека на русском языке. "
            "Всегда отвечай строго в формате JSON без дополнительных пояснений."
        )
        prompt = (
            f"Тебе дан список наблюдений за человеком. "
            f"Для каждого наблюдения сгенерируй по {count} коротких комментариев (1 предложение) "
            f'для каждого из 5 настроений × 2 пола. Обращение на "ты".\n\n'
            f"Наблюдения:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            f"Настроения:\n"
            f"A — цинично: очень грубо и жестоко оскорби, насколько способен, не литературно, "
            f"с насмешкой, можно за гранью, обругай, обзови, жести по-страшному, но без матов\n"
            f"B — прямолинейно: честно, цинично и сухо, говоришь как есть\n"
            f"C — нормально: нейтральный тон, обычная речь\n"
            f"D — комплиментарно: мягко, с лёгкой похвалой\n"
            f"E — хвалебно: восторженно, максимально позитивно\n\n"
            f"Формат ответа — JSON список, по одному объекту на каждое наблюдение:\n"
            f'[{{"id": 0, "A": {{"male": ["ф1",...], "female": ["ф1",...]}}, '
            f'"B": {{"male": [...], "female": [...]}}, '
            f'"C": {{"male": [...], "female": [...]}}, '
            f'"D": {{"male": [...], "female": [...]}}, '
            f'"E": {{"male": [...], "female": [...]}}}}, ...]'
        )
        # 3. Call the AI
        raw = await self.ai_client2.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if not raw:
            return [None] * len(phrases)

        # 4. Parse JSON response
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        text = re.sub(r",\s*([}\]])", r"\1", text)

        try:
            data: list[dict[str, Any]] = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse multi-variants JSON: %s", text[:500], exc_info=exc
            )
            return [None] * len(phrases)

        # 5. Map results back to input phrases by id
        by_id = {item["id"]: item for item in data if "id" in item}
        results: list[dict[str, Any] | None] = []
        for i, (phrase, tag) in enumerate(phrases):
            item = by_id.get(i)
            if item is None:
                results.append(None)
                continue
            variants = {k: v for k, v in item.items() if k in {"A", "B", "C", "D", "E"}}
            results.append(
                {"original": phrase, "tag": tag, "variants": variants}
                if variants
                else None
            )
        return results

    async def generate_variants_batch(
        self,
        phrases: list[tuple[str, str]],  # (phrase, tag)
        count: int = 5,
        chunk_size: int = 8,
        sleep_sec: float = 35.0,
    ) -> list[dict[str, Any] | None]:
        """Process phrases in chunks through _generate_variants_multi with rate-limit sleep

        :param:
            phrases: full list of (phrase, tag) tuples to process
            count: number of variants per mood/gender combination
            chunk_size: number of phrases per Groq request
            sleep_sec: seconds to sleep between chunks to avoid rate limits

        :returns:
            results: list of variant dicts or None per input phrase, in order
        """
        results: list[dict[str, Any] | None] = []
        for i in range(0, len(phrases), chunk_size):
            chunk = phrases[i : i + chunk_size]
            chunk_results = await self._generate_variants_multi(
                phrases=chunk, count=count
            )
            results.extend(chunk_results)
            if i + chunk_size < len(phrases):
                await asyncio.sleep(sleep_sec)
        return results

    @log_decorator(level=logging.DEBUG)
    async def generate_variants_mistral(
        self, phrase: str, tag: str, count: int = 5
    ) -> dict[str, Any] | None:
        """Generate mood/gender variants for a single phrase using Mistral

        :param:
            phrase: the observation phrase to generate variants for
            tag: the observation category tag
            count: number of variants per mood/gender combination

        :returns:
            variants: dict of mood → gender → phrase list, or None on failure
        """
        # 1. Validate client type
        if not isinstance(self.ai_client, MistralClient):
            logger.error("ai_client is not MistralClient")
            return None

        # 2. Build system instruction and prompt
        system = (
            "Ты генерируешь короткие комментарии о поведении человека на русском языке. "
            "Всегда отвечай строго в формате JSON без дополнительных пояснений."
        )
        prompt = (
            f'Наблюдение за человеком по категории "{tag}":\n'
            f'"{phrase}"\n\n'
            f"Сгенерируй по {count} вариантов короткого комментария (1 предложение) "
            f"для каждого из 5 настроений. Для каждого настроения нужны два набора: "
            f'для мужчины (male) и для женщины (female). Обращение на "ты".\n\n'
            f"Настроения:\n"
            f"A — цинично: очень грубо и жестоко оскорби, с насмешкой, можно за гранью, но без матов\n"
            f"B — прямолинейно: честно, цинично и сухо, говоришь как есть\n"
            f"C — нормально: нейтральный тон, обычная речь\n"
            f"D — комплиментарно: мягко, с лёгкой похвалой\n"
            f"E — хвалебно: восторженно, максимально позитивно\n\n"
            f"Формат ответа:\n"
            f'{{"A": {{"male": ["ф1","ф2","ф3","ф4","ф5"], "female": ["ф1","ф2","ф3","ф4","ф5"]}}, '
            f'"B": {{"male": [...], "female": [...]}}, '
            f'"C": {{"male": [...], "female": [...]}}, '
            f'"D": {{"male": [...], "female": [...]}}, '
            f'"E": {{"male": [...], "female": [...]}}}}'
        )
        # 3. Call the AI
        raw = await self.ai_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"response_format": {"type": "json_object"}},
            timeout=aiohttp.ClientTimeout(total=60),
        )
        if not raw:
            return None

        # 4. Parse JSON response and return mood-keyed dict
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        text = re.sub(r",\s*([}\]])", r"\1", text)

        try:
            data = json.loads(text)
            return {k: v for k, v in data.items() if k in {"A", "B", "C", "D", "E"}}
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse mistral variants JSON: %s", text[:500], exc_info=exc
            )
            return None

    async def _generate_variants_multi_mistral(
        self,
        phrases: list[tuple[str, str]],
        count: int = 5,
    ) -> list[dict[str, Any] | None]:
        """Generate variants for a small chunk of phrases in a single Mistral request

        :param:
            phrases: list of (phrase, tag) tuples
            count: number of variants per mood/gender combination

        :returns:
            results: list of variant dicts (original, tag, variants) or None per phrase
        """
        # 1. Validate client type
        if not isinstance(self.ai_client, MistralClient):
            logger.error("ai_client is not MistralClient")
            return [None] * len(phrases)

        # 2. Build numbered items list and prompt
        items = [
            {"id": i, "phrase": phrase, "tag": tag}
            for i, (phrase, tag) in enumerate(phrases)
        ]
        system = (
            "Ты генерируешь короткие комментарии о поведении человека на русском языке. "
            "Всегда отвечай строго в формате JSON без дополнительных пояснений."
        )
        prompt = (
            f"Тебе дан список наблюдений за человеком. "
            f"Для каждого наблюдения сгенерируй по {count} коротких комментариев (1 предложение) "
            f'для каждого из 5 настроений × 2 пола. Обращение на "ты".\n\n'
            f"Наблюдения:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            f"Настроения:\n"
            f"A — цинично: очень грубо и жестоко оскорби, с насмешкой, можно за гранью, но без матов\n"
            f"B — прямолинейно: честно, цинично и сухо, говоришь как есть\n"
            f"C — нормально: нейтральный тон, обычная речь\n"
            f"D — комплиментарно: мягко, с лёгкой похвалой\n"
            f"E — хвалебно: восторженно, максимально позитивно\n\n"
            f'Формат ответа — JSON объект с ключом "results":\n'
            f'{{"results": [{{"id": 0, "A": {{"male": ["ф1",...], "female": ["ф1",...]}}, '
            f'"B": {{"male": [...], "female": [...]}}, "C": {{"male": [...], "female": [...]}}, '
            f'"D": {{"male": [...], "female": [...]}}, "E": {{"male": [...], "female": [...]}}}}, ...]}}'
        )
        # 3. Call the AI
        raw = await self.ai_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"response_format": {"type": "json_object"}},
            timeout=aiohttp.ClientTimeout(total=60),
        )
        if not raw:
            return [None] * len(phrases)

        # 4. Parse JSON response
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        text = re.sub(r",\s*([}\]])", r"\1", text)

        try:
            data = json.loads(text)
            items_out: list[dict[str, Any]] = data.get("results", [])
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse mistral multi-variants JSON: %s",
                text[:500],
                exc_info=exc,
            )
            return [None] * len(phrases)

        # 5. Map results back to input phrases by id
        by_id = {item["id"]: item for item in items_out if "id" in item}
        results: list[dict[str, Any] | None] = []
        for i, (phrase, tag) in enumerate(phrases):
            item = by_id.get(i)
            if item is None:
                results.append(None)
                continue
            variants = {k: v for k, v in item.items() if k in {"A", "B", "C", "D", "E"}}
            results.append(
                {"original": phrase, "tag": tag, "variants": variants}
                if variants
                else None
            )
        return results

    async def generate_variants_batch_mistral(
        self,
        phrases: list[tuple[str, str]],
        count: int = 5,
        chunk_size: int = 10,
        sleep_sec: float = 30.0,
    ) -> list[dict[str, Any] | None]:
        results: list[dict[str, Any] | None] = []
        for i in range(0, len(phrases), chunk_size):
            chunk = phrases[i : i + chunk_size]
            chunk_results = await self._generate_variants_multi_mistral(
                phrases=chunk, count=count
            )
            results.extend(chunk_results)
            if i + chunk_size < len(phrases):
                await asyncio.sleep(sleep_sec)
        return results

    @log_decorator(level=logging.DEBUG)
    async def check(self, text: str) -> Any:
        text_embedding = await self.ai_client.embed(text, task_type="query")
        if not text_embedding:
            return None

        try:
            points = await self.vector_client.search(
                query_vector=text_embedding,
                raise_exception=True,
                limit=10,
                with_payload=True,
            )

            logger.info(points)

            res = []
            for point in points:
                temp = {
                    "score": point.score,
                    "message_id": (
                        point.payload.get("message_id") if point.payload else None
                    ),
                    "chunk_id": (
                        point.payload.get("chunk_id") if point.payload else None
                    ),
                    "total_chunks": (
                        point.payload.get("total_chunks") if point.payload else None
                    ),
                    "text": point.payload.get("text") if point.payload else None,
                }
                res.append(temp)
            return res

        except Exception as exc:
            logger.error("ERROR!!!!!", exc_info=exc)
            return False
