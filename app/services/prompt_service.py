import json as _json

from app.common.enums import TagEnum

_TAG_LABELS: dict[str, dict[str, str]] = {
    "ru": {
        "behavior": "поведение",
        "appearance": "внешность",
        "age": "возраст",
        "mood": "настроение",
        "posture": "поза",
        "hairstyle": "причёска",
    },
    "en": {
        "behavior": "behaviour",
        "appearance": "appearance",
        "age": "age",
        "mood": "mood",
        "posture": "posture",
        "hairstyle": "hairstyle",
    },
}

_RESTRICTED_MESSAGES: dict[str, str] = {
    "ru": "Мне запрещено тебя комментировать",
    "en": "I'm not allowed to comment on you",
}

PROMPT_PIXTRAL_RU = f"""Опиши подробно, ПОДРОБНО(!) что происходит на изображениях: \
чем занимается человек, что открыто на экране, какое его поведение по плану:
конкретно нужно два наблюдения по каждому тэгу — одно конкретное, одно образное или абстрактное:
1. {TagEnum.BEHAVIOR.value} (как он себя ведет, что делает, может засыпает или наоборот слишком бодрый)
2. {TagEnum.APPEARANCE.value} (опрятный, причесанный, ухоженный, лысый, косой, хромой, больной)
3. {TagEnum.AGE.value} (старый, молодой, сопляк, скоро помрет (если очень старый), вчера родился (совсем юный))
4. {TagEnum.MOOD.value} (приуныл, ржет, веселый, в петлю полезть готов)
5. {TagEnum.POSTURE.value} (сидит, раком, как царь, забитый) и тп. по такому принципу
6. {TagEnum.HAIRSTYLE.value} (сама прическа или головной убор по-простому, если прически не видно)
Очень подробно надо, придирчиво. Вариант может быть вопросительным или восклицательным.

Для каждого фото выдай ровно по 5 оригинальных, отличных друг от друга вариантов по каждому из 6 тэгов.
Каждый вариант содержит два поля:
- concrete: конкретное наблюдение из 5-6 слов
- abstract: более образная или абстрактная мысль из 5-6 слов
Итого на одно фото: 6 тэгов × 5 вариантов = 30 записей.
Строго не меньше 5 вариантов на каждый тег — без исключений.

Важно: варианты в рамках одного тега не должны повторяться даже между разными фото в батче.
ВСЕ фразы должны быть ТОЛЬКО на русском языке.

Тэги (значение поля tag): {", ".join(t.value for t in TagEnum)}"""

PROMPT_PIXTRAL_EN = f"""Describe in thorough detail what is happening in the images: \
what the person is doing, what is visible on screen, and their overall behavior.
For each tag provide two observations per variant — one concrete, one abstract or figurative:
1. {TagEnum.BEHAVIOR.value} (what they are doing — falling asleep, overly energetic, distracted, glued to the screen)
2. {TagEnum.APPEARANCE.value} (neat, well-groomed, messy, bald, sloppy, sickly)
3. {TagEnum.AGE.value} (old, young, a kid, a teenager, middle-aged)
4. {TagEnum.MOOD.value} (gloomy, laughing, cheerful, stressed, checked out)
5. {TagEnum.POSTURE.value} (sitting upright, slouching, like royalty, hunched over)
6. {TagEnum.HAIRSTYLE.value} (describe the hair or headwear plainly; if not visible, note that)
Be very thorough and critical. Variants can be phrased as questions or exclamations.

For each photo provide exactly 5 original, distinct variants for each of the 6 tags.
Each variant contains two fields:
- concrete: a specific, literal 5-6 word observation about the person
- abstract: a more figurative or metaphorical 5-6 word take on the same observation
Total per photo: 6 tags × 5 variants = 30 entries.
Strictly no fewer than 5 variants per tag — without exception.

Important: variants within the same tag must remain unique across all photos in the batch.
ALL phrases must be in English ONLY.

Tag values for the tag field: {", ".join(t.value for t in TagEnum)}"""


PROMPT_MISTRAL_VARIANTS_RU = (
    "Ты генерируешь короткие комментарии о поведении человека на русском языке. "
    "Всегда отвечай строго в формате JSON без дополнительных пояснений.\n\n"
    "Тебе дан список наблюдений за человеком. "
    "Каждое наблюдение состоит из двух частей, соединённых через '. ': "
    "конкретное описание и более образная мысль. Комментируй весь образ целиком.\n"
    "Для каждого наблюдения сгенерируй по 5 коротких комментариев (1 предложение) "
    'для каждого из 5 настроений × 2 пола. Обращение на "ты".\n\n'
    "Настроения:\n"
    "A — цинично: очень грубо и жестоко оскорби, с насмешкой, можно за гранью, но без матов\n"
    "B — прямолинейно: честно, цинично и сухо, говоришь как есть\n"
    "C — нормально: нейтральный тон, обычная речь\n"
    "D — комплиментарно: мягко, с лёгкой похвалой\n"
    "E — хвалебно: восторженно, максимально позитивно\n\n"
    'Формат ответа — JSON объект с ключом "results":\n'
    '{"results": [{"id": <phrase_id>, '
    '"A": {"male": ["ф1","ф2","ф3","ф4","ф5"], "female": ["ф1","ф2","ф3","ф4","ф5"]}, '
    '"B": {"male": [...], "female": [...]}, '
    '"C": {"male": [...], "female": [...]}, '
    '"D": {"male": [...], "female": [...]}, '
    '"E": {"male": [...], "female": [...]}}, ...]}'
)

PROMPT_MISTRAL_VARIANTS_EN = (
    "You generate short comments about a person's behaviour in English. "
    "Always respond strictly in JSON format without any additional text.\n\n"
    "You are given a list of observations about a person. "
    "Each observation consists of two parts joined by '. ': "
    "a concrete description and a more figurative reflection. Comment on the overall impression as a whole.\n"
    "For each observation generate 5 short comments (1 sentence) "
    "for each of 5 tones × 2 genders. Address the person as 'you'.\n\n"
    "Tones:\n"
    "A — cynically: insult as harshly and brutally as you can, mockingly, can go over the top, but no slurs\n"
    "B — bluntly: honest, dry, and cynical, say it as it is\n"
    "C — normally: neutral tone, ordinary speech\n"
    "D — complimentary: gently, with a light touch of praise\n"
    "E — enthusiastically: rapturously, as positive as possible\n\n"
    'Response format — JSON object with key "results":\n'
    '{"results": [{"id": <phrase_id>, '
    '"A": {"male": ["p1","p2","p3","p4","p5"], "female": ["p1","p2","p3","p4","p5"]}, '
    '"B": {"male": [...], "female": [...]}, '
    '"C": {"male": [...], "female": [...]}, '
    '"D": {"male": [...], "female": [...]}, '
    '"E": {"male": [...], "female": [...]}}, ...]}}'
)

PROMPT_MISTRAL_TRANSLATE_RU = (
    "Ты переводчик. Переводишь короткие наблюдения о поведении человека с русского на английский язык. "
    "Всегда отвечай строго в формате JSON без дополнительных пояснений.\n\n"
    "Тебе дан список наблюдений на русском, каждое с вариантами комментариев по настроениям A-E. "
    "Каждое наблюдение состоит из ДВУХ предложений, разделённых точкой: первое конкретное, второе образное.\n"
    "Для каждого наблюдения:\n"
    '1. Переведи исходную фразу на английский (поле "translated") — сохрани оба предложения, разделив точкой: "первое. второе"\n'
    "2. Переведи все варианты комментариев (A-E, male/female) на английский, сохраняя тон каждого настроения\n\n"
    "Настроения:\n"
    "A — cynically: very harsh and brutal, mockingly, can go over the top, but no slurs\n"
    "B — bluntly: honest, dry, say it as it is\n"
    "C — normally: neutral tone, ordinary speech\n"
    "D — complimentary: gently, with a light touch of praise\n"
    "E — enthusiastically: rapturously, as positive as possible\n\n"
    'Формат ответа — JSON объект с ключом "results":\n'
    '{"results": [{"id": <phrase_id>, "translated": "<перевод фразы на английский>", '
    '"A": {"male": ["p1","p2","p3","p4","p5"], "female": ["p1","p2","p3","p4","p5"]}, '
    '"B": {"male": [...], "female": [...]}, '
    '"C": {"male": [...], "female": [...]}, '
    '"D": {"male": [...], "female": [...]}, '
    '"E": {"male": [...], "female": [...]}}, ...]}'
)

PROMPT_MISTRAL_TRANSLATE_EN = (
    "You are a translator. You translate short observations about a person's behaviour from English to Russian. "
    "Always respond strictly in JSON format without any additional text.\n\n"
    "You are given a list of observations in English, each with tone variants A-E. "
    "Each observation consists of TWO sentences separated by a dot: the first is concrete, the second is figurative.\n"
    "For each observation:\n"
    '1. Translate the original phrase to Russian (field "translated") — preserve both sentences separated by a dot: "первое. второе"\n'
    "2. Translate all tone variants (A-E, male/female) to Russian, preserving the character of each tone\n\n"
    "Tones:\n"
    "A — цинично: очень грубо и жестоко оскорби, с насмешкой, можно за гранью, но без матов\n"
    "B — прямолинейно: честно, цинично и сухо, говоришь как есть\n"
    "C — нормально: нейтральный тон, обычная речь\n"
    "D — комплиментарно: мягко, с лёгкой похвалой\n"
    "E — хвалебно: восторженно, максимально позитивно\n\n"
    'Response format — JSON object with key "results":\n'
    '{"results": [{"id": <phrase_id>, "translated": "<перевод фразы на русский>", '
    '"A": {"male": ["ф1","ф2","ф3","ф4","ф5"], "female": ["ф1","ф2","ф3","ф4","ф5"]}, '
    '"B": {"male": [...], "female": [...]}, '
    '"C": {"male": [...], "female": [...]}, '
    '"D": {"male": [...], "female": [...]}, '
    '"E": {"male": [...], "female": [...]}}, ...]}'
)

_PROMPTS: dict[tuple[str, str], str] = {
    ("pixtral_vision", "ru"): PROMPT_PIXTRAL_RU,
    ("pixtral_vision", "en"): PROMPT_PIXTRAL_EN,
    ("mistral_variants", "ru"): PROMPT_MISTRAL_VARIANTS_RU,
    ("mistral_variants", "en"): PROMPT_MISTRAL_VARIANTS_EN,
    ("mistral_translate", "ru"): PROMPT_MISTRAL_TRANSLATE_RU,
    ("mistral_translate", "en"): PROMPT_MISTRAL_TRANSLATE_EN,
}


class PromptService:
    """Registry of pre-built prompts keyed by (marker, lang) tuples"""

    @classmethod
    def get(cls, marker: str, lang: str) -> str:
        """Return the prompt string for the given marker and language

        :param:
            marker: prompt identifier (e.g. 'pixtral_vision')
            lang: language code ('ru' or 'en')

        :raise:
            ValueError: if no prompt is registered for the given (marker, lang) pair

        :returns:
            prompt: the complete prompt string
        """
        key = (marker, lang)
        prompt = _PROMPTS.get(key)
        if prompt is None:
            raise ValueError(f"No prompt for marker='{marker}', lang='{lang}'")
        return prompt

    @staticmethod
    def get_t1_vision_prompt(lang: str, allowed_tags: list[str]) -> str:
        """Build a dynamic T1 vision prompt for the given language and allowed tag categories

        :param:
            lang: language code ('ru' or 'en')
            allowed_tags: list of tag keys to request observations for

        :returns:
            prompt: complete prompt string with per-tag instructions and JSON example
        """
        labels = _TAG_LABELS.get(lang, _TAG_LABELS["en"])
        tag_lines = "\n".join(
            f"- {labels.get(tag, tag)} ({tag})" for tag in allowed_tags
        )
        if lang == "ru":
            example = _json.dumps(
                {
                    "gender": "male",
                    "phrases": {
                        tag: ["конкретное наблюдение", "образная мысль"]
                        for tag in allowed_tags
                    },
                },
                ensure_ascii=False,
            )
            return (
                "Посмотри на фото и определи пол человека: 'male' или 'female' (по умолчанию 'male').\n"
                "По каждой из следующих категорий дай два наблюдения на русском языке в виде массива:\n"
                "- первое: конкретное (5-6 слов, нижний регистр, без знаков препинания)\n"
                "- второе: более абстрактное или образное (5-6 слов, нижний регистр, без знаков препинания)\n"
                f"{tag_lines}\n"
                "Никаких пояснений — только JSON:\n"
                f"{example}"
            )
        example = _json.dumps(
            {
                "gender": "male",
                "phrases": {
                    tag: ["concrete observation", "abstract reflection"]
                    for tag in allowed_tags
                },
            },
            ensure_ascii=False,
        )
        return (
            "Look at the photo and determine the gender: 'male' or 'female' (default 'male').\n"
            "For each category below give two observations in English as an array:\n"
            "- first: concrete (5-6 words, lowercase, no trailing punctuation)\n"
            "- second: more abstract or figurative (5-6 words, lowercase, no trailing punctuation)\n"
            f"{tag_lines}\n"
            "No explanations — only JSON:\n"
            f"{example}"
        )

    @staticmethod
    def get_restricted_message(lang: str) -> str:
        """Return the 'all tags restricted' message for the given language

        :param:
            lang: language code ('ru' or 'en')

        :returns:
            message: localised restriction message string
        """
        return _RESTRICTED_MESSAGES.get(lang, _RESTRICTED_MESSAGES["en"])
