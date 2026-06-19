from app.common.enums import TagEnum

PROMPT_PIXTRAL_RU = f"""Опиши подробно, ПОДРОБНО(!) что происходит на изображениях: \
чем занимается человек, что открыто на экране, какое его поведение по плану:
конкретно нужно четкое описание в одном расширенном предложении по тэгам:
1. {TagEnum.BEHAVIOR.value} (как он себя ведет, что делает, может засыпает или наоборот слишком бодрый) - предложение из 5-6 слов,
2. {TagEnum.APPEARANCE.value} (опрятный, причесанный, ухоженный, лысый, косой, хромой, больной) - предложение из 5-6 слов,
3. {TagEnum.AGE.value} (старый, молодой, сопляк, скоро помрет (если очень старый), вчера родился (совсем юный)) - предложение из 5-6 слов,
4. {TagEnum.MOOD.value} (приуныл, ржет, веселый, в петлю полезть готов) - предложение из 5-6 слов,
5. {TagEnum.POSTURE.value} (сидит, раком, как царь, забитый) и тп. по такому принципу - предложение из 5-6 слов.
6. {TagEnum.HAIRSTYLE.value} (сама прическа или головной убор по-простому, если прически не видно) - предложение из 5-6 слов
Очень подробно надо, придирчиво. При этом вариант может быть в том числе вопросительным или восклицательным.

для каждого фото надо выдать по 5 оригинальных, отличных друг от друга вариантов пунктов от 1 до 6 тэгов в виде
1. {TagEnum.BEHAVIOR.value}: [Вариант 1_1. Вариант 1_2. Вариант 1_3. Вариант 1_4. Вариант 1_5]
2. {TagEnum.APPEARANCE.value}: [Вариант 2_1. Вариант 2_2. Вариант 2_3. Вариант 2_4. Вариант 2_5]
...
6 {TagEnum.HAIRSTYLE.value}: [Вариант 6_1. Вариант 6_2. Вариант 6_3. Вариант 6_4. Вариант 6_5]

Обращаю внимание, что каждый вариант - это не одно слово, а целое предложение из 5-6 слов.

И очень важное дополнительное условие - варианты в рамках одного тега даже в разных фотках не должны повторяться,
а оставаться уникальными. Т.е. если в первой фотографии в прическе есть например "нет волос на голове",
то такого варианта не должно быть в описании прически остальных фотографий.

Все варианты возвращает в нижнем регистре lower(). Ответ должен быть списком list из списков list словарей dict.
Никаких дополнительных фраз - только list[list[dict[int, list[str]]]]
[
    [
        {{"{TagEnum.BEHAVIOR.value}": ["Вариант 1_1", "Вариант 1_2", "Вариант 1_3", "Вариант 1_4", "Вариант 1_5"]}},
        {{"{TagEnum.APPEARANCE.value}": ["Вариант 2_1", "Вариант 2_2", "Вариант 2_3", "Вариант 2_4", "Вариант 2_5"]}},
        ...
        {{"{TagEnum.HAIRSTYLE.value}": ["Вариант 6_1", "Вариант 6_2", "Вариант 6_3", "Вариант 6_4", "Вариант 6_5"]}},
    ],
    ...
]"""

PROMPT_PIXTRAL_EN = f"""Describe in thorough detail what is happening in the images: \
what the person is doing, what is visible on screen, and their overall behavior.
Provide one clear, specific sentence per tag:
1. {TagEnum.BEHAVIOR.value} (what they are doing — are they falling asleep, overly energetic, distracted?) — 5-6 words
2. {TagEnum.APPEARANCE.value} (neat, well-groomed, messy, bald, sloppy, sickly) — 5-6 words
3. {TagEnum.AGE.value} (old, young, a kid, a teenager, middle-aged) — 5-6 words
4. {TagEnum.MOOD.value} (gloomy, laughing, cheerful, stressed, checked out) — 5-6 words
5. {TagEnum.POSTURE.value} (sitting upright, slouching, like royalty, hunched over) — 5-6 words
6. {TagEnum.HAIRSTYLE.value} (describe the hair or headwear plainly; if not visible, note that) — 5-6 words
Be very thorough and critical. Variants can be phrased as questions or exclamations.

For each photo provide 5 original, distinct variants for tags 1-6:
1. {TagEnum.BEHAVIOR.value}: [Variant 1_1. Variant 1_2. Variant 1_3. Variant 1_4. Variant 1_5]
2. {TagEnum.APPEARANCE.value}: [Variant 2_1. ...]
...
6. {TagEnum.HAIRSTYLE.value}: [Variant 6_1. ...]

Each variant must be a full sentence of 5-6 words, not a single word.
Variants within the same tag must remain unique across all photos in the batch.

Return all variants in lowercase. Output only list[list[dict[int, list[str]]]] — no extra text.
[
    [
        {{"{TagEnum.BEHAVIOR.value}": ["Variant 1_1", "Variant 1_2", "Variant 1_3", "Variant 1_4", "Variant 1_5"]}},
        {{"{TagEnum.APPEARANCE.value}": ["Variant 2_1", "Variant 2_2", "Variant 2_3", "Variant 2_4", "Variant 2_5"]}},
        ...
        {{"{TagEnum.HAIRSTYLE.value}": ["Variant 6_1", "Variant 6_2", "Variant 6_3", "Variant 6_4", "Variant 6_5"]}},
    ],
    ...
]"""

PROMPT_T1_VISION_RU = (
    "Посмотри на фото и определи пол человека на изображении, а также дай ровно 3 коротких наблюдения о нём. "
    "Пол: 'male' или 'female'. Если определить невозможно — используй 'male'. "
    "Каждое наблюдение — одно предложение из 5-6 слов в нижнем регистре, без знаков препинания в конце. "
    "Наблюдения должны касаться разных аспектов: поведения, внешности или настроения. "
    "Никаких пояснений — только JSON-объект:\n"
    '{"gender": "male", "phrases": ["наблюдение 1", "наблюдение 2", "наблюдение 3"]}'
)

PROMPT_T1_VISION_EN = (
    "Look at the photo, determine the gender of the person in the image, and give exactly 3 short observations about them. "
    "Gender: 'male' or 'female'. If undetermined — use 'male'. "
    "Each observation is one sentence of 5-6 words in lowercase, no trailing punctuation. "
    "Observations should cover different aspects: behaviour, appearance, or mood. "
    "No explanations — only a JSON object:\n"
    '{"gender": "male", "phrases": ["observation 1", "observation 2", "observation 3"]}'
)

PROMPT_MISTRAL_VARIANTS_RU = (
    "Ты генерируешь короткие комментарии о поведении человека на русском языке. "
    "Всегда отвечай строго в формате JSON без дополнительных пояснений.\n\n"
    "Тебе дан список наблюдений за человеком. "
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
    "Для каждого наблюдения:\n"
    '1. Переведи исходную фразу на английский (поле "translated")\n'
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
    "For each observation:\n"
    '1. Translate the original phrase to Russian (field "translated")\n'
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
    ("t1_vision", "ru"): PROMPT_T1_VISION_RU,
    ("t1_vision", "en"): PROMPT_T1_VISION_EN,
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
