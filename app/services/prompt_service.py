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

_PROMPTS: dict[tuple[str, str], str] = {
    ("pixtral_vision", "ru"): PROMPT_PIXTRAL_RU,
    ("pixtral_vision", "en"): PROMPT_PIXTRAL_EN,
}


class PromptService:
    """Registry of pre-built prompts keyed by (marker, lang) tuples"""

    def get(self, marker: str, lang: str) -> str:
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
