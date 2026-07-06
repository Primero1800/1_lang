import os

from dotenv import load_dotenv

load_dotenv()

from app.common.enums import TagEnum  # noqa: E402

DATASET_NAME_W1 = "w1-phrase-quality"

# Total phrases to sample across all tags; splits evenly per tag
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "102"))
LIMIT_PER_TAG = EVAL_SAMPLE_SIZE // len(TagEnum)
