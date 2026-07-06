import os

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = f"{os.environ['APP_NAME']}_eval"

from app.common.enums import LangEnum, TagEnum  # noqa: E402

DATASET_NAME_W1 = "w1-phrase-quality"

# Total phrases to sample across all tags
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "96"))

# One batch = one tag's share of the total sample
EVAL_BATCH_SIZE = EVAL_SAMPLE_SIZE // len(TagEnum)

# Judge temperatures — 0 for reproducible, deterministic scoring
JUDGE_TEMPERATURE_W1 = float(os.getenv("JUDGE_TEMPERATURE_W1", "0"))
