import os

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = f"{os.environ['APP_NAME']}_eval"

from app.common.enums import LangEnum, TagEnum  # noqa: E402

DATASET_NAME_W1 = "w1-phrase-quality"
DATASET_NAME_W2 = "w2-variants-quality"

# Total phrases to sample across all tags
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "96"))
EVAL_SAMPLE_SIZE_W2 = int(os.getenv("EVAL_SAMPLE_SIZE_W2", "16"))

# One batch = one tag's share of the total sample
EVAL_BATCH_SIZE = EVAL_SAMPLE_SIZE // len(TagEnum)

# Judge temperatures — 0 for reproducible, deterministic scoring
JUDGE_TEMPERATURE_W1 = float(os.getenv("JUDGE_TEMPERATURE_W1", "0"))
JUDGE_TEMPERATURE_W2 = float(os.getenv("JUDGE_TEMPERATURE_W2", "0"))
