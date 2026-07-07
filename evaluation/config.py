import os

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = f"{os.environ['APP_NAME']}_eval"

from app.common.enums import LangEnum, TagEnum  # noqa: E402

DATASET_NAME_W1 = "w1-phrase-quality"
DATASET_NAME_W2 = "w2-variants-quality"
DATASET_NAME_W3 = "w3-grammar-quality"
DATASET_NAME_T1 = "t1-retrieval-quality"

# Total phrases to sample across all tags
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "96"))
EVAL_SAMPLE_SIZE_W2 = int(os.getenv("EVAL_SAMPLE_SIZE_W2", "16"))
EVAL_SAMPLE_SIZE_W3 = int(os.getenv("EVAL_SAMPLE_SIZE_W3", "16"))

# One batch = one tag's share of the total sample
EVAL_BATCH_SIZE = EVAL_SAMPLE_SIZE // len(TagEnum)
EVAL_BATCH_SIZE_W2 = int(os.getenv("EVAL_BATCH_SIZE_W2", "2"))
EVAL_BATCH_SIZE_W3 = int(os.getenv("EVAL_BATCH_SIZE_W3", "4"))

# T1 synthetic dataset params
OBSERVATIONS_PER_TAG_T1 = int(os.getenv("OBSERVATIONS_PER_TAG_T1", "10"))
EVAL_LANG_T1 = os.getenv("EVAL_LANG_T1", "ru")

# Judge temperatures — 0 for reproducible, deterministic scoring
JUDGE_TEMPERATURE_W1 = float(os.getenv("JUDGE_TEMPERATURE_W1", "0"))
JUDGE_TEMPERATURE_W2 = float(os.getenv("JUDGE_TEMPERATURE_W2", "0"))
JUDGE_TEMPERATURE_W3 = float(os.getenv("JUDGE_TEMPERATURE_W3", "0"))
JUDGE_TEMPERATURE_T1 = float(os.getenv("JUDGE_TEMPERATURE_T1", "0"))
