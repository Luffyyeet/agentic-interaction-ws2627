import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# .env lives in the repository root, one level above this file's folder
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

REQUIRED_VARS = ["OPENAI_BASE_URL", "MODEL", "OPENAI_API_KEY", "DATA_BASE_URL"]


@dataclass
class Config:
    base_url: str
    model: str
    api_key: str
    data_base_url: str


def load_config() -> Config:
    # override=False: variables already set in the environment win over .env
    load_dotenv(ENV_FILE, override=False)

    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing configuration: {', '.join(missing)}. "
            f"Set them in {ENV_FILE} or as environment variables."
        )

    return Config(
        base_url=os.environ["OPENAI_BASE_URL"],
        model=os.environ["MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        data_base_url=os.environ["DATA_BASE_URL"].rstrip("/"),
    )
