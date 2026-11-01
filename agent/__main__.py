import sys

from openai import OpenAIError

from agent.agent import answer
from agent.config import load_config


def main() -> int:
    # Windows defaults to an old encoding that cannot print every character the model may use
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print('Usage: python -m agent "<your question>"', file=sys.stderr)
        return 1

    try:
        config = load_config()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    question = sys.argv[1]
    try:
        print(answer(question, config))
    except OpenAIError as error:
        print(f"Model request failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
