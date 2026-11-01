import sys

from openai import OpenAIError

from agent.agent import answer
from agent.config import load_config
from agent.runlog import RunLog


def main() -> int:
    # Windows defaults to an old encoding that cannot print every character the model may use
    sys.stdout.reconfigure(encoding="utf-8")

    # Created first, so that even a run that fails early leaves a log
    log = RunLog()

    if len(sys.argv) < 2:
        print('Usage: python -m agent "<your question>"', file=sys.stderr)
        log.write("error", ok=False, error="No question given.")
        return 1

    try:
        config = load_config()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        log.write("error", ok=False, error=str(error))
        return 1
    log.model = config.model

    question = sys.argv[1]
    try:
        print(answer(question, config, log))
    except OpenAIError as error:
        print(f"Model request failed: {error}", file=sys.stderr)
        log.write("error", ok=False, error=f"Model request failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
