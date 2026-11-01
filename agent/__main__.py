import sys

from agent.config import load_config


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m agent "<your question>"', file=sys.stderr)
        return 1

    try:
        config = load_config()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    question = sys.argv[1]
    print(f"Received question: {question}")
    print(f"Using model {config.model} at {config.base_url}")
    print(f"Calendar data from {config.data_base_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
