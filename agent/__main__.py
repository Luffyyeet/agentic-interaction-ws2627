import sys

from agent.agent import AgentStopped, answer
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
    except AgentStopped as stop:
        # Ended without an answer: say why and what was done, then exit as an error
        print(stop, file=sys.stderr)
        print(stop.report, file=sys.stderr)
        log.write("error", ok=False, error=str(stop), reason=stop.reason, step=stop.step)
        return 1
    except KeyboardInterrupt:
        # The person stopped the run with Ctrl+C
        print("Stopped by the user.", file=sys.stderr)
        log.write("error", ok=False, error="Stopped by the user.", reason="interrupted")
        return 1
    except Exception as error:
        # Last safety net: whatever else goes wrong, the run still ends visibly in the log
        print(f"Unexpected error: {error}", file=sys.stderr)
        log.write("error", ok=False, error=f"Unexpected error: {error!r}", reason="unexpected")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
