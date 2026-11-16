import json
import time

from openai import OpenAI, OpenAIError

from agent.config import Config
from agent.runlog import RunLog
from agent.tools import CALENDAR_TOOL, get_calendar_day

SYSTEM_PROMPT = (
    "You are a calendar assistant. To answer questions about appointments, "
    "always look them up with the get_calendar_day tool, never guess. "
    "Call the tool once for every day the question needs. "
    "If a lookup fails, say for which day the answer is missing and why. "
    "If there is no data for a day, say that no data is available, not that the day is free. "
    "When you have everything, answer without calling a tool. "
    "Answer briefly, in the language of the user's question."
)


class AgentStopped(Exception):
    """The run ended without a final answer: step limit reached or the model not usable."""

    def __init__(self, reason: str, message: str, step: int, report: str) -> None:
        super().__init__(message)
        self.reason = reason  # short code for the log, e.g. "step_limit"
        self.step = step      # the step in which the run stopped
        self.report = report  # what was done before the stop, for the user


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def make_report(done: list) -> str:
    """List the tool calls of the run. Written by the code, so it works even when the model does not."""
    if not done:
        return "No tool was called before the stop."
    lines = ["Tool calls before the stop:"]
    for name, args, result in done:
        status = "ok" if result["ok"] else f"failed ({result['error']})"
        lines.append(f"  {name}({json.dumps(args, ensure_ascii=False)}): {status}")
    return "\n".join(lines)


def calendar(args: dict, config: Config) -> dict:
    return get_calendar_day(config.data_base_url, args.get("date", ""))


# All tools the agent offers: name -> (description for the model, function that runs it).
# A new tool needs only one more entry here.
TOOLS = {
    "get_calendar_day": (CALENDAR_TOOL, calendar),
}
TOOL_DESCRIPTIONS = [description for description, _ in TOOLS.values()]


def parse_args(raw: str) -> tuple[dict, str | None]:
    """Read the arguments the model wrote. Returns (args, None) or ({}, what is wrong with them)."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}, f"The arguments are not valid JSON: {raw!r}"
    if not isinstance(args, dict):
        return {}, f"The arguments must be a JSON object, not {raw!r}"
    return args, None


def run_tool(name: str, args: dict, config: Config) -> dict:
    """Execute the tool the model asked for. Bad requests come back as a result, not a crash."""
    if name not in TOOLS:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    _, function = TOOLS[name]
    try:
        return function(args, config)
    except Exception as error:
        # A crash inside one tool ends only this step, not the whole run
        return {"ok": False, "error": f"Tool '{name}' failed: {error!r}"}


def answer(question: str, config: Config, log: RunLog) -> str:
    # max_retries: on a rate limit (HTTP 429) the library waits as long as the server asks and tries again
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=5)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log.write("user_task", task=question)
    done = []  # (tool, args, result) of every tool call, for the report if the run stops

    # Every step is one model call. The model always gets the tools and decides itself
    # whether it needs another one or is done.
    for step in range(1, config.max_steps + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=TOOL_DESCRIPTIONS,
            )
        except OpenAIError as error:
            # Still failing after the retries: without the model the run cannot go on
            raise AgentStopped("model_error", f"Model request failed in step {step}: {error}",
                               step, make_report(done)) from error
        model_ms = elapsed_ms(start)
        reply = response.choices[0].message

        if not reply.tool_calls:
            # No tool request: this is the final answer, the run ends here
            text = reply.content or ""
            log.write("final", duration_ms=model_ms, step=step, answer=text)
            return text

        # Keep the model's tool request in the conversation, then add one result per call
        messages.append(reply)
        for call in reply.tool_calls:
            name = call.function.name
            args, problem = parse_args(call.function.arguments)
            if problem:
                # Log the raw text too, so the log shows what the model actually sent
                log.write("tool_call", tool=name, args=args, duration_ms=model_ms, step=step,
                          raw_args=call.function.arguments)
            else:
                log.write("tool_call", tool=name, args=args, duration_ms=model_ms, step=step)

            start = time.perf_counter()
            if problem:
                # Do not run the tool on broken arguments; the model gets the error and can retry
                result = {"ok": False, "error": problem}
            else:
                result = run_tool(name, args, config)
            log.write("tool_result", tool=name, args=args, ok=result["ok"],
                      duration_ms=elapsed_ms(start), step=step, result=result)
            done.append((name, args, result))

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # The loop ran out without a final answer: this is an error, not a success
    raise AgentStopped("step_limit", f"Step limit of {config.max_steps} reached without a final answer.",
                       config.max_steps, make_report(done))
