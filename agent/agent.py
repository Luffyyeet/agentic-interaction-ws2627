import json
import time

from openai import BadRequestError, OpenAI, OpenAIError

from agent.config import Config
from agent.runlog import RunLog
from agent.tools import (
    CALENDAR_TOOL,
    TRAIN_TOOL,
    WEATHER_TOOL,
    get_calendar_day,
    get_home,
    get_train_connections,
    get_weather,
)

SYSTEM_PROMPT = (
    "You are a day and travel planner. You can look up appointments, the weather and train connections. "
    "Always look these things up with the tools, never guess and never use your own knowledge for them. "
    "Call a tool once for every day, city and direction the question needs. "
    "Do all lookups first, also when you have to ask the user something. "
    "Every time, price and weather value in your answer must be copied exactly from a tool result. "
    "If a question has several parts, answer every part. "
    "If a lookup fails, say for which part the answer is missing and why. "
    "If no data is available, say exactly that for this part. It does not mean that the day is free, "
    "that it stays dry or that no train runs. Never fill the gap with values from another day, city or route. "
    "For trains, name the train with its departure and arrival time. "
    "For weather, name the condition and the rain probability. "
    "If the question is about a trip, its weather part means the destination. "
    "You can only look things up, you cannot book or change anything. "
    "When you have everything, answer without calling a tool. "
    "Answer briefly. "
)

HOME_KNOWN = (
    "The user's home is {home}. A trip without a named starting point starts in {home}; say so in the answer. "
    "A question about the weather that names no place at all means {home}. "
    "A place named in the question always comes first. "
)
HOME_UNKNOWN = (
    "The user's home is unknown, never guess it. If a trip has no named starting point, look up the weather "
    "of the destination if the question asks for it, and ask the user for the starting point in your answer. "
    "If a question about the weather names no place at all, ask the user for the place. "
)
# Last part of the prompt, because the tool results are in another language than some questions
LANGUAGE_RULE = (
    "Language: an English question gets an English answer, a German question gets a German answer, and so on. "
    "Only the language of the question counts, not the language of the tool results: translate their values."
)


class AgentStopped(Exception):
    """The run ended without a final answer: step limit reached or the model not usable."""

    def __init__(self, reason: str, message: str, step: int, report: str) -> None:
        super().__init__(message)
        self.reason = reason  # short code for the log, e.g. "step_limit"
        self.step = step      # the step in which the run stopped
        self.report = report  # what was done before the stop, for the user


def make_system_prompt(home: str | None) -> str:
    """The fixed prompt, the rule for questions that do not name a place, and the language rule."""
    return SYSTEM_PROMPT + (HOME_KNOWN.format(home=home) if home else HOME_UNKNOWN) + LANGUAGE_RULE


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


def weather(args: dict, config: Config) -> dict:
    return get_weather(config.data_base_url, args.get("city", ""), args.get("date", ""))


def trains(args: dict, config: Config) -> dict:
    return get_train_connections(config.data_base_url, args.get("from_city", ""),
                                 args.get("to_city", ""), args.get("date", ""))


# All tools the agent offers: name -> (description for the model, function that runs it).
# A new tool needs only one more entry here.
TOOLS = {
    "get_calendar_day": (CALENDAR_TOOL, calendar),
    "get_weather": (WEATHER_TOOL, weather),
    "get_train_connections": (TRAIN_TOOL, trains),
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
    # max_retries: on a rate limit (HTTP 429) the library waits as long as the server asks and tries again.
    # timeout: give up on a single request after 120 s instead of the library's default of 10 minutes.
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=5, timeout=120)

    home = get_home(config.data_base_url)
    messages = [
        {"role": "system", "content": make_system_prompt(home)},
        {"role": "user", "content": question},
    ]
    log.write("user_task", task=question, home=home)
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
            if isinstance(error, BadRequestError) and error.code == "tool_use_failed":
                # The model wrote a broken tool call (e.g. a wrong tool name) and the server
                # rejected the reply. Only this step failed: log it, then the model tries again.
                problem = f"Model produced an invalid tool call: {error.message}"
                log.write("tool_call", ok=False, duration_ms=elapsed_ms(start), step=step, error=problem)
                done.append(("invalid tool call", {}, {"ok": False, "error": problem}))
                continue
            # Still failing after the retries: without the model the run cannot go on
            raise AgentStopped("model_error", f"Model request failed in step {step}: {error}",
                               step, make_report(done)) from error
        model_ms = elapsed_ms(start)
        reply = response.choices[0].message

        if not reply.tool_calls:
            # No tool request: this is the final answer, the run ends here
            text = reply.content or ""
            if not text.strip():
                # No tool call and no text is not an answer: end as an error, not as a success.
                # finish_reason tells why, e.g. "stop" (wrote nothing) or "length" (cut off).
                finish_reason = response.choices[0].finish_reason
                raise AgentStopped("empty_answer",
                                   f"The model gave an empty answer in step {step} (finish_reason: {finish_reason}).",
                                   step, make_report(done))
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
