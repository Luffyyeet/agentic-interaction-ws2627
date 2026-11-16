import json
import time

from openai import OpenAI

from agent.config import Config
from agent.runlog import RunLog
from agent.tools import CALENDAR_TOOL, get_calendar_day

SYSTEM_PROMPT = (
    "You are a calendar assistant. To answer questions about appointments, "
    "always look them up with the get_calendar_day tool, never guess. "
    "Answer briefly, in the language of the user's question."
)


def elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def calendar(args: dict, config: Config) -> dict:
    return get_calendar_day(config.data_base_url, args.get("date", ""))


# All tools the agent offers: name -> (description for the model, function that runs it).
# A new tool needs only one more entry here.
TOOLS = {
    "get_calendar_day": (CALENDAR_TOOL, calendar),
}
TOOL_DESCRIPTIONS = [description for description, _ in TOOLS.values()]


def run_tool(name: str, args: dict, config: Config) -> dict:
    """Execute the tool the model asked for. Bad requests come back as a result, not a crash."""
    if name not in TOOLS:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    _, function = TOOLS[name]
    return function(args, config)


def answer(question: str, config: Config, log: RunLog) -> str:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log.write("user_task", task=question)

    # Every step is one model call. The model always gets the tools and decides itself
    # whether it needs another one or is done.
    for step in range(1, config.max_steps + 1):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            tools=TOOL_DESCRIPTIONS,
        )
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
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            log.write("tool_call", tool=name, args=args, duration_ms=model_ms, step=step)

            start = time.perf_counter()
            result = run_tool(name, args, config)
            log.write("tool_result", tool=name, args=args, ok=result["ok"],
                      duration_ms=elapsed_ms(start), step=step, result=result)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # Temporary: step 4 turns this into a proper end with a report and a log entry
    raise RuntimeError(f"Step limit of {config.max_steps} reached without a final answer.")
