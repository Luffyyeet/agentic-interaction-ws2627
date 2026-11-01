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


def run_tool(name: str, args: dict, config: Config) -> dict:
    """Execute the tool the model asked for. Bad requests come back as a result, not a crash."""
    if name == "get_calendar_day":
        return get_calendar_day(config.data_base_url, args.get("date", ""))
    return {"ok": False, "error": f"Unknown tool '{name}'."}


def answer(question: str, config: Config, log: RunLog) -> str:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    log.write("user_task", task=question)

    # 1st model call: the model sees the question and the tool description
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        tools=[CALENDAR_TOOL],
    )
    first_call_ms = elapsed_ms(start)
    reply = response.choices[0].message

    if not reply.tool_calls:
        # The model answered directly without using the tool
        text = reply.content or ""
        log.write("final", duration_ms=first_call_ms, answer=text)
        return text

    # Keep the model's tool request in the conversation, then add one result per call
    messages.append(reply)
    for call in reply.tool_calls:
        name = call.function.name
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        log.write("tool_call", tool=name, args=args, duration_ms=first_call_ms)

        start = time.perf_counter()
        result = run_tool(name, args, config)
        log.write("tool_result", tool=name, args=args, ok=result["ok"],
                  duration_ms=elapsed_ms(start), result=result)

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # 2nd model call: the model reads the tool results and writes the answer
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
    )
    text = response.choices[0].message.content or ""
    log.write("final", duration_ms=elapsed_ms(start), answer=text)
    return text
