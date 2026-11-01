import json

from openai import OpenAI

from agent.config import Config
from agent.tools import CALENDAR_TOOL, get_calendar_day

SYSTEM_PROMPT = (
    "You are a calendar assistant. To answer questions about appointments, "
    "always look them up with the get_calendar_day tool, never guess. "
    "Answer briefly, in the language of the user's question."
)


def run_tool(name: str, arguments: str, config: Config) -> dict:
    """Execute the tool the model asked for. Bad requests come back as a result, not a crash."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Arguments were not valid JSON."}

    if name == "get_calendar_day":
        return get_calendar_day(config.data_base_url, args.get("date", ""))
    return {"ok": False, "error": f"Unknown tool '{name}'."}


def answer(question: str, config: Config) -> str:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # 1st model call: the model sees the question and the tool description
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        tools=[CALENDAR_TOOL],
    )
    reply = response.choices[0].message

    if not reply.tool_calls:
        # The model answered directly without using the tool
        return reply.content or ""

    # Keep the model's tool request in the conversation, then add one result per call
    messages.append(reply)
    for call in reply.tool_calls:
        result = run_tool(call.function.name, call.function.arguments, config)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # 2nd model call: the model reads the tool results and writes the answer
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
    )
    return response.choices[0].message.content or ""
