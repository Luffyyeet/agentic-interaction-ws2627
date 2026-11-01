from datetime import datetime

import httpx


def get_calendar_day(data_base_url: str, date: str) -> dict:
    """Fetch the appointments of one day. Never raises: problems come back as a result."""
    # The model chooses the date, so check it before building a URL from it
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"ok": False, "error": f"Invalid date '{date}'. Use the format YYYY-MM-DD."}

    url = f"{data_base_url}/calendar/{date}.json"
    try:
        response = httpx.get(url, timeout=10)
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"Calendar service not reachable: {error}"}

    # 404 is not a failure: it means there is no calendar data for that day
    if response.status_code == 404:
        return {"ok": True, "date": date, "events": [], "note": "No calendar data for this date."}
    if response.status_code != 200:
        return {"ok": False, "error": f"Calendar service answered with HTTP {response.status_code}."}

    day = response.json()
    return {"ok": True, "date": day["date"], "events": day["events"]}


# Description of the tool in the format the Chat Completions API expects.
# This is all the model knows about the tool, so it decides from this text when and how to call it.
CALENDAR_TOOL = {
    "type": "function",
    "function": {
        "name": "get_calendar_day",
        "description": (
            "Returns all appointments in the user's calendar for one specific day, "
            "with start and end time, title and location. "
            "Use it whenever the user asks what is planned, scheduled or in the calendar on a date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "The day to look up, in the format YYYY-MM-DD, e.g. 2026-10-29.",
                },
            },
            "required": ["date"],
        },
    },
}
