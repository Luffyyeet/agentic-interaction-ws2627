import re
from datetime import datetime

import httpx

# The data service knows cities only by their short form, e.g. "muenchen"
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
ENGLISH_NAMES = {"munich": "muenchen"}


def clean_date(date: str) -> str | None:
    """Return the date as YYYY-MM-DD, or None if it is not a valid date."""
    # The model chooses the date, so check it before building a URL from it.
    # isoformat also adds missing zeros: "2026-11-2" would otherwise give a wrong 404.
    try:
        return datetime.strptime(date, "%Y-%m-%d").date().isoformat()
    except (ValueError, TypeError):
        return None


def clean_city(city: str) -> str | None:
    """Turn a city name into the short form used in the URLs ("München" -> "muenchen"), or None if invalid."""
    if not isinstance(city, str):
        return None
    name = city.strip().lower().translate(UMLAUTS).replace(" ", "-")
    # Only letters and hyphens may go into the URL, no "/" or ".."
    if not re.fullmatch(r"[a-z-]+", name):
        return None
    return ENGLISH_NAMES.get(name, name)


def invalid_date(date: str) -> dict:
    return {"ok": False, "error": f"Invalid date '{date}'. Use the format YYYY-MM-DD."}


def invalid_city(city: str) -> dict:
    return {"ok": False, "error": f"Invalid city '{city}'. Use only the name of the city, e.g. Ulm."}


def fetch_json(url: str, service: str) -> dict:
    """GET one file of the data service. Never raises: problems come back as a result.

    Three outcomes: {"ok": False, "error": ...}, {"ok": True, "data": None} for a 404 (no data exists),
    or {"ok": True, "data": ...} with the content of the file.
    """
    try:
        response = httpx.get(url, timeout=10)
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{service} not reachable: {error}"}

    # 404 is not a failure: it means there is no data
    if response.status_code == 404:
        return {"ok": True, "data": None}
    if response.status_code != 200:
        return {"ok": False, "error": f"{service} answered with HTTP {response.status_code}."}

    try:
        return {"ok": True, "data": response.json()}
    except ValueError:
        return unexpected_format(service)


def unexpected_format(service: str) -> dict:
    return {"ok": False, "error": f"{service} sent data in an unexpected format."}


def get_calendar_day(data_base_url: str, date: str) -> dict:
    """Fetch the appointments of one day. Never raises: problems come back as a result."""
    day = clean_date(date)
    if day is None:
        return invalid_date(date)

    reply = fetch_json(f"{data_base_url}/calendar/{day}.json", "Calendar service")
    if not reply["ok"]:
        return reply
    data = reply["data"]

    # No empty "events" list here, since the model would read that as "the day is free".
    if data is None:
        return {"ok": True, "date": day, "data_available": False,
                "note": "No calendar data exists for this date. This does NOT mean the day is free."}

    # The service is outside our control: a broken answer becomes a result, not a crash
    try:
        result = {"ok": True, "date": data["date"], "events": data["events"]}
    except (KeyError, TypeError):
        return unexpected_format("Calendar service")

    # An empty list is a free day, not missing data: say so, so the model does not mix up the two
    if not result["events"]:
        result["note"] = "Calendar data exists for this date and it has no appointments: the day is free."
    return result


def get_weather(data_base_url: str, city: str, date: str) -> dict:
    """Fetch the weather forecast for one city and day. Never raises: problems come back as a result."""
    place = clean_city(city)
    if place is None:
        return invalid_city(city)
    day = clean_date(date)
    if day is None:
        return invalid_date(date)

    reply = fetch_json(f"{data_base_url}/weather/{place}/{day}.json", "Weather service")
    if not reply["ok"]:
        return reply
    data = reply["data"]

    if data is None:
        return {"ok": True, "city": city, "date": day, "data_available": False,
                "note": "No weather data exists for this city and date. The weather is unknown: "
                        "this does NOT mean that it stays dry. Do not use another day or city instead."}

    try:
        return {
            "ok": True,
            "city": data["city"],
            "date": data["date"],
            "condition": data["condition"],
            "temp_c": data["temp_c"],
            "rain_probability": data["rain_probability"],
            "wind_kmh": data["wind_kmh"],
        }
    except (KeyError, TypeError):
        return unexpected_format("Weather service")


def get_train_connections(data_base_url: str, from_city: str, to_city: str, date: str) -> dict:
    """Fetch the train connections between two cities on one day. Never raises: problems come back as a result."""
    start = clean_city(from_city)
    if start is None:
        return invalid_city(from_city)
    end = clean_city(to_city)
    if end is None:
        return invalid_city(to_city)
    day = clean_date(date)
    if day is None:
        return invalid_date(date)

    reply = fetch_json(f"{data_base_url}/transit/{start}/{end}/{day}.json", "Train service")
    if not reply["ok"]:
        return reply
    data = reply["data"]

    if data is None:
        return {"ok": True, "from": from_city, "to": to_city, "date": day, "data_available": False,
                "note": "No train data exists for this route and date. The connections are unknown: "
                        "this does NOT mean that no train runs. Do not use another day or route instead."}

    try:
        result = {
            "ok": True,
            "from": data["from"],
            "to": data["to"],
            "date": data["date"],
            "connections": data["connections"],
        }
    except (KeyError, TypeError):
        return unexpected_format("Train service")

    # Same idea as a free day in the calendar: an empty list is an answer, not missing data
    if not result["connections"]:
        result["note"] = "Train data exists for this route and date, but it lists no connections."
    return result


# Descriptions of the tools in the format the Chat Completions API expects.
# This is all the model knows about a tool, so it decides from this text when and how to call it.
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

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Returns the weather forecast for one city on one day: condition, temperature in °C (temp_c), "
            "rain probability from 0 to 1 (rain_probability, 0.6 means 60 %) and wind in km/h (wind_kmh). "
            "It covers the whole day, there are no values for single hours. "
            "Use it whenever the user asks about the weather, rain, an umbrella or what to wear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city only, without station or country, e.g. Ulm or München.",
                },
                "date": {
                    "type": "string",
                    "description": "The day of the forecast, in the format YYYY-MM-DD, e.g. 2026-10-29.",
                },
            },
            "required": ["city", "date"],
        },
    },
}

TRAIN_TOOL = {
    "type": "function",
    "function": {
        "name": "get_train_connections",
        "description": (
            "Returns all train connections from one city to another on one day, with departure and arrival time, "
            "number of changes (changes), train and price in euros (price_eur). "
            "It covers one direction only: call it again with the cities swapped for the way back. "
            "Use it whenever the user asks how to get to a city, when a train leaves or what it costs. "
            "It only looks connections up, it does not book anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string",
                    "description": "Name of the city where the trip starts, without station or country, e.g. Ulm.",
                },
                "to_city": {
                    "type": "string",
                    "description": "Name of the city where the trip ends, without station or country, e.g. München.",
                },
                "date": {
                    "type": "string",
                    "description": "The day of the trip, in the format YYYY-MM-DD, e.g. 2026-10-29.",
                },
            },
            "required": ["from_city", "to_city", "date"],
        },
    },
}
