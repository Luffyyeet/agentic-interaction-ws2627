# agentic-interaction-ws2627

Course project for **Agentic Human-AI Interaction** (Ulm University, winter term 2026/27):
a day and travel planner agent that works with simulated tools (calendar, timetable, weather, notes).

## Requirements

- Python 3.11 or newer

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create a file named `.env` in the repository root (it is never committed):

```
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL=openai/gpt-oss-20b
OPENAI_API_KEY=your-key-here
DATA_BASE_URL=https://jhbelz.github.io/agentic-interaction-data/v1
```

Environment variables override the values in `.env`.

## Run

From the repository root:

```bash
python -m agent "Was steht am 29.10.2026 im Kalender?"
```

The answer is printed to standard output, and the exit code is 0. Errors go to standard error with exit code 1.

## Tools

| Tool | Parameters | What it does |
|---|---|---|
| `get_calendar_day` | `date` (`YYYY-MM-DD`) | Reads the appointments of one day from `{DATA_BASE_URL}/calendar/{date}.json`. HTTP 404 means no data and is returned as an empty day, not an error. |

The flow is one pass: model → tool → model. The model gets the question and the tool description, asks for the tool, the agent runs it, and the model answers from the result.

## Run log

Each run writes its own file `logs/<start time>_<run id>.jsonl` (not committed), one JSON object per line:

| Field | Content |
|---|---|
| `ts` | Time stamp, ISO 8601 (UTC) |
| `run_id` | One ID per run |
| `event` | `user_task`, `tool_call`, `tool_result`, `final` or `error` |
| `tool`, `args` | Tool name and arguments (`null` where not applicable) |
| `ok` | Whether the step succeeded |
| `model` | Model name as sent to the endpoint |
| `duration_ms` | Duration of the step |

Some events add a field with their content: `task`, `result`, `answer` or `error`.

## Structure

```
agent/
  __main__.py   entry point: reads the question, prints the answer
  agent.py      model → tool → model
  tools.py      calendar tool and its description for the model
  config.py     loads .env, environment variables take precedence
  runlog.py     JSONL run log
```
