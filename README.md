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

Optional: `MAX_STEPS` sets the maximum number of model calls per run (default `10`).

## Run

From the repository root:

```bash
python -m agent "Was steht am 29.10.2026 im Kalender?"
python -m agent "What is on 29 Oct 2026 and on 30 Oct 2026?"
```

The answer is printed to standard output, and the exit code is 0. Errors go to standard error with exit code 1.

## Tools

| Tool | Parameters | What it does |
|---|---|---|
| `get_calendar_day` | `date` (`YYYY-MM-DD`) | Reads the appointments of one day from `{DATA_BASE_URL}/calendar/{date}.json`. HTTP 404 means that no data exists for that day; it is returned as "no data available", not as an error and not as a free day. |

## Agent loop

Every step is one model call. The model always gets the tools and decides by itself whether it needs another tool call or is done:

1. The model gets the conversation so far and the tool descriptions.
2. If it asks for tools, the agent runs them and adds the results to the conversation, then the next step starts.
3. If it answers without a tool call, that answer is the result and the run ends.

How a run can end:

| End | Exit code | Last log event | `reason` |
|---|---|---|---|
| Final answer | 0 | `final` | – |
| Step limit (`MAX_STEPS`) reached without an answer | 1 | `error` | `step_limit` |
| Model replied with neither a tool call nor text (the error names the model's `finish_reason`) | 1 | `error` | `empty_answer` |
| Model not usable (wrong key or model, rate limit still hit after 5 retries, timeout of 120 s per request) | 1 | `error` | `model_error` |
| Stopped with Ctrl+C | 1 | `error` | `interrupted` |
| Unexpected error | 1 | `error` | `unexpected` |

When a run stops without an answer, standard error also lists the tool calls made before the stop.

Errors in a single step do not end the run. They are passed to the model as a tool result with `ok: false`, and the model can react to them in the next step:

- an invalid date, broken or wrong arguments, an unknown tool,
- the calendar service not reachable or answering with unexpected data,
- a tool call that the model wrote in an invalid form and that the server rejected (HTTP 400 `tool_use_failed`); this step is logged as a failed `tool_call`, and the next step tries again.

On a rate limit (HTTP 429) the agent waits as long as the server asks and tries again.

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
`tool_call`, `tool_result`, `final`, and `error` with reason `step_limit`, `empty_answer` or `model_error` also have `step`, the number of the model call they belong to.
`error` events have `reason` (see the table above). A `tool_call` with arguments that are not valid JSON also has `raw_args`, the text the model sent.

## Structure

```
agent/
  __main__.py   entry point: reads the question, prints the answer, decides the exit code
  agent.py      agent loop: model → tools → model … until an answer or the step limit
  tools.py      calendar tool and its description for the model
  config.py     loads .env, environment variables take precedence, MAX_STEPS
  runlog.py     JSONL run log
```
