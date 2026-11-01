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

The answer is printed to standard output. Each run writes a JSONL log file to `logs/`.
