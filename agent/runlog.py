import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# logs/ in the repository root, next to the agent/ folder
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class RunLog:
    """Writes one JSONL file per run: one JSON object per line, one line per step."""

    def __init__(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.model = None  # known only after the config is loaded
        LOG_DIR.mkdir(exist_ok=True)
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_DIR / f"{started}_{self.run_id[:8]}.jsonl"

    def write(self, event: str, *, ok: bool = True, tool: str | None = None,
              args: dict | None = None, duration_ms: int = 0, **extra) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "tool": tool,
            "args": args,
            "ok": ok,
            "model": self.model,
            "duration_ms": duration_ms,
            **extra,
        }
        # "a" = append, so every step adds a line instead of overwriting the file
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
