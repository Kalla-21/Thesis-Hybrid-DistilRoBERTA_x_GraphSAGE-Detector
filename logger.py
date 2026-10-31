# logger.py
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
import traceback
from datetime import datetime

import numpy as np

LOG_DIR = os.path.join("outputs", "logs")


def _json_default(obj):
    """Make numpy / torch scalars and arrays JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class RunLogger:
    """Collects timestamped events and keeps them in one JSON file, rewritten after every event
    (so a crash never loses what already happened)."""

    def __init__(self, name, log_dir=LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{name}.json")
        self.verbose = os.environ.get("EVELYNN_VERBOSE") == "1"
        self._t0 = time.time()
        self.doc = {
            "run": name,
            "started": datetime.now().isoformat(timespec="seconds"),
            "finished": None,
            "status": "running",
            "duration_s": None,
            "summary": {},
            "events": [],
        }
        self._flush()

    def event(self, stage, message, **data):
        record = {"t": round(time.time() - self._t0, 2), "stage": stage, "message": message}
        record.update(data)
        self.doc["events"].append(record)
        if self.verbose:
            extra = f" {data}" if data else ""
            print(f"[{stage}] {message}{extra}", flush=True)
        self._flush()

    def finish(self, status="ok", **summary):
        self.doc["finished"] = datetime.now().isoformat(timespec="seconds")
        self.doc["status"] = status
        self.doc["duration_s"] = round(time.time() - self._t0, 1)
        self.doc["summary"].update(summary)
        self._flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            if self.doc["status"] == "running":
                self.finish("ok")
        elif self.doc["status"] == "running":
            self.doc["error"] = {
                "type": exc_type.__name__,
                "message": str(exc),
                "traceback": traceback.format_exception(exc_type, exc, tb),
            }
            self.finish("failed")
        return False  # never swallow the exception

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, indent=2, default=_json_default)
        os.replace(tmp, self.path)