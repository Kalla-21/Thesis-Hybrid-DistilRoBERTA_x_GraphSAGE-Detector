# logger.py
# this is mainly to replace the print lines in previous iterations of the code, and to log the events in a JSON file for later analysis. 
# It is also used in complementary_analysis.py to log the events of the run.
# produces most of the .json files in the outputs/logs folder, which are used to track the progress of the runs and to debug any issues that may arise.
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
    #make sure that the objects are JSON serializable, and if not, convert them to a string representation.
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class RunLogger:
 #   A context manager for logging the progress of a run to a JSON file. It logs events, status, and summary information. 

    def __init__(self, name, log_dir=LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{name}.json") # the path to the JSON file where the log will be saved. 
        # The file is named based on the provided name and is stored in the specified log directory.
        self.verbose = os.environ.get("EVELYNN_VERBOSE") == "1" #if == 1, then the logger will print the events to the console as well as saving them to the JSON file.
        self._t0 = time.time() # records the start time of the run, which is used to calculate the elapsed time for each event and the total duration of the run.
        self.doc = { # formats the log document with initial information about the run, including its name, start time, status, and placeholders for events and summary data.
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
            print(f"[{stage}] {message}{extra}", flush=True) # print what "stage" is happening, the "message" describing the event, and any additional data associated with the event.
        self._flush() 

    def finish(self, status="ok", **summary):
        self.doc["finished"] = datetime.now().isoformat(timespec="seconds")
        self.doc["status"] = status # ok or failed, indicating whether the run completed successfully or encountered an error.
        self.doc["duration_s"] = round(time.time() - self._t0, 1)
        self.doc["summary"].update(summary)
        self._flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None: # print error message if the run completed successfully, or log the error details if an exception occurred during the run.
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

    def _flush(self): # this is so there is no half-written log file if the program crashes or is interrupted.
        #It writes the current state of the log document to a temporary file and then atomically replaces the main log file with the temporary file.
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, indent=2, default=_json_default)
        os.replace(tmp, self.path)