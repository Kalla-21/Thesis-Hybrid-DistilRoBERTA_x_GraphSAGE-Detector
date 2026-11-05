# pipeline.py
"""
Runs everything in order and records each step in outputs/logs/pipeline.json.

    python pipeline.py              # cross-validation (both splits) -> stats -> complementarity -> evelynn.pt
    python pipeline.py --no-build   # same, without building the final model
"""
from logger import RunLogger

import subprocess
import sys
import time
# The `pipeline.py` script orchestrates the entire workflow of the Evelynn model, including cross-validation,
# statistical tests, complementary analysis, and building the final model. It uses the `RunLogger` class to log each step of the process, 
# including the command executed, exit code, and duration. The script allows for an optional argument to skip building the final model.
SEEDS = [42]         
SPLITS = ("random", "group")


def main(build=True):
    with RunLogger("pipeline") as log:
        # Define a helper function `step` that runs a command as a subprocess, 
        # logs the command, exit code, and duration, and raises an error if the command fails.
        def step(args, label):
            t0 = time.time()
            code = subprocess.run([sys.executable, *args]).returncode
            log.event("step", label, command=" ".join(args), exit_code=code,
                      seconds=round(time.time() - t0, 1))
            if code != 0:
                log.finish("failed", failed_step=label)
                raise SystemExit(code)

        for seed in SEEDS:
            for split in SPLITS: # Run cross-validation for each split (random and group) and seed, logging the command and exit code.
                step(["train_cv.py", split, str(seed)], f"cross-validation: {split} split, seed {seed}")
        step(["stats_test.py"], "paired statistical tests")
        for split in SPLITS: # Run complementary analysis for each split, using the first seed in the SEEDS list.
            step(["complementary_analysis.py", split, str(SEEDS[0])], f"complementarity: {split} split")
        if build:
            step(["build_model.py"], "build final model (evelynn.pt)")


if __name__ == "__main__":
    main(build="--no-build" not in sys.argv)