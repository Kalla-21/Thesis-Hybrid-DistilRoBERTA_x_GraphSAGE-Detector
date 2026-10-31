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

SEEDS = [42]         
SPLITS = ("random", "group")


def main(build=True):
    with RunLogger("pipeline") as log:

        def step(args, label):
            t0 = time.time()
            code = subprocess.run([sys.executable, *args]).returncode
            log.event("step", label, command=" ".join(args), exit_code=code,
                      seconds=round(time.time() - t0, 1))
            if code != 0:
                log.finish("failed", failed_step=label)
                raise SystemExit(code)

        for seed in SEEDS:
            for split in SPLITS:
                step(["train_cv.py", split, str(seed)], f"cross-validation: {split} split, seed {seed}")
        step(["stats_test.py"], "paired statistical tests")
        for split in SPLITS:
            step(["complementary_analysis.py", split, str(SEEDS[0])], f"complementarity: {split} split")
        if build:
            step(["build_model.py"], "build final model (evelynn.pt)")


if __name__ == "__main__":
    main(build="--no-build" not in sys.argv)