#pipeline.py
import subprocess
import sys

def step(args, label):
    print(f"\n=== {label} ===")
    rc = subprocess.run([sys.executable, *args]).returncode
    if rc != 0:
        print(f"\n[FAILED] {label} (exit code {rc}). Check the log above.")
        sys.exit(rc)
 
 
def main():
    for split in ("random", "group"):
        step(["train_cv.py", split], f"Cross-validation, {split} split")
    step(["stats_test.py"], "Paired statistical tests")
    for split in ("random", "group"):
        step(["complementary_analysis.py", split], f"Complementarity analysis, {split} split")
    print("\nPipeline complete.")
 
 
if __name__ == "__main__":
    main()
