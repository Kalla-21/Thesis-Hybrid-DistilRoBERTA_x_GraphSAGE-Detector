import subprocess
import sys

def main():
    print("Starting Training Pipeline...")
    train_proc = subprocess.run([sys.executable, "train.py"])
    
    if train_proc.returncode == 0:
        print("\nTraining successful. Initiating Test Evaluation...")
        subprocess.run([sys.executable, "test.py"])
    else:
        print("\nTraining failed. Check logs for details.")

if __name__ == "__main__":
    main()