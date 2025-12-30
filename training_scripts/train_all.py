import sys
import subprocess
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_DIR.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

def run(cmd):
    print("\n=== Running:", " ".join(map(str, cmd)), "===\n")
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(REPO_ROOT))


def main():
    py = sys.executable

    # digits
    run([py, TRAIN_DIR / "train.py", "--only", "cnn", "--epochs", "15", "--force"])
    run([py, TRAIN_DIR / "train.py", "--only", "baseline", "--force"])

    # advanced digits
    run([
        py, TRAIN_DIR / "advanced_train.py",
        "--task", "digits",
        "--epochs", "12",
        "--batch", "256",
        "--lr", "0.001",
        "--data_dir", str(DATA_DIR),
        "--out", str(MODELS_DIR / "digits_advanced.pth"),
        "--num_workers", "0"
    ])

    # letters cnn
    run([py, TRAIN_DIR / "trainv2.py", "--epochs", "20", "--lr", "0.0005", "--force"])

    # advanced letters
    run([
        py, TRAIN_DIR / "advanced_train.py",
        "--task", "letters",
        "--epochs", "12",
        "--batch", "256",
        "--lr", "0.001",
        "--data_dir", str(DATA_DIR),
        "--out", str(MODELS_DIR / "letters_advanced.pth"),
        "--num_workers", "0"
    ])

    # letters baseline
    run([
        py, TRAIN_DIR / "train_letters_baseline.py",
        "--train_n", "60000",
        "--test_n", "14800",
        "--C", "3.0",
        "--max_iter", "2500",
        "--force"
    ])

    print("\nAll training finished.")
    print("Now run:\n  streamlit run appv2.py\n")


if __name__ == "__main__":
    main()
