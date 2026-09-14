"""
Standalone pre-download step (Step 6 of the setup guide).

Run this once before `python app.py` so the first real launch doesn't
stall on a ~2.49GB download:

    python scripts/download_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import model_manager  # noqa: E402


def main():
    print(f"Model repo:  {config.HF_MODEL_REPO}")
    print(f"Model file:  {config.HF_MODEL_FILENAME}")
    print(f"Target path: {config.LOCAL_MODEL_PATH}\n")
    model_manager.get_model()
    # Plain ASCII, not an emoji: Windows' default cp1252 console encoding
    # can't render "✅" and this print would crash after the download had
    # already succeeded (same class of bug fixed for model_manager.py's
    # download message in PR #1 -- that fix just didn't cover this file).
    print("\n[OK] Model downloaded and loads successfully.")


if __name__ == "__main__":
    main()
