#!/usr/bin/env python3
"""
active_learning.py
------------------
Iterative improvement loop: after you've run the detector for a while
and collected false positives (via --record-negatives), this script
helps you incorporate them into retraining.

Workflow:
  1. Run: python src/realtime_detect.py --record-negatives
  2. Let it run and collect any false positives in data/raw/negative/recorded/
  3. Run: python src/active_learning.py --review
     This plays each recording and asks you to label it:
       [k] keep as negative (it was a false positive)
       [d] delete (it was actually a real cough, keep it out)
       [c] move to cough/ (it was a real cough that should be positive)
  4. Run: python src/prepare_dataset.py && python src/train.py
  5. Repeat

This loop is the most powerful way to fix real-world errors.
"""

import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
RECORDED_NEG = ROOT / "data" / "raw" / "negative" / "recorded"
COUGH_DIR = ROOT / "data" / "raw" / "cough"
REVIEWED_NEG = ROOT / "data" / "raw" / "negative"


def play_audio(path: Path):
    """Play an audio file (Mac/Linux)."""
    import subprocess
    import sys
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        try:
            import sounddevice as sd
            import soundfile as sf
            audio, sr = sf.read(path)
            sd.play(audio, sr)
            sd.wait()
        except Exception as e:
            print(f"  (Could not play audio: {e})")


def review_recordings():
    """Interactive review of recorded false positives."""
    files = sorted(RECORDED_NEG.glob("*.wav"))
    if not files:
        print("No recorded files found in", RECORDED_NEG)
        print("Run: python src/realtime_detect.py --record-negatives")
        return

    print(f"\nReviewing {len(files)} recordings from {RECORDED_NEG}")
    print("=" * 55)
    print("Controls:")
    print("  [k] Keep as negative (false positive → good training data)")
    print("  [c] Move to cough/ (real cough — positive label)")
    print("  [d] Delete (ambiguous, don't use)")
    print("  [p] Play again")
    print("  [q] Quit\n")

    kept = moved = deleted = 0

    for i, path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {path.name}")
        play_audio(path)

        while True:
            choice = input("  Action [k/c/d/p/q]: ").strip().lower()
            if choice == "k":
                dest = REVIEWED_NEG / f"reviewed_{path.name}"
                shutil.move(str(path), dest)
                kept += 1
                print(f"  → Kept as negative: {dest.name}")
                break
            elif choice == "c":
                dest = COUGH_DIR / f"reviewed_{path.name}"
                shutil.move(str(path), dest)
                moved += 1
                print(f"  → Moved to cough/: {dest.name}")
                break
            elif choice == "d":
                path.unlink()
                deleted += 1
                print("  → Deleted")
                break
            elif choice == "p":
                play_audio(path)
            elif choice == "q":
                print("\nStopped early.")
                break
            else:
                print("  Invalid choice. Use k, c, d, p, or q.")

        if choice == "q":
            break

    print(f"\n✅  Review complete:")
    print(f"  Kept as negatives: {kept}")
    print(f"  Moved to cough/:   {moved}")
    print(f"  Deleted:           {deleted}")
    print("\nNow retrain:")
    print("  python src/prepare_dataset.py")
    print("  python src/train.py")


def main():
    parser = argparse.ArgumentParser(description="Active learning for cough detector")
    parser.add_argument(
        "--review",
        action="store_true",
        help="Interactively review recorded samples",
    )
    args = parser.parse_args()

    if args.review:
        review_recordings()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
