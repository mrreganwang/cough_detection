#!/usr/bin/env python3
"""
record_samples.py
-----------------
Interactive script to record your own cough / non-cough samples.

This is the fastest way to improve accuracy for YOUR environment:
  1. Record ~20 of your own coughs → data/raw/cough/
  2. Record ~40 samples of background noise in your space → data/raw/negative/
  3. Re-run prepare_dataset.py and train.py

Usage:
  python src/record_samples.py --class cough     # record cough samples
  python src/record_samples.py --class negative  # record non-cough samples
  python src/record_samples.py --class both      # interactive guided session
"""

import time
import argparse
import numpy as np
from pathlib import Path
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).parent.parent
SR = 22050


def record_clip(duration: float = 3.0, sr: int = SR, device=None) -> np.ndarray:
    """Record a single clip of `duration` seconds."""
    n = int(duration * sr)
    audio = sd.rec(n, samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0]


def save_clip(audio: np.ndarray, out_dir: Path, prefix: str, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{index:04d}.wav"
    sf.write(path, audio, SR)
    return path


def record_session(label: str, out_dir: Path, prefix: str, target_count: int = 20, device=None):
    print(f"\n{'='*55}")
    print(f"Recording {label} samples → {out_dir}")
    print(f"Target: {target_count} clips")
    print(f"{'='*55}")
    print("Instructions:")
    if label == "cough":
        print("  - Press ENTER, then cough naturally into the mic")
        print("  - Try different types: dry cough, wet cough, single, double")
        print("  - Each recording is 2 seconds")
    else:
        print("  - Press ENTER, then make the target sound or just wait")
        print("  - Include: talking, background TV/music, typing, breathing")
        print("  - Each recording is 3 seconds")
    print("\nPress Ctrl+C to stop early.\n")

    duration = 2.0 if label == "cough" else 3.0
    index = len(list(out_dir.glob("*.wav")))  # continue from existing count
    saved = 0

    try:
        while saved < target_count:
            input(f"  [{saved+1}/{target_count}] Press ENTER to record ({duration}s)... ")
            print(f"  🔴 Recording...", end="", flush=True)
            audio = record_clip(duration=duration, device=device)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                print(" ✗  Too quiet! Check your microphone.")
                continue
            path = save_clip(audio, out_dir, prefix, index)
            index += 1
            saved += 1
            print(f" ✓  Saved ({rms:.4f} RMS) → {path.name}")
    except KeyboardInterrupt:
        print(f"\n  Stopped. Recorded {saved} clips.")

    return saved


def guided_session(device=None):
    """Walk the user through recording a balanced set of samples."""
    print("\nGuided Recording Session")
    print("=" * 55)
    print("We'll record coughs first, then non-cough sounds.")
    print("The more variety you record, the better the accuracy.\n")

    # Coughs
    n_cough = record_session(
        "cough",
        ROOT / "data" / "raw" / "cough",
        "my_cough",
        target_count=25,
        device=device,
    )

    # Non-cough sounds
    print("\n\nNow let's record some NON-COUGH sounds for contrast.")
    print("These are just as important as the cough examples!\n")

    negatives = [
        ("talking", "Talk naturally for 3 seconds (read something aloud)"),
        ("throat_clear", "Clear your throat (NOT a full cough)"),
        ("breathing", "Breathe audibly near the mic"),
        ("silence", "Just stay quiet, let background noise record"),
        ("laugh", "Laugh or chuckle"),
    ]

    for neg_type, instruction in negatives:
        print(f"\n  Next: {neg_type}")
        print(f"  → {instruction}")
        input("  Press ENTER when ready (records 3s)... ")
        for i in range(4):
            if i > 0:
                input(f"    Clip {i+1}/4: Press ENTER... ")
            audio = record_clip(duration=3.0, device=device)
            save_clip(audio, ROOT / "data" / "raw" / "negative", f"recorded_{neg_type}", i)
        print(f"  ✓  Recorded 4 {neg_type} clips")

    print("\n\n✅  Recording session complete!")
    print(f"   Recorded {n_cough} cough clips")
    print("   Recorded ~20 non-cough clips")
    print("\nNow retrain:")
    print("   python src/prepare_dataset.py")
    print("   python src/train.py")


def main():
    parser = argparse.ArgumentParser(description="Record training samples for cough detector")
    parser.add_argument(
        "--class",
        choices=["cough", "negative", "both"],
        default="both",
        dest="cls",
        help="Which class to record (default: both = guided session)",
    )
    parser.add_argument("--count", type=int, default=20, help="Number of clips to record")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']}")
        return

    if args.cls == "both":
        guided_session(device=args.device)
    elif args.cls == "cough":
        record_session("cough", ROOT / "data" / "raw" / "cough", "my_cough",
                       target_count=args.count, device=args.device)
    else:
        record_session("negative", ROOT / "data" / "raw" / "negative", "recorded",
                       target_count=args.count, device=args.device)


if __name__ == "__main__":
    main()
