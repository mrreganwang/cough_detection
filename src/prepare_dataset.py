#!/usr/bin/env python3
"""
prepare_dataset.py
------------------
Converts raw audio files into log-mel spectrogram features for training.

Handles all sources produced by download_data.py:
  - ESC-50 cough clips (5s each, prefix: esc50_)
  - Kaggle: himanshu007121/cough-audio-dataset (prefix: kaggle_cad_)
  - Kaggle: COUGHVID WAV (prefix: coughvid_, up to ~30s each)
  - Kaggle: COVID-19 cough classification (prefix: covid19_)
  - MUSAN speech negatives (prefix: musan_speech_)
  - Synthetic noise negatives (prefix: synthetic_)
  - User-recorded samples (prefix: my_)

Memory-safe design:
  - Hard cap of MAX_COUGH_SPECTROGRAMS and MAX_NEG_SPECTROGRAMS to bound RAM
  - Files are randomly sampled (not truncated) so all sources stay represented
  - Augmentation is OFF by default; only enabled if dataset < 500 specs or
    explicitly requested with --augment-factor N
  - Streams features into an memmap file instead of holding everything in RAM
"""

import sys
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import librosa
import librosa.effects

ROOT = Path(__file__).parent.parent
RAW_COUGH    = ROOT / "data" / "raw" / "cough"
RAW_NEG      = ROOT / "data" / "raw" / "negative"
# User-recorded negatives are identified by the "recorded_" filename prefix
PROC_DIR     = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Minimum number of spectrograms guaranteed from user-recorded negatives.
# Recorded clips are oversampled (repeated) until this floor is reached.
# This ensures the model sees your specific false-positive sounds enough
# times to learn to reject them, even if you only recorded a handful of clips.
MIN_RECORDED_NEG_SPECS = 1000

# ---- Feature extraction parameters -----------------------------------------
SR         = 22050
DURATION   = 1.0
N_SAMPLES  = int(SR * DURATION)
N_FFT      = 1024
HOP_LENGTH = 256
N_MELS     = 128
FMIN       = 80
FMAX       = 8000
N_FRAMES   = N_SAMPLES // HOP_LENGTH + 1   # ~87 time frames
# -----------------------------------------------------------------------------

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

# Hard memory caps. Each spectrogram is (128, 87) float32 = ~43 KB.
# 15k cough specs  = ~650 MB  (safe)
# 45k neg specs    = ~1.9 GB  (safe)
# Total + overhead = ~3 GB    (fine on 8 GB+ Mac)
MAX_COUGH_SPECTROGRAMS = 15_000
MAX_NEG_SPECTROGRAMS   = 45_000

# Augmentation is OFF by default.
# With large diverse datasets (COUGHVID, Kaggle) real variety already exists.
# Use --augment-factor N to enable it only if your cough data is small/homogeneous.

# Per-source max windows from a single file (prevents one long file dominating)
SOURCE_WINDOW_CAPS = {
    "esc50":      4,
    "kaggle_cad": 6,
    "coughvid":   8,
    "covid19":    6,
    "musan":      5,
    "synthetic":  3,
    "my":         8,
    "freesound":  5,
    "reviewed":   6,
    "recorded":   8,
    "default":    5,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---- Utilities --------------------------------------------------------------

def get_source(path: Path) -> str:
    name = path.name
    for prefix in SOURCE_WINDOW_CAPS:
        if prefix == "default":
            continue
        if name.startswith(prefix):
            return prefix
    return "default"


def collect_files(directory: Path) -> list[Path]:
    return sorted(
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    )


def load_audio(path: Path) -> np.ndarray | None:
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        if len(audio) < SR * 0.1:
            return None
        return audio
    except Exception as e:
        print(f"  Warning: could not load {path.name}: {e}")
        return None


def extract_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Extract a log-mel spectrogram. Must stay identical to realtime_detect.py.
    Output shape: (N_MELS, N_FRAMES) = (128, 87)
    """
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    log_mel = (log_mel + 40.0) / 40.0
    return log_mel.astype(np.float32)


def sliding_windows(audio: np.ndarray) -> list[np.ndarray]:
    hop = N_SAMPLES // 2
    wins = []
    for start in range(0, len(audio) - N_SAMPLES + 1, hop):
        wins.append(audio[start : start + N_SAMPLES])
    if len(audio) > N_SAMPLES:
        wins.append(audio[-N_SAMPLES:])
    return wins


# ---- Augmentation -----------------------------------------------------------

def aug_add_noise(a):
    snr = random.uniform(10, 30)
    p   = np.mean(a**2) + 1e-9
    return (a + np.random.normal(0, np.sqrt(p / 10**(snr/10)), len(a))).astype(np.float32)

def aug_time_stretch(a):
    s = librosa.effects.time_stretch(a, rate=random.uniform(0.8, 1.25))
    if len(s) < N_SAMPLES: s = np.pad(s, (0, N_SAMPLES - len(s)))
    return s[:N_SAMPLES].astype(np.float32)

def aug_pitch_shift(a):
    return librosa.effects.pitch_shift(a, sr=SR, n_steps=random.uniform(-2, 2)).astype(np.float32)

def aug_time_mask(a):
    a = a.copy()
    ml = int(len(a) * random.uniform(0, 0.15))
    if ml > 0:
        s = random.randint(0, len(a) - ml)
        a[s:s+ml] = 0.0
    return a

def aug_room(a):
    d = random.randint(SR//50, SR//10)
    r = np.zeros(d+1, dtype=np.float32); r[0]=1.0; r[-1]=random.uniform(0.2,0.5)
    return np.convolve(a, r)[:len(a)].astype(np.float32)

_AUGS = [aug_add_noise, aug_time_stretch, aug_pitch_shift, aug_time_mask, aug_room]

def augment_window(audio: np.ndarray) -> np.ndarray:
    a = audio.copy()
    for fn in random.sample(_AUGS, k=random.randint(1, 3)):
        try: a = fn(a)
        except Exception: pass
    return a


# ---- Processing -------------------------------------------------------------

def determine_aug_factor(n_real_specs: int, user_override: int | None) -> int:
    """Returns 0 (no augmentation) unless the user explicitly requests it,
    or if the dataset is very small (< 500 specs)."""
    if user_override is not None:
        return user_override
    if n_real_specs < 500:
        print(f"  Small dataset ({n_real_specs} specs) -- enabling 4x augmentation.")
        return 4
    return 0


def oversample_files(file_list: list[Path], min_specs: int) -> list[Path]:
    """
    Repeat files from file_list until we have enough to produce at least
    min_specs spectrograms (estimated). Returns the expanded list.
    This guarantees user-recorded clips appear frequently enough in training
    to actually influence the model.
    """
    if not file_list:
        return []
    # Rough estimate: assume SOURCE_WINDOW_CAPS["recorded"] windows per file
    cap = SOURCE_WINDOW_CAPS["recorded"]
    estimated_specs = len(file_list) * cap
    if estimated_specs >= min_specs:
        return file_list
    repeats = -(-min_specs // max(estimated_specs, 1))  # ceiling division
    oversampled = (file_list * repeats)[:repeats * len(file_list)]
    print(f"    Oversampling {len(file_list)} recorded files x{repeats} "
          f"-> ~{len(oversampled) * cap} specs (target: {min_specs})")
    return oversampled


def process_files(
    file_list: list[Path],
    label: int,
    max_specs: int,
    augment: bool = False,
    aug_factor: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Process files into spectrograms up to max_specs total (including augments).
    Uses a memory-mapped temp file so RAM stays bounded.

    Returns:
      X : (N, N_MELS, N_FRAMES) float32
      y : (N,) int8
      source_counts : dict
    """
    # Pre-allocate a memmap on disk (avoids holding everything in RAM)
    mmap_path = PROC_DIR / f"_tmp_label{label}.dat"
    mmap = np.memmap(mmap_path, dtype=np.float32, mode="w+",
                     shape=(max_specs, N_MELS, N_FRAMES))

    n_written = 0
    source_counts: dict[str, int] = defaultdict(int)
    skipped_silent = 0

    random.shuffle(file_list)

    for path in tqdm(file_list, desc=f"  label={label}", unit="file"):
        if n_written >= max_specs:
            break

        audio = load_audio(path)
        if audio is None:
            continue

        source = get_source(path)
        cap    = SOURCE_WINDOW_CAPS.get(source, SOURCE_WINDOW_CAPS["default"])
        wins   = sliding_windows(audio)[:cap]

        for win in wins:
            if n_written >= max_specs:
                break
            if np.sqrt(np.mean(win**2)) < 0.001:
                skipped_silent += 1
                continue

            mmap[n_written] = extract_spectrogram(win)
            n_written += 1
            source_counts[source] += 1

            if augment and aug_factor > 0:
                for _ in range(aug_factor):
                    if n_written >= max_specs:
                        break
                    mmap[n_written] = extract_spectrogram(augment_window(win))
                    n_written += 1
                    source_counts[f"{source}_aug"] += 1

    if skipped_silent:
        print(f"    (skipped {skipped_silent} silent windows)")

    # Copy out only the written portion, then delete the temp file
    X = np.array(mmap[:n_written])
    y = np.full(n_written, label, dtype=np.int8)
    del mmap
    mmap_path.unlink(missing_ok=True)

    return X, y, dict(source_counts)


def print_breakdown(counts: dict, title: str):
    total = sum(counts.values())
    print(f"  {title} ({total} spectrograms):")
    for src, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(30, max(1, int(30 * n / max(total, 1))))
        print(f"    {src:<24} {n:>6}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Prepare cough detection dataset")
    parser.add_argument("--augment-factor", type=int, default=None,
                        help="Augmented copies per cough window. Default: 0 (off). Use only for small datasets.")
    parser.add_argument("--max-cough", type=int, default=MAX_COUGH_SPECTROGRAMS,
                        help=f"Max cough spectrograms (default: {MAX_COUGH_SPECTROGRAMS})")
    parser.add_argument("--max-neg",   type=int, default=MAX_NEG_SPECTROGRAMS,
                        help=f"Max negative spectrograms (default: {MAX_NEG_SPECTROGRAMS})")
    parser.add_argument("--min-recorded-specs", type=int, default=MIN_RECORDED_NEG_SPECS,
                        help=f"Min spectrograms from recorded negatives via oversampling (default: {MIN_RECORDED_NEG_SPECS})")
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    args = parser.parse_args()

    print("Cough Detector -- Dataset Preparation")
    print("=" * 60)
    print(f"  Sample rate:   {SR} Hz")
    print(f"  Window:        {DURATION}s ({N_SAMPLES} samples)")
    print(f"  Spectrogram:   {N_MELS} mels x {N_FRAMES} frames")
    print(f"  Freq range:    {FMIN}-{FMAX} Hz")
    print(f"  Max cough:     {args.max_cough:,}")
    print(f"  Max negative:  {args.max_neg:,}")
    approx_ram = (args.max_cough + args.max_neg) * N_MELS * N_FRAMES * 4 / 1e9
    print(f"  Approx RAM:    ~{approx_ram:.1f} GB")
    print()

    cough_files = collect_files(RAW_COUGH)
    neg_files   = collect_files(RAW_NEG)

    # Quick file-level source breakdown
    def src_breakdown(files):
        d = defaultdict(int)
        for f in files: d[get_source(f)] += 1
        return dict(d)

    print(f"Raw files: {len(cough_files)} cough, {len(neg_files)} negative")
    for src, n in sorted(src_breakdown(cough_files).items(), key=lambda x: -x[1]):
        print(f"  cough/{src:<22} {n:>5} files")
    for src, n in sorted(src_breakdown(neg_files).items(), key=lambda x: -x[1]):
        print(f"  neg/{src:<24} {n:>5} files")
    print()

    if not cough_files:
        print("No cough files found. Run: python src/download_data.py")
        sys.exit(1)

    # ---- First pass: count real cough specs to set aug factor ---------------
    # Quick estimate: sample 200 files to get avg windows/file
    sample = random.sample(cough_files, min(200, len(cough_files)))
    total_wins = 0
    for path in tqdm(sample, desc="  Estimating windows/file", leave=False):
        audio = load_audio(path)
        if audio is None: continue
        cap  = SOURCE_WINDOW_CAPS.get(get_source(path), SOURCE_WINDOW_CAPS["default"])
        total_wins += min(len(sliding_windows(audio)), cap)
    avg_wins = total_wins / max(len(sample), 1)
    est_real_specs = int(avg_wins * len(cough_files))
    est_real_specs = min(est_real_specs, args.max_cough)

    aug_factor = determine_aug_factor(est_real_specs, args.augment_factor)
    print(f"Estimated real cough specs: ~{est_real_specs:,}")
    print(f"Auto aug factor: {aug_factor}x  "
          f"({'user override' if args.augment_factor else 'auto-scaled'})")
    print()

    # ---- Process coughs -----------------------------------------------------
    print("Extracting cough features...")
    X_cough, y_cough, cough_counts = process_files(
        cough_files, label=1,
        max_specs=args.max_cough,
        augment=True, aug_factor=aug_factor,
    )
    print(f"  -> {len(X_cough):,} cough spectrograms")
    print_breakdown(cough_counts, "Cough sources")

    # ---- Process negatives --------------------------------------------------
    # Split recorded clips out so we can oversample them independently
    recorded_files = [f for f in neg_files if f.name.startswith("recorded_")]
    other_neg_files = [f for f in neg_files if not f.name.startswith("recorded_")]

    # Cap bulk negatives to 3x cough count
    max_neg = min(args.max_neg, len(X_cough) * 3)

    # Reserve space for oversampled recorded clips, reduce bulk cap to match
    recorded_budget = min(args.min_recorded_specs, max_neg // 4) if recorded_files else 0
    bulk_cap = max_neg - recorded_budget

    print(f"\nExtracting bulk negative features (cap: {bulk_cap:,})...")
    X_neg, y_neg, neg_counts = process_files(
        other_neg_files, label=0,
        max_specs=bulk_cap,
        augment=False,
    )
    print(f"  -> {len(X_neg):,} bulk negative spectrograms")
    print_breakdown(neg_counts, "Bulk negative sources")

    # ---- Oversampled recorded negatives -------------------------------------
    if recorded_files:
        print(f"\nFound {len(recorded_files)} user-recorded negative files (recorded_ prefix)")
        oversampled = oversample_files(recorded_files, args.min_recorded_specs)
        print(f"Extracting recorded negative features (budget: {recorded_budget:,})...")
        X_rec, y_rec, rec_counts = process_files(
            oversampled, label=0,
            max_specs=recorded_budget,
            augment=False,
        )
        print(f"  -> {len(X_rec):,} recorded negative spectrograms")
        print_breakdown(rec_counts, "Recorded negative sources")
        X_neg = np.concatenate([X_neg, X_rec], axis=0)
        y_neg = np.concatenate([y_neg, y_rec], axis=0)
        del X_rec, y_rec
        neg_counts.update(rec_counts)
    else:
        print("\n(No user-recorded negatives found in data/raw/negative/recorded/)")
        print("  If you have false positives, run:")
        print("  python src/realtime_detect.py --record-negatives")

    print(f"\nTotal negatives: {len(X_neg):,}")

    # ---- Combine and split --------------------------------------------------
    print("\nCombining and splitting...")
    X = np.concatenate([X_cough, X_neg], axis=0)
    y = np.concatenate([y_cough, y_neg], axis=0).astype(np.int8)

    # Free the per-class arrays before allocating more
    del X_cough, y_cough, X_neg, y_neg

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    N       = len(X)
    n_test  = int(N * args.test_split)
    n_val   = int(N * args.val_split)
    n_train = N - n_val - n_test

    print(f"\nDataset: {N:,} total  "
          f"({int(y.sum()):,} coughs {100*y.mean():.1f}%  /  "
          f"{int((y==0).sum()):,} non-coughs {100*(1-y.mean()):.1f}%)")
    print(f"  Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    approx_gb = X.nbytes / 1e9
    print(f"  Array size: {approx_gb:.2f} GB")

    out_path = PROC_DIR / "dataset.npz"
    print(f"\nSaving to {out_path} ...")
    np.savez_compressed(
        out_path,
        X_train=X[:n_train],          y_train=y[:n_train],
        X_val  =X[n_train:n_train+n_val], y_val=y[n_train:n_train+n_val],
        X_test =X[n_train+n_val:],    y_test=y[n_train+n_val:],
        sr=np.array(SR), n_mels=np.array(N_MELS),
        n_samples=np.array(N_SAMPLES), fmin=np.array(FMIN),
        fmax=np.array(FMAX), n_fft=np.array(N_FFT),
        hop_length=np.array(HOP_LENGTH),
    )
    print("Done.")
    print("\nNext step: python src/train.py")


if __name__ == "__main__":
    main()
