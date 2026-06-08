#!/usr/bin/env python3
"""
download_data.py
----------------
Downloads free, publicly available audio datasets for training the cough detector.

Sources:
  1. ESC-50         — 40 cough clips + ~600 hard/ambient negatives (no login needed)
                      https://github.com/karolpiczak/ESC-50
  2. Kaggle datasets — requires a free Kaggle account + API key (kaggle.json)
       a) himanshu007121/cough-audio-dataset  — labelled cough / not_cough WAV files
       b) nasrulhakim86/coughvid-wav          — COUGHVID crowdsourced cough recordings
       c) andrewmvd/covid19-cough-audio-classification — 2800 cough recordings
  3. MUSAN          — speech negatives (~2 GB, optional, skip with --skip-musan)
  4. Synthetic      — generated noise/burst negatives (always created, no download)

─── Kaggle setup (one-time, free) ──────────────────────────────────────────────
  1. Create a free account at https://www.kaggle.com
  2. Go to https://www.kaggle.com/settings → API → "Create New Token"
  3. This downloads kaggle.json — place it at:
       Mac/Linux:  ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
       Windows:    C:\\Users\\<you>\\.kaggle\\kaggle.json
  4. Install the Kaggle CLI:  pip install kaggle
  5. Run:  python src/download_data.py
     (Kaggle datasets are downloaded automatically if kaggle.json is found)
────────────────────────────────────────────────────────────────────────────────

Usage:
  python src/download_data.py                  # downloads everything
  python src/download_data.py --skip-musan     # skip the large MUSAN download
  python src/download_data.py --skip-kaggle    # skip Kaggle (no API key available)
  python src/download_data.py --freesound-key KEY   # also pull FreeSound clips
"""

import os
import sys
import csv
import json
import shutil
import zipfile
import tarfile
import argparse
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
RAW_COUGH = ROOT / "data" / "raw" / "cough"
RAW_NEG   = ROOT / "data" / "raw" / "negative"
TMP       = ROOT / "data" / "tmp"

for d in (RAW_COUGH, RAW_NEG, TMP):
    d.mkdir(parents=True, exist_ok=True)

# ── ESC-50 class index reference ─────────────────────────────────────────────
# Animals 0–9 | Natural 10–19 | Human non-speech 20–29
# Interior 30–39 | Exterior 40–49
#
# Human non-speech:
#   20=crying_baby  21=sneezing   22=clapping   23=breathing  24=coughing
#   25=footsteps    26=laughing   27=brushing_teeth  28=snoring  29=drinking_sipping

ESC50_COUGH_CLASSES = {"coughing": 24}

ESC50_HARD_NEGATIVE_CLASSES = {
    "sneezing":    21,   # closest acoustic match to cough
    "clapping":    22,   # transient burst
    "breathing":   23,   # respiratory
    "laughing":    26,   # vocal bursts
    "snoring":     28,   # respiratory
    "crying_baby": 20,   # vocal bursts
    "dog":          0,   # barks similar to coughs
    "rooster":      1,   # harsh calls
    "frog":         4,
    "crow":         9,
}

ESC50_AMBIENT_CLASSES = {
    "door_knock":       30,
    "mouse_click":      31,
    "keyboard_typing":  32,
    "door_wood_creaks": 33,
    "can_opening":      34,
    "washing_machine":  35,
    "vacuum_cleaner":   36,
    "clock_alarm":      37,
    "clock_tick":       38,
    "glass_breaking":   39,
    "helicopter":       40,
    "chainsaw":         41,
    "siren":            42,
    "car_horn":         43,
    "engine":           44,
    "rain":             10,
    "sea_waves":        11,
    "crackling_fire":   12,
    "crickets":         13,
    "chirping_birds":   14,
    "water_drops":      15,
    "wind":             16,
}


# ── Kaggle dataset specs ──────────────────────────────────────────────────────
# Each entry: (kaggle_slug, description, handler_function)
KAGGLE_DATASETS = [
    {
        "slug":    "himanshu007121/cough-audio-dataset",
        "desc":    "Cough Audio Dataset (labelled cough/not_cough WAVs)",
        "handler": "handle_cough_audio_dataset",
    },
    {
        "slug":    "nasrulhakim86/coughvid-wav",
        "desc":    "COUGHVID crowdsourced cough recordings (WAV)",
        "handler": "handle_coughvid_wav",
    },
    {
        "slug":    "andrewmvd/covid19-cough-audio-classification",
        "desc":    "COVID-19 Cough Audio Classification (~2800 clips)",
        "handler": "handle_covid19_cough",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=desc or dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def copy_audio(src: Path, dest_dir: Path, prefix: str) -> bool:
    """Copy an audio file into dest_dir with a prefixed name. Returns True on success."""
    ext = src.suffix.lower()
    if ext not in {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}:
        return False
    dest = dest_dir / f"{prefix}_{src.name}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return True


def check_kaggle_available() -> bool:
    """Return True if the kaggle CLI is installed and kaggle.json is present."""
    # Check CLI
    result = subprocess.run(["kaggle", "--version"], capture_output=True)
    if result.returncode != 0:
        return False
    # Check credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        # Also check KAGGLE_USERNAME / KAGGLE_KEY env vars
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            return False
    return True


def kaggle_download(slug: str, dest_dir: Path) -> Path:
    """
    Download a Kaggle dataset zip into dest_dir using the kaggle CLI.
    Returns path to the extracted directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = slug.split("/")[1] + ".zip"
    zip_path = dest_dir / zip_name

    if not zip_path.exists():
        print(f"    Downloading {slug} ...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"    ✗ kaggle CLI failed for {slug}")
            return None

    # Find the downloaded zip (kaggle sometimes names it differently)
    zips = list(dest_dir.glob("*.zip"))
    if not zips:
        print(f"    ✗ No zip found in {dest_dir} after download")
        return None

    zip_path = zips[0]
    extract_dir = dest_dir / zip_path.stem
    if not extract_dir.exists():
        print(f"    Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    return extract_dir


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle dataset handlers
# Each function receives the extracted directory and copies files into
# RAW_COUGH / RAW_NEG with appropriate prefixes.
# ─────────────────────────────────────────────────────────────────────────────

def handle_cough_audio_dataset(extract_dir: Path) -> tuple[int, int]:
    """
    himanshu007121/cough-audio-dataset
    Structure (typical):
      cough/          *.wav  — positive cough samples
      not_cough/      *.wav  — negative (non-cough) samples
    """
    cough_count = neg_count = 0

    # Walk the whole tree looking for cough / not_cough folders
    for folder in extract_dir.rglob("*"):
        if not folder.is_dir():
            continue
        name_lower = folder.name.lower().replace(" ", "_").replace("-", "_")

        if name_lower in ("cough", "coughs", "positive", "pos"):
            for f in folder.iterdir():
                if copy_audio(f, RAW_COUGH, "kaggle_cad"):
                    cough_count += 1

        elif name_lower in ("not_cough", "no_cough", "negative", "neg", "non_cough",
                            "noise", "background", "other"):
            for f in folder.iterdir():
                if copy_audio(f, RAW_NEG, "kaggle_cad_neg"):
                    neg_count += 1

    # Fallback: if no named folders matched, treat ALL audio as coughs
    # (this dataset is primarily a cough dataset)
    if cough_count == 0:
        print("    (No cough/ folder found — treating all audio as cough samples)")
        for f in extract_dir.rglob("*.wav"):
            if copy_audio(f, RAW_COUGH, "kaggle_cad"):
                cough_count += 1

    return cough_count, neg_count


def handle_coughvid_wav(extract_dir: Path) -> tuple[int, int]:
    """
    nasrulhakim86/coughvid-wav
    Structure:
      *.wav files (UUID-named), plus metadata.csv with columns:
        uuid, cough_detected, status, ...
      status: 'COVID-19' | 'healthy' | 'symptomatic' | NaN
      cough_detected: 0.0–1.0 confidence that the clip contains a cough

    We use cough_detected >= 0.8 as positive, < 0.3 as negative.
    """
    cough_count = neg_count = 0

    # Find metadata CSV
    meta_files = list(extract_dir.rglob("metadata.csv")) + \
                 list(extract_dir.rglob("*.csv"))
    meta = {}

    if meta_files:
        meta_path = meta_files[0]
        print(f"    Reading metadata: {meta_path.name}")
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uuid", "").strip()
                if not uid:
                    continue
                try:
                    confidence = float(row.get("cough_detected", 0) or 0)
                except ValueError:
                    confidence = 0.0
                meta[uid] = confidence

    wav_files = list(extract_dir.rglob("*.wav"))
    print(f"    Found {len(wav_files)} WAV files, {len(meta)} metadata entries")

    for wav in wav_files:
        uid = wav.stem  # filename without extension = UUID

        if meta:
            confidence = meta.get(uid, -1)
            if confidence >= 0.8:
                if copy_audio(wav, RAW_COUGH, "coughvid"):
                    cough_count += 1
            elif confidence >= 0 and confidence < 0.3:
                if copy_audio(wav, RAW_NEG, "coughvid_neg"):
                    neg_count += 1
            # clips with 0.3–0.8 confidence are ambiguous — skip them
        else:
            # No metadata available — treat everything as coughs
            # (COUGHVID is a cough dataset, most clips are coughs)
            if copy_audio(wav, RAW_COUGH, "coughvid"):
                cough_count += 1

    return cough_count, neg_count


def handle_covid19_cough(extract_dir: Path) -> tuple[int, int]:
    """
    andrewmvd/covid19-cough-audio-classification
    Structure:
      asthma_sound_dataset/
        cough/            *.wav
        breathing/        *.wav  → useful as negatives
        sneezing/         *.wav  → hard negatives
      covid19_cough/
        cough_*.wav
      metadata.csv or similar

    All cough folders → RAW_COUGH
    Breathing/sneezing folders → RAW_NEG (hard negatives)
    """
    cough_count = neg_count = 0

    for folder in extract_dir.rglob("*"):
        if not folder.is_dir():
            continue
        name = folder.name.lower().replace(" ", "_").replace("-", "_")

        # Positive: any folder named 'cough' or 'coughs' or 'covid*cough'
        if "cough" in name and "no" not in name and "not" not in name:
            for f in folder.iterdir():
                if copy_audio(f, RAW_COUGH, f"covid19_{name}"):
                    cough_count += 1

        # Hard negatives: breathing, sneezing
        elif name in ("breathing", "breath", "sneezing", "sneeze", "snore",
                      "non_cough", "no_cough", "not_cough", "healthy",
                      "background", "noise"):
            for f in folder.iterdir():
                if copy_audio(f, RAW_NEG, f"covid19_{name}"):
                    neg_count += 1

    # Also catch loose cough WAVs at root level
    for f in extract_dir.glob("cough*.wav"):
        if copy_audio(f, RAW_COUGH, "covid19"):
            cough_count += 1

    return cough_count, neg_count


# ─────────────────────────────────────────────────────────────────────────────
# Main dataset downloaders
# ─────────────────────────────────────────────────────────────────────────────

def download_esc50() -> tuple[int, int]:
    """Download ESC-50 and sort cough (target=24) + negative clips."""
    esc50_zip = TMP / "ESC-50-master.zip"
    if not esc50_zip.exists():
        print("\n[ESC-50] Downloading (~600 MB)...")
        download_file(
            "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
            esc50_zip, "ESC-50",
        )
    else:
        print("\n[ESC-50] Already downloaded, skipping.")

    esc50_dir = TMP / "ESC-50-master"
    if not esc50_dir.exists():
        print("    Extracting...")
        with zipfile.ZipFile(esc50_zip) as zf:
            zf.extractall(TMP)

    meta_csv  = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    all_cough = set(ESC50_COUGH_CLASSES.values())
    all_neg   = set(list(ESC50_HARD_NEGATIVE_CLASSES.values()) +
                    list(ESC50_AMBIENT_CLASSES.values()))
    assert not (all_cough & all_neg), \
        f"ESC-50 class overlap: {all_cough & all_neg}"

    cough_count = neg_count = 0
    with open(meta_csv) as f:
        for row in csv.DictReader(f):
            target = int(row["target"])
            src    = audio_dir / row["filename"]
            if not src.exists():
                continue
            if target in all_cough:
                shutil.copy2(src, RAW_COUGH / f"esc50_{row['filename']}")
                cough_count += 1
            elif target in all_neg:
                shutil.copy2(src, RAW_NEG / f"esc50_{row['filename']}")
                neg_count += 1

    print(f"    ✓ ESC-50: {cough_count} cough clips, {neg_count} negative clips")
    return cough_count, neg_count


def download_kaggle_datasets() -> tuple[int, int]:
    """Download all configured Kaggle datasets."""
    if not check_kaggle_available():
        print("\n[Kaggle] ⚠️  Skipping — kaggle CLI not found or no credentials.")
        print("   To enable Kaggle downloads:")
        print("     1. pip install kaggle")
        print("     2. Get your API token: https://www.kaggle.com/settings → API")
        print("     3. Save it to ~/.kaggle/kaggle.json")
        print("     4. chmod 600 ~/.kaggle/kaggle.json")
        print("     5. Re-run this script")
        return 0, 0

    print("\n[Kaggle] Kaggle credentials found — downloading cough datasets...")
    total_cough = total_neg = 0
    handlers = {
        "handle_cough_audio_dataset": handle_cough_audio_dataset,
        "handle_coughvid_wav":        handle_coughvid_wav,
        "handle_covid19_cough":       handle_covid19_cough,
    }

    for ds in KAGGLE_DATASETS:
        slug    = ds["slug"]
        desc    = ds["desc"]
        handler = handlers[ds["handler"]]
        print(f"\n  → {desc}")
        print(f"    ({slug})")

        extract_dir = kaggle_download(slug, TMP / "kaggle" / slug.split("/")[1])
        if extract_dir is None:
            print(f"    ✗ Download failed, skipping.")
            continue

        try:
            c, n = handler(extract_dir)
            print(f"    ✓ {c} cough clips, {n} negative clips")
            total_cough += c
            total_neg   += n
        except Exception as e:
            print(f"    ✗ Handler error: {e}")

    return total_cough, total_neg


def download_musan_speech_subset() -> int:
    """Download MUSAN speech subset for hard speech negatives."""
    print("\n[MUSAN] Downloading speech negatives (~11 GB total, Ctrl-C to skip)...")
    tar_path = TMP / "musan.tar.gz"
    if not tar_path.exists():
        try:
            download_file(
                "https://www.openslr.org/resources/17/musan.tar.gz",
                tar_path, "MUSAN",
            )
        except KeyboardInterrupt:
            print("\n    Skipped. Add speech .wav files to data/raw/negative/ manually.")
            return 0

    musan_dir = TMP / "musan"
    if not musan_dir.exists():
        print("    Extracting MUSAN...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(TMP)

    speech_dir = musan_dir / "speech"
    if not speech_dir.exists():
        print("    MUSAN speech/ dir not found, skipping.")
        return 0

    count = 0
    for wav in list(speech_dir.rglob("*.wav"))[:500]:
        dest = RAW_NEG / f"musan_speech_{wav.name}"
        if not dest.exists():
            shutil.copy2(wav, dest)
            count += 1

    print(f"    ✓ MUSAN: {count} speech clips added as negatives")
    return count


def create_synthetic_negatives() -> int:
    """Generate simple synthetic noise/burst negatives (no download needed)."""
    import numpy as np
    import soundfile as sf

    print("\n[Synthetic] Generating ambient negatives...")
    SR, n_clips, count = 22050, 50, 0
    rng = np.random.default_rng(42)

    for i in range(n_clips):
        dest = RAW_NEG / f"synthetic_{i:04d}.wav"
        if dest.exists():
            count += 1
            continue

        duration = rng.uniform(0.5, 3.0)
        n = int(duration * SR)
        choice = i % 5

        if choice == 0:
            audio = rng.normal(0, 0.05, n).astype(np.float32)
        elif choice == 1:
            white = rng.normal(0, 1, n)
            audio = np.cumsum(white)
            audio = (audio / np.max(np.abs(audio)) * 0.1).astype(np.float32)
        elif choice == 2:
            freq = rng.choice([440, 880, 1000, 1500])
            t = np.linspace(0, duration, n)
            audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        elif choice == 3:
            t = np.linspace(0, duration, n)
            freq = np.linspace(200, 3000, n)
            audio = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        else:
            audio = np.zeros(n, dtype=np.float32)
            for _ in range(rng.integers(1, 4)):
                start = rng.integers(0, max(1, n - SR // 4))
                blen  = rng.integers(SR // 20, SR // 8)
                end   = min(start + blen, n)
                burst = rng.normal(0, 0.5, end - start).astype(np.float32)
                audio[start:end] = burst * np.exp(-np.linspace(0, 10, end - start))

        sf.write(dest, audio, SR)
        count += 1

    print(f"    ✓ {count} synthetic clips")
    return count


def download_freesound_coughs(api_key: str) -> int:
    """Download cough clips from FreeSound (optional, requires free API key)."""
    print("\n[FreeSound] Downloading additional cough clips...")
    headers = {"Authorization": f"Token {api_key}"}
    params  = {
        "query": "cough",
        "filter": "duration:[0.2 TO 5.0] type:wav",
        "fields": "id,name,previews,duration",
        "page_size": 50,
        "sort": "rating_desc",
    }
    try:
        r = requests.get("https://freesound.org/apiv2/search/text/",
                         params=params, headers=headers, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        count = 0
        for item in tqdm(results, desc="FreeSound coughs"):
            url = (item["previews"].get("preview-hq-mp3")
                   or item["previews"].get("preview-lq-mp3"))
            if url:
                dest = RAW_COUGH / f"freesound_{item['id']}.mp3"
                if not dest.exists():
                    try:
                        download_file(url, dest, desc="")
                        count += 1
                    except Exception:
                        pass
        print(f"    ✓ FreeSound: {count} clips")
        return count
    except Exception as e:
        print(f"    ✗ FreeSound failed: {e}")
        return 0


def print_summary():
    exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    cough_files = [f for f in RAW_COUGH.iterdir() if f.suffix.lower() in exts]
    neg_files   = [f for f in RAW_NEG.iterdir()   if f.suffix.lower() in exts]

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Cough clips:    {len(cough_files):>5}")
    print(f"  Negative clips: {len(neg_files):>5}")
    print(f"  Total:          {len(cough_files) + len(neg_files):>5}")
    print()

    # Breakdown by source
    sources_cough = {}
    for f in cough_files:
        src = f.name.split("_")[0]
        sources_cough[src] = sources_cough.get(src, 0) + 1
    print("  Cough sources:")
    for src, n in sorted(sources_cough.items(), key=lambda x: -x[1]):
        print(f"    {src:<20} {n}")

    print()
    if len(cough_files) < 50:
        print("⚠️  Very few cough samples — accuracy will be poor.")
        print("   Set up Kaggle credentials and re-run, or record your own:")
        print("     python src/record_samples.py --class cough")
    elif len(cough_files) < 200:
        print("⚠️  Moderate cough count. More data = better accuracy.")
        print("   Consider adding Kaggle datasets (see instructions above).")
    else:
        print("✅ Good cough sample count.")

    print()
    print("Next step: python src/prepare_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Download cough detector training data")
    parser.add_argument("--skip-musan",   action="store_true",
                        help="Skip the large MUSAN download")
    parser.add_argument("--skip-kaggle",  action="store_true",
                        help="Skip all Kaggle downloads")
    parser.add_argument("--freesound-key", type=str, default=None,
                        help="FreeSound API key (optional)")
    args = parser.parse_args()

    print("Cough Detector — Data Download")
    print("=" * 60)

    download_esc50()

    if not args.skip_kaggle:
        download_kaggle_datasets()
    else:
        print("\n[Kaggle] Skipped (--skip-kaggle).")

    if not args.skip_musan:
        download_musan_speech_subset()
    else:
        print("\n[MUSAN] Skipped (--skip-musan).")

    create_synthetic_negatives()

    if args.freesound_key:
        download_freesound_coughs(args.freesound_key)

    print_summary()


if __name__ == "__main__":
    main()
