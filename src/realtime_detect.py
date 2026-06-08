#!/usr/bin/env python3
"""
realtime_detect.py
------------------
Real-time cough detection from microphone input.

Usage:
  python src/realtime_detect.py                    # default settings
  python src/realtime_detect.py --threshold 0.80   # stricter
  python src/realtime_detect.py --list-devices     # show mic options
  python src/realtime_detect.py --device 1         # use device index 1
  python src/realtime_detect.py --record-negatives # save FP audio for retraining

How it works:
  1. Continuously read audio from the microphone in small chunks
  2. Energy gate: skip frames that are near-silent (< RMS threshold)
  3. Buffer audio into 1-second windows with 50% overlap
  4. Extract log-mel spectrogram from each window
  5. Run CNN classifier
  6. If probability > threshold, print "Cough {timestamp}"
  7. Suppress duplicates within a 1.5-second cooldown window

Accuracy tips:
  - Set threshold higher (0.80–0.90) if you get false positives
  - Set threshold lower (0.55–0.65) if you're missing real coughs
  - Run --record-negatives to collect hard negative samples,
    then retrain with the new data
"""

import sys
import json
import time
import queue
import argparse
import threading
import datetime
import numpy as np
from pathlib import Path
from collections import deque

import torch
import sounddevice as sd
import librosa

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

sys.path.insert(0, str(Path(__file__).parent))
from train import CoughCNN

# ─── Default feature parameters (overridden by model config if available) ────
DEFAULT_FEATURE_PARAMS = {
    "sr": 22050,
    "n_mels": 128,
    "n_samples": 22050,     # 1 second
    "fmin": 80,
    "fmax": 8000,
    "n_fft": 1024,
    "hop_length": 256,
}
DEFAULT_THRESHOLD = 0.45
COOLDOWN_SECONDS = 0.75      # Minimum seconds between cough detections
ENERGY_GATE_RMS = 0.003     # Skip near-silent frames


# ─── Feature extraction (must match prepare_dataset.py exactly) ──────────────

def extract_spectrogram(audio: np.ndarray, params: dict) -> np.ndarray:
    """Extract log-mel spectrogram. Must match prepare_dataset.py exactly."""
    n_samples = params["n_samples"]
    sr = params["sr"]

    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        n_mels=params["n_mels"],
        fmin=params["fmin"],
        fmax=params["fmax"],
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    log_mel = (log_mel + 40.0) / 40.0
    return log_mel.astype(np.float32)


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print(f"❌  Model not found at {ckpt_path}")
        print("   Run the full pipeline first:")
        print("     python src/download_data.py")
        print("     python src/prepare_dataset.py")
        print("     python src/train.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = CoughCNN(n_mels=cfg["n_mels"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_params = ckpt.get("feature_params", DEFAULT_FEATURE_PARAMS)

    # Try to load recommended threshold from evaluate.py output
    recommended_threshold = DEFAULT_THRESHOLD
    config_path = MODEL_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_json = json.load(f)
        recommended_threshold = cfg_json.get("recommended_threshold", DEFAULT_THRESHOLD)
        if "feature_params" in cfg_json:
            feature_params = cfg_json["feature_params"]

    return model, feature_params, recommended_threshold


# ─── Audio recording for hard negatives ──────────────────────────────────────

class NegativeRecorder:
    """Records audio clips classified as coughs for manual review / retraining."""
    def __init__(self, out_dir: Path, sr: int):
        import soundfile as sf
        self.sf = sf
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.count = 0

    def save(self, audio: np.ndarray, prob: float):
        fname = self.out_dir / f"fp_{self.count:04d}_p{prob:.2f}.wav"
        self.sf.write(fname, audio, self.sr)
        self.count += 1
        return fname


# ─── Real-time detector ───────────────────────────────────────────────────────

class CoughDetector:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_params: dict,
        threshold: float,
        device: torch.device,
        cooldown: float = COOLDOWN_SECONDS,
        record_negatives: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.fp = feature_params
        self.threshold = threshold
        self.device = device
        self.cooldown = cooldown
        self.verbose = verbose

        self.sr = feature_params["sr"]
        self.n_samples = feature_params["n_samples"]  # 1 second of audio
        self.hop_samples = self.n_samples // 2         # 50% overlap

        # Ring buffer: hold enough audio for 2 windows
        self.buffer = deque(maxlen=self.n_samples * 2)
        self.samples_since_last_hop = 0

        self.last_detection_time = 0.0
        self.audio_queue = queue.Queue()
        self.running = False

        if record_negatives:
            neg_dir = ROOT / "data" / "raw" / "negative" / "recorded"
            self.recorder = NegativeRecorder(neg_dir, self.sr)
            print(f"  Recording false positives to {neg_dir}")
        else:
            self.recorder = None

        # For verbose mode: track recent probabilities
        self.recent_probs = deque(maxlen=20)

    def _classify(self, audio_window: np.ndarray) -> float:
        """Run model inference on a 1-second audio window. Returns cough probability."""
        spec = extract_spectrogram(audio_window, self.fp)
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x).squeeze()
            prob = torch.sigmoid(logit).item()
        return prob

    def _process_audio(self):
        """Worker thread: processes audio chunks from the queue."""
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Add chunk to ring buffer
            self.buffer.extend(chunk.tolist())
            self.samples_since_last_hop += len(chunk)

            # Process a window every hop_samples
            if self.samples_since_last_hop >= self.hop_samples and len(self.buffer) >= self.n_samples:
                self.samples_since_last_hop = 0

                audio_window = np.array(list(self.buffer)[-self.n_samples:], dtype=np.float32)

                # Energy gate: skip silent windows
                rms = np.sqrt(np.mean(audio_window ** 2))
                if rms < ENERGY_GATE_RMS:
                    if self.verbose:
                        print(f"\r  [silent rms={rms:.4f}]    ", end="", flush=True)
                    continue

                prob = self._classify(audio_window)
                self.recent_probs.append(prob)

                if self.verbose:
                    bar_len = int(prob * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)
                    marker = " ← COUGH!" if prob >= self.threshold else ""
                    print(f"\r  [{bar}] {prob:.3f}{marker}    ", end="", flush=True)

                if prob >= self.threshold:
                    now = time.time()
                    if now - self.last_detection_time >= self.cooldown:
                        self.last_detection_time = now
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        if self.verbose:
                            print()  # newline before the detection line
                        print(f"Cough {ts}", flush=True)

                        if self.recorder:
                            saved = self.recorder.save(audio_window, prob)
                            if self.verbose:
                                print(f"  [Saved potential FP to {saved.name}]")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — called for each audio chunk."""
        if status:
            pass  # ignore overflow/underflow warnings in normal operation
        # Convert to mono float32
        audio = indata[:, 0].copy().astype(np.float32)
        self.audio_queue.put(audio)

    def start(self, device_index=None):
        """Start listening to the microphone."""
        self.running = True
        self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.process_thread.start()

        # Block size: 512 samples ≈ 23ms at 22050 Hz (low latency)
        block_size = 512
        try:
            with sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=self._audio_callback,
                device=device_index,
            ):
                print("Listening... (Press Ctrl+C to stop)\n")
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.running = False

    def stop(self):
        self.running = False


# ─── CLI ─────────────────────────────────────────────────────────────────────

def list_audio_devices():
    print("\nAvailable audio input devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            default = " ← default" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {d['name']}{default}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time cough detection from microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/realtime_detect.py                    # default settings
  python src/realtime_detect.py --threshold 0.80   # stricter (fewer false positives)
  python src/realtime_detect.py --threshold 0.60   # looser (catch more coughs)
  python src/realtime_detect.py --list-devices     # show available microphones
  python src/realtime_detect.py --device 1         # use mic at index 1
  python src/realtime_detect.py --verbose          # show real-time probability bars
  python src/realtime_detect.py --record-negatives # save detections for retraining
        """,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold 0–1 (default: from model config, ~0.72)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default mic)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show real-time probability bars (useful for calibration)",
    )
    parser.add_argument(
        "--record-negatives",
        action="store_true",
        help="Save all detections to data/raw/negative/recorded/ for retraining",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=COOLDOWN_SECONDS,
        help=f"Minimum seconds between detections (default: {COOLDOWN_SECONDS})",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Device selection
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Cough Detector — Real-Time Inference")
    print("=" * 50)
    print(f"  Compute device: {device}")

    model, feature_params, recommended_threshold = load_model(device)

    threshold = args.threshold if args.threshold is not None else recommended_threshold
    print(f"  Detection threshold: {threshold:.2f}")
    print(f"  Cooldown:            {args.cooldown}s")
    print(f"  Sample rate:         {feature_params['sr']} Hz")
    if args.verbose:
        print("  Verbose mode:        ON (showing probability bars)")
    print()

    detector = CoughDetector(
        model=model,
        feature_params=feature_params,
        threshold=threshold,
        device=device,
        cooldown=args.cooldown,
        record_negatives=args.record_negatives,
        verbose=args.verbose,
    )

    detector.start(device_index=args.device)


if __name__ == "__main__":
    main()
