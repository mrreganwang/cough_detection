# Cough Detector

Real-time cough detection from a laptop microphone using a CNN trained on log-mel spectrograms.

## Dataset

Training data is drawn from four public sources:

**Cough (positive class)**
- [ESC-50](https://github.com/karolpiczak/ESC-50) — 40 cough clips from a curated environmental sound dataset
- [Cough Audio Dataset](https://www.kaggle.com/datasets/himanshu007121/cough-audio-dataset) — labelled cough/not-cough WAV recordings
- [COUGHVID](https://www.kaggle.com/datasets/nasrulhakim86/coughvid-wav) — crowdsourced cough recordings from hundreds of people on different devices; only clips with ≥80% cough confidence are used
- [COVID-19 Cough Audio Classification](https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification) — ~2800 cough recordings collected for COVID-19 research

**Non-cough (negative class)**
- ESC-50 — sneezing, clapping, breathing, laughing, dog barks, and ambient sounds; chosen specifically because they resemble coughs and make for hard negatives
- [MUSAN](https://www.openslr.org/17/) — 500 clips of real speech
- Synthetic — generated white noise, brown noise, tone bursts, and click transients
- User-recorded — any sounds captured via `--record-negatives` at runtime (e.g. blowing air, throat clearing) are automatically oversampled to ensure the model learns to reject them

In total the dataset contains roughly 15,000 cough spectrograms and 45,000 non-cough spectrograms after windowing. Data files are not included in this repository; run `python src/download_data.py` to fetch them.

## Setup

```bash
git clone https://github.com/mrreganwang/cough_detection.git
cd cough_detector
python -m venv venv
source venv/bin/activate
```

## Usage

```bash
# 1. Download training data (requires free Kaggle account)
python src/download_data.py

# 2. Prepare dataset, convert audio files to mel spectrogram
python src/prepare_dataset.py

# 3. Train
python src/train.py

# 4. Inference. Detect cough signals in real-time
python src/realtime_detect.py --verbose --threshold 0.45
```

Output:
```
Cough 2024-01-15 14:23:07.412
Cough 2024-01-15 14:23:07.891
```

## Kaggle Setup

Required for the main cough datasets. One-time setup:

1. Create a free account at kaggle.com
2. Go to Settings → API → Create New Token
3. Move the downloaded file: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

## Tuning

```bash
python src/realtime_detect.py --threshold 0.85   # fewer false positives
python src/realtime_detect.py --threshold 0.60   # catch more coughs
python src/realtime_detect.py --verbose          # show live probability
python src/realtime_detect.py --list-devices     # choose microphone
```

## Fixing False Positives

If a sound triggers false detections, record it as a negative example and retrain:

```bash
python src/record_samples.py --negative           # press Enter to record the sound
python src/prepare_dataset.py                     
python src/train.py
```

## Requirements

- Python 3.10+
- macOS (MPS acceleration) or Linux/Windows (CPU)
- ~4 GB disk space for datasets
- ~3 GB RAM for dataset preparation
