# ASR Signal Quality Analysis

Comparing Whisper and Wav2Vec2 on CommonVoice, correlating errors with audio signal properties (RMS, silence ratio, SNR, spectral features).

## Structure
- `scripts/run_inference.py` -> runs both models and saves predictions + WER
- `scripts/analyze.py`  -> signal feature extraction + correlation analysis
