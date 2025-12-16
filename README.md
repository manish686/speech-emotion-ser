# Audio Emotion Recognition

This repository contains a Colab notebook for building and evaluating a speech emotion recognition (SER) system using a domain-specific audio dataset.

## Contents

- `audio_emotion_recognition.ipynb`  
  End-to-end pipeline for:
  - Loading and labeling audio files by emotion category  
  - Splitting data into train / validation / test sets  
  - Zero-shot evaluation using a pretrained emotion model  
  - Fine-tuning the pretrained model on the given dataset  
  - Computing metrics (Accuracy, Precision, Recall, F1-score)

## Requirements

The notebook is designed to run in Google Colab. Main libraries:

- Python 3.x  
- `torch`, `torchaudio`  
- `transformers`  
- `librosa`, `soundfile`  
- `scikit-learn`, `pandas`, `numpy`

These are installed directly inside the notebook cells via `pip`.

## Dataset

The notebook assumes a folder structure similar to:

