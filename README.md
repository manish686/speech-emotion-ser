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

data/
happy/
*.wav
sad/
*.wav
angry/
*.wav
neutral/
*.wav
...


Each subfolder name is treated as the emotion label.  
The notebook creates `train.csv`, `val.csv`, and `test.csv` with:

- `filepath`: path to the audio file  
- `label`: corresponding emotion class

Update the base `data` path in the notebook if your structure is different.

## How to Use

1. Open the notebook in Google Colab.  
2. Mount Google Drive or upload your dataset to Colab.  
3. Set the dataset path in the configuration cell.  
4. Run all cells in order:
   - Data indexing and split  
   - Zero-shot evaluation (pretrained model, no fine-tuning)  
   - Fine-tuning on train/val  
   - Final evaluation on test set

## Tasks Covered

The notebook supports the following tasks:

1. **Zero-shot model evaluation** using a pretrained speech emotion model (no training on your data).  
2. **Training / Fine-tuning** the acoustic model on the domain-specific dataset and evaluating on the test set.  
3. **Metric reporting**: Accuracy, Precision, Recall, F1-score for different approaches.  
4. Comparison and discussion of zero-shot vs fine-tuned vs custom architectures can be built using the logged metrics.

## Reproducibility

- Random seeds are set where appropriate (e.g., `numpy`, `torch`, `train_test_split`) to get stable splits and training behavior, as far as possible in Colab.
- For consistent results across runs, keep the same random seed and dataset organization.

## License

Add your preferred license information here (e.g., MIT, Apache-2.0), or your institute’s default.


