

## This project demonstrates an audio classification pipeline to identify whether an audio file contains a "cat" or a "dog" sound. The pipeline uses `librosa` for feature extraction and `scikit-learn` for training a Support Vector Classifier (SVC) model.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Features](#features)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction

The goal of this project is to classify audio files into two categories: "cat" or "dog." It involves:

1. Extracting features from audio files (e.g., MFCCs, spectral centroid, chroma, etc.).
2. Training a machine learning model to classify the audio based on the extracted features.
3. Evaluating the model on test data.

## Installation

To run this project, you'll need the following dependencies:

- Python 3.6+
- `librosa` for audio processing
- `scikit-learn` for model training and evaluation
- `numpy` for numerical operations
- `tqdm` for progress bars
- `matplotlib` for visualizations

You can install the required libraries using `pip`:

```bash
pip install librosa scikit-learn numpy tqdm matplotlib
Data
The dataset used in this project contains audio files categorized into two classes:

cat: Audio samples of cat sounds
dog: Audio samples of dog sounds
```
##You can download the dataset from Kaggle by following this link.
https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs

## The dataset is structured as follows:
```
cats_dogs/
    ├── train/
    │   ├── cat/
    │   │   └── cat_1.wav
    │   └── dog/
    │       └── dog_1.wav
    ├── test/
    │   ├── cat/
    │   │   └── cat_test_1.wav
    │   └── dog/
    │       └── dog_test_1.wav

```
## Features
The following audio features are extracted from each sample:

MFCC (Mel-Frequency Cepstral Coefficients): A feature that represents the short-term power spectrum of sound, commonly used in speech and audio processing.
Spectral Centroid: Measures the "center of mass" of the spectrum, often associated with the timbre of the sound.
Root Mean Square (RMS): Represents the energy of the audio signal.
Zero Crossing Rate: The rate at which the audio signal changes sign, useful for detecting percussive sounds.
Chroma: Represents the 12 pitch classes in music, capturing harmonic content.
Tempo: The speed of the audio in beats per minute (BPM).
These features are averaged over time and used as input to the machine learning model.

## Model Training
The features are used to train a Support Vector Classifier (SVC) model. The dataset is divided into a training set (used to train the model) and a test set (used to evaluate the model's performance). The model is trained using scikit-learn's SVC class.

## Prediction
Once the model is trained, you can use it to predict the class (cat or dog) of any audio file using the getPred() function. Here's how you can predict a sample file:

sample_path = "path_to_audio_file.wav"
getPred(sample_path)
The model will output either "cat" or "dog" based on the audio content.

## Evaluation
The model is evaluated on the test set, and performance metrics such as accuracy, precision, recall, and F1-score are computed using classification_report() from sklearn.metrics.

Sample output:

              precision    recall  f1-score   support

         cat       0.81      0.87      0.84        39
         dog       0.80      0.71      0.75        28

    accuracy                           0.81        67
   macro avg       0.80      0.79      0.80        67
weighted avg       0.81      0.81      0.80        67

## License
This project is licensed under the MIT License - see the LICENSE file for details.
