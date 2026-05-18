# MAVEN: Multimodal Assistive Visual-Audio Emotion Network

MAVEN is a web-based multimodal emotion recognition system that analyzes emotions using speech and facial expressions. The platform is designed as an assistive learning tool for neurodivergent users by combining real-time emotion detection with structured self-assessment and progress tracking.

The system supports emotion recognition from:

* Live microphone input
* Uploaded audio files
* Images
* Videos
* Live webcam streams

MAVEN integrates speech and visual inference pipelines through a browser-accessible interface with backend APIs and dashboard analytics.

---

## Features

* Multimodal emotion recognition using audio and visual inputs
* Real-time inference through browser-based interaction
* Speech Emotion Recognition (SER) using transformer-based embeddings
* Visual Emotion Recognition (VER) using CNN and temporal modeling
* Support for uploaded and live media streams
* Structured self-assessment module
* Dashboard for longitudinal progress tracking
* REST API-based backend architecture
* Modular deployment and extensible pipeline design

---

## System Architecture

MAVEN follows a layered architecture consisting of:

1. **Frontend Layer**

   * Built using HTML, CSS, JavaScript, and React-based components
   * Provides modules for audio/video upload, webcam input, and testing workflows

2. **Backend Layer**

   * Flask/FastAPI-based REST APIs
   * Handles preprocessing, validation, inference, logging, and response generation

3. **Machine Learning Layer**

   * Speech and visual emotion recognition models
   * Supports real-time inference

4. **Database Layer**

   * MySQL-based logging and analytics storage
   * Tracks predictions, timestamps, and user assessment data

---

# Emotion Categories

All datasets are harmonized into four unified emotion classes:

* Angry
* Happy
* Sad
* Neutral

---

# Speech Emotion Recognition Pipeline

The speech pipeline processes live or uploaded audio data using transformer-based contextual embeddings.

## Workflow

1. Audio resampling to 16 kHz
2. Noise reduction
3. Voice preprocessing
4. WavLM embedding extraction
5. Statistical pooling
6. MLP-based classification

## Speech Model

* Backbone: WavLM / wav2vec-style embeddings
* Pooling: Mean, Max, Standard Deviation
* Classifier: Multi-Layer Perceptron (MLP)
* Loss Function: Weighted Cross-Entropy
* Optimizer: Adam

## Datasets Used

* IEMOCAP
* RAVDESS
* SAVEE
* TESS
* CREMA-D

## Performance

* Cross-dataset accuracy: ~71–73%
* Macro F1 Score: ~0.68

The training strategy uses speaker-independent splits to reduce identity leakage and improve generalization.

---

# Visual Emotion Recognition Pipeline

The visual pipeline supports emotion recognition from images, videos, and live webcam streams.

## Workflow

1. Face detection using MTCNN
2. Face cropping and normalization
3. Spatial feature extraction using ResNet50
4. Temporal modeling using BiLSTM
5. Self-attention pooling
6. Softmax-based emotion classification

## Visual Model

* Face Detection: MTCNN
* Spatial Backbone: ResNet50
* Temporal Modeling: BiLSTM
* Training Strategy: Cross-domain generalization

## Datasets Used

* FER
* AffectNet
* IEMOCAP
* RAVDESS
* CREMA-D

## Performance

* Cross-corpus test accuracy: ~88.72% under controlled evaluation settings

---

# Self-Assessment Module

The platform includes a structured testing module for emotion recognition practice.

## Capabilities

* Displays pre-labeled samples
* Records user-selected emotions
* Computes real-time accuracy
* Updates dashboard analytics dynamically

## Dashboard Metrics

* Total sessions
* Overall accuracy
* Emotion-wise statistics
* Longitudinal performance trends

---

# Tech Stack

## Frontend

* HTML
* CSS
* JavaScript
* React
* TypeScript

## Backend

* Flask
* FastAPI

## Machine Learning

* PyTorch
* Hugging Face Transformers
* OpenCV
* librosa
* MTCNN

## Database

* MySQL

---

# Deployment Workflow
1. User uploads or records media
2. Backend validates input
3. Preprocessing pipeline is executed
4. Emotion inference is performed
5. Predictions are returned as JSON
6. Results are logged into the database
7. Dashboard updates in real time

---

# Project Goals

MAVEN focuses on:
* Real-time multimodal emotion recognition
* Cross-dataset generalization
* Accessible deployment on standard devices
* Structured assistive learning workflows
* Scalable and modular architecture design

---

# Current Limitations

* Reduced performance under low illumination or motion blur
* Partial confusion between acoustically similar emotions
* Limited to four emotion categories
* Real-world noisy environments may affect inference stability

---

