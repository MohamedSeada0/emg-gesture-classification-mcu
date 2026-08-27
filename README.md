# EMG Gesture Classification MCU

This repository contains the code used for the experiments presented in:

**EMG Gesture Classifiers Toward Deployment on Low-Resource Microcontrollers: A Compact CNN and Post-Training INT8 Quantization Study**

## Overview

The project implements:
- Compact CNN training for EMG gesture classification.
- Subject-wise train/validation/test splitting.
- FP32 PReLU and matched ReLU baselines.
- Post-training INT8 dynamic and static quantization.
- Evaluation of accuracy, model size, and peak activation memory.
- Generation of figures and analysis results reported in the paper.

## Dataset

The experiments use the **UCI EMG Data for Gestures** dataset.

The dataset is not included in this repository. Please download it from the UCI Machine Learning Repository and place it in the expected data directory.
https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures

## Main Experiments
The repository includes scripts for:
- FP32 model training.
- Repeated training with different random seeds.
- ReLU baseline training.
- Dynamic INT8 quantization.
- Static INT8 quantization.
- Classification analysis and figure generation.


The experiments use a subject-wise split:

- Training: 25 subjects
- Validation: 5 subjects
- Test: 6 subjects
