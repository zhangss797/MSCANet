# MSCANet: Multi-Scale Channel Attention Network for EEG-Based Parkinson's Disease Detection

This repository provides the PyTorch implementation of **MSCANet**, a lightweight multi-scale channel attention network for EEG-based Parkinson's disease (PD) classification.

MSCANet combines multi-scale temporal feature extraction with adaptive scale weighting to capture complementary EEG temporal patterns while maintaining a compact model architecture. The implementation supports both subject-level stratified K-fold cross-validation and leave-one-subject-out (LOSO) validation.

## Repository Structure

```text
MSCANet/
├── mscanet_model.py      # MSCANet architecture and MSCA block
├── train_mscanet.py      # Training, evaluation, and validation pipeline
└── README.md             # Project documentation
```

## Model Overview

The main components of MSCANet are:

- A temporal convolution module for initial EEG representation learning.
- A multi-scale channel attention (MSCA) block with parallel depthwise temporal convolutions using kernel sizes **7, 25, and 51**.
- Channel-wise adaptive scale weighting with Softmax normalization across the scale dimension.
- A residual connection for preserving the original latent representation.
- A lightweight convolutional feature refinement module followed by global pooling and classification.

The default dropout rate used in the released implementation is **0.3**.

## Data Preparation

The training script expects **preprocessed and segmented EEGLAB `.set` files**, with one file corresponding to one participant.

File names should begin with:

```text
Control*.set   # Healthy control participants
PD*.set        # Parkinson's disease participants
```

The preprocessing procedure used in the manuscript includes:

- 0.5–45 Hz band-pass filtering
- 60 Hz notch filtering
- Downsampling to 256 Hz
- Common average reference (CAR)
- ICA-based artifact removal with manual inspection
- Non-overlapping EEG segmentation

The primary experiments use **2-second non-overlapping segments**. The repository does not redistribute the original EEG recordings. Please obtain the dataset from its original source and complete preprocessing before running the training script.

To prevent data leakage, all cross-validation procedures are performed at the **subject level**, so EEG segments from the same participant are never included in both the training and test sets within a fold.

In addition, fold-wise normalization is performed using statistics estimated only from the training data and then applied to the corresponding test data.

## Requirements

The implementation requires Python 3 and the following main packages:

```text
torch
numpy
pandas
scikit-learn
mne
matplotlib
tqdm
thop
```

They can be installed using pip, for example:

```bash
pip install torch numpy pandas scikit-learn mne matplotlib tqdm thop
```

## Usage

### 1. Subject-level K-fold cross-validation

The default setting uses 10-fold subject-level cross-validation:

```bash
python train_mscanet.py --data-dir ./data --protocol kfold
```

The main default training settings are:

```text
Epochs:             60
Batch size:         32
Learning rate:      1e-5
Optimizer:          AdamW
Weight decay:       1e-2
Label smoothing:    0.1
Dropout:            0.3
Random seed:        42
Number of folds:    10
```

The number of folds can be changed using:

```bash
python train_mscanet.py --data-dir ./data --protocol kfold --n-splits 5
```

### 2. Leave-One-Subject-Out validation

LOSO validation can be performed using:

```bash
python train_mscanet.py --data-dir ./data --protocol loso
```

In LOSO validation, one participant is held out for testing in each iteration, while all remaining participants are used for model training.

### 3. Model complexity

To additionally report the number of parameters, FLOPs, and network-only inference latency:

```bash
python train_mscanet.py --data-dir ./data --protocol kfold --complexity
```

The reported inference latency refers only to the neural network forward pass and does not include EEG preprocessing or artifact-removal procedures.

## Evaluation

For K-fold cross-validation, the script reports both:

- **Segment-level performance**, where each EEG segment is treated as an individual prediction.
- **Subject-level performance**, where prediction probabilities from all EEG segments belonging to the same participant are averaged to obtain one final participant-level prediction using a threshold of 0.5.

For LOSO validation, subject-level predictions from all held-out participants are pooled to calculate the final performance metrics.

The reported metrics include:

- Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- Balanced Accuracy
- AUC, when applicable

## Output Files

By default, experimental outputs are saved to `./results/`.

Depending on the selected validation protocol, the script generates files such as:

```text
kfold_subject_splits.csv
kfold_subject_predictions.csv
loso_subject_splits.csv
loso_subject_predictions.csv
```

These files provide subject-level split information and predictions to facilitate reproducibility and independent verification of the evaluation protocol.

## Reproducibility

A fixed random seed of **42** is used by default. PyTorch deterministic settings are enabled where applicable. All subject-level train/test partitions are saved during execution so that the experimental splits can be inspected and reproduced.

## Notes

- This repository contains the model and evaluation code used for the MSCANet study.
- The raw EEG dataset is not included in this repository.
- Input data must already be preprocessed and segmented into EEGLAB epochs before running the scripts.
- External multicenter validation is not included in the current release and remains an important direction for future work.

## Citation

If you use this code in your research, please cite the corresponding MSCANet manuscript. Citation information will be updated after publication.
