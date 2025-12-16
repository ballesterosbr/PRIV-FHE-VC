# FHE-Enabled CNN for Medical Image Classification

Privacy-preserving medical image classification using Concrete ML's fully homomorphic encryption (FHE).

## What it does

Trains convolutional neural networks on MedMNIST datasets and compiles them for encrypted inference using FHE. Multiple quantization strategies (PTQ/QAT) and image formats (RGB/Grayscale) are evaluated to assess accuracy-performance trade-offs.

**Use case**: Enable secure medical image classification where neither the model owner nor the data owner sees each other's sensitive information during inference.

## Quick Start

### Requirements
```bash
uv sync
```

### Run experiments

Each script trains a CNN on multiple MedMNIST datasets, then performs FHE inference with different quantization configurations:
```bash
# Grayscale 128x128 with Quantization-Aware Training (QAT)
uv run multi_dataset_fhe_cnn_GREY_128x128_QAT.py

# Grayscale 64x64 with Post-Training Quantization (PTQ)
uv run multi_dataset_fhe_cnn_GREY_64x64_PTQ.py

# RGB 128x128 with QAT
uv run multi_dataset_fhe_cnn_RGB_128x128_QAT.py

# RGB 64x64 with PTQ
uv run multi_dataset_fhe_cnn_RGB_64x64_PTQ.py
```

**Note**: Scripts automatically resume from previous progress. Trained models are cached in `models/` directory except for 64x64 PTQ configuration.

## Script Variants

### Quantization Strategy

- **QAT (Quantization-Aware Training)**: Uses Brevitas to train with quantization constraints from the start. Better accuracy for FHE but longer training.
- **PTQ (Post-Training Quantization)**: Quantizes after standard training. Faster but may have slightly lower FHE accuracy.

### Image Format

- **GREY**: Single-channel grayscale images (most MedMNIST datasets)
- **RGB**: Three-channel color images (PathMNIST, DermaMNIST, etc.)

### Resolution

- **128x128**: Higher resolution, more parameters, better accuracy
- **64x64**: Lower resolution, faster inference, lighter models

## Medical Datasets

Scripts process multiple MedMNIST datasets automatically:
- **PneumoniaMNIST**: Chest X-ray pneumonia detection
- **BreastMNIST**: Breast ultrasound tumor classification
- **OCTMNIST**: Retinal OCT scans
- **TissueMNIST**: Kidney tissue pathology
- **OrganAMNIST/CMNIST/SMNIST**: Abdominal CT organ segmentation
- **PathMNIST**: Colon pathology (9 tissue types)
- **DermaMNIST**: Dermatoscopic lesion classification
- **RetinaMNIST**: Fundus photography for retinopathy
- **BloodMNIST**: Blood cell microscopy

## Study Design Note

The CNN architectures are intentionally **not optimized** for individual datasets. The goal is to analyze **FHE degradation** compared to standard (clear) ML inference across diverse medical imaging tasks with a fixed architecture.

## Output

- `results_<FORMAT>_<SIZE>_<METHOD>_<DATASET>.csv` - FHE inference timings and predictions
- `results_<FORMAT>_<SIZE>_<METHOD>_clear.csv` - Standard (non-encrypted) baseline accuracy
- `models/trained_models_*/` - Cached PyTorch models for reuse

## FHE Configuration Search

Scripts automatically test multiple FHE parameter combinations:
- **Bit widths**: 4, 5, 6, 7, 8 bits
- **Rounding thresholds**: 5, 6 bits
- **P-error**: 0.008 (fixed)

Each configuration produces different accuracy/speed trade-offs in encrypted inference.

## Some Additional Features

- **Early stopping**: Prevents overfitting during training
- **Resume capability**: Automatically continues incomplete experiments
- **Multi-dataset**: Batch processing of all compatible MedMNIST datasets
- **Memory management**: Clears GPU/CPU between experiments
