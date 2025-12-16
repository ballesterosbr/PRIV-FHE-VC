# PRIV-FHE-VC: Privacy-Preserving Predictive Inference via FHE

This repository contains the implementation and experiments for the PRIV-FHE-VC project, exploring privacy-preserving machine learning inference using Fully Homomorphic Encryption (FHE) in healthcare contexts.

## Project Overview

The project establishes a secure framework for encrypted health data processing, enabling healthcare providers to perform ML inference on encrypted patient data without accessing sensitive information. The focus is on evaluating **FHE degradation** compared to standard (clear) ML inference across diverse medical data types.

## Repository Structure
```
├── code/
│   ├── tabular/       # FHE inference on structured health records
│   ├── images/        # CNN-based classification on medical images (MedMNIST)
│   ├── queries/       # Privacy-preserving genomic queries
│   └── time-series/   # Encrypted analysis of temporal medical data
├── csvs/              # Experimental results (benchmarks and metrics)
```

## Use Cases

- **Tabular Data**: Privacy-preserving predictions on structured clinical records
- **Medical Images**: Encrypted classification of X-rays, CT scans, and pathology images
- **Genomic Queries**: Secure variant matching without revealing coordinates (CKKS-based)
- **Time-Series**: Encrypted analysis of continuous monitoring data (ECG, vitals)

## Technologies
- **Concrete ML**: FHE-compatible ML model compilation ([GitHub](https://github.com/zama-ai/concrete-ml))
- **OpenFHE**: CKKS scheme for encrypted comparisons ([GitHub](https://github.com/openfheorg/openfhe-development))
- **DoublePIR**: Private information retrieval library ([GitHub](https://github.com/ahenzinger/simplepir))


## Getting Started

Each subdirectory contains its own README with specific setup instructions. All implementations use `uv` for dependency management.

---

**Acknowledgments**: This project has received funding from the European Union's Horizon Europe research and innovation programme under Grant Agreement No. 101095717 — SECURED.