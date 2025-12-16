# FHE Tree-Based Models for Tabular Medical Data

Privacy-preserving classification on structured health records using Concrete ML's encrypted tree-based models.

## What it does

Benchmarks Decision Trees and Random Forests under FHE encryption to evaluate accuracy-performance trade-offs on tabular clinical data. Compares standard (clear) scikit-learn models against their FHE-encrypted counterparts across multiple hyperparameter configurations.

**Study goal**: Quantify FHE-induced degradation in accuracy and inference time compared to unencrypted baselines.

## Datasets

### Arrhythmia Classification
- **Source**: [UCI Arrhythmia dataset](https://archive.ics.uci.edu/dataset/5/arrhythmia)
- **Task**: Binary classification (Normal vs. Risk)
- **Features**: Clinical measurements with missing value imputation
- **Script**: `gen_models.py`

### Synthetic Data
- **Source**: scikit-learn `make_classification`
- **Purpose**: Systematic hyperparameter exploration
- **Script**: `gen_models_make.py` (uses multiprocessing for parallel experiments)

## Models Evaluated

- **DecisionTreeClassifier** (sklearn baseline)
- **RandomForestClassifier** (sklearn baseline)
- **FHEDecisionTreeClassifier** (Concrete ML)
- **FHERandomForestClassifier** (Concrete ML)

## Hyperparameter Grid

- **max_depth**: 3 to 14 (tree complexity)
- **n_bits**: 2 to 11 (quantization precision for FHE models)
- **n_estimators**: 10 (number of trees in Random Forest models)
- **n_features**: 5 to 19 (synthetic data only)

Each configuration produces metrics: training time, compilation time (FHE only), prediction time, accuracy, and F1 score.

## Requirements
```bash
pip install concrete-ml scikit-learn pandas numpy
```

## Usage
```bash
# Arrhythmia dataset benchmark
python gen_models.py

# Synthetic data benchmark (parallel execution)
python gen_models_make.py
```

## Output

Results are saved as CSV files:
- `stats_DecisionTreeClassifier.csv`
- `stats_RandomForestClassifier.csv`
- `stats_FHEDecisionTreeClassifier.csv`
- `stats_FHERandomForestClassifier.csv`

Columns include: model name, hyperparameters, timing metrics, accuracy, and F1 score.

## Note

Models are **not optimized** for individual datasets. The architecture remains fixed to isolate FHE-specific performance degradation from model engineering effects.