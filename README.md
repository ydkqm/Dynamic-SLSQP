# Dynamic-SLSQP

An ensemble framework for predicting mouse intravenous acute toxicity on the pLD50 scale. The model combines frozen ChemBERTa representations, physicochemical descriptors and molecular fingerprints, and 3D conformer descriptors. Its predictions integrate multi-seed neural networks and seven traditional regression methods.

## 📁 Directory Structure

| Folder/File | Description |
|---|---|
| `data/` | Location for user-supplied raw data and training, validation, and test splits. Dataset files are not included. |
| `prepare_data.py` | Data cleaning, structural redundancy filtering, and fixed train/validation/test splitting. Combines the original `ry.py` and `split_data.py` workflows. |
| `run.py` | Main training entry point, including input validation, feature-cache management, and run metadata. |
| `main_model.py` | Feature extraction, multimodal neural network, five-fold cross-fitting, traditional regressors, and SLSQP weight optimization. |
| `predict.py` | Prediction for new molecules using the saved five-fold ensemble. |
| `runtime_artifacts.py` | Saving preprocessors, neural checkpoints, predictions, and optional fitted regressors. |
| `reference_results/` | Archived performance tables, ensemble weights, and split-file hashes. |
| `environment/` | Server dependency snapshot and extracted core-package versions. |
| `requirements.txt` | Dependency ranges for installing the main-model workflow. |
| `verify_release.py` | CPU checks for data preparation, cache consistency, and saved-predictor replay. |
| `verification.json` | Recorded release checks and their scope. |
| `SHA256SUMS.txt` | SHA256 checksums of the release files. |
| `CHANGELOG.md` | Changes relative to the archived research code. |

## ⚙️ Python Environment

| Library | Version |
|---|---|
| Python | CPython 3.12* |
| PyTorch | 2.8.0+cu128 |
| NumPy | 2.3.2 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| RDKit | 2026.3.2 |
| Transformers | 5.9.0 |
| joblib | 1.5.3 |
| tqdm | Numeric version not recorded |

These versions were extracted from the supplied server environment export. *Python 3.12 is identified from its `cp312` wheel tag; the patch version was not recorded. The export is an environment record, not a verified portable installation lockfile. The `cu128` suffix identifies the PyTorch CUDA build.

### Installation

```bash
git clone https://github.com/ydkqm/Dynamic-SLSQP.git
cd Dynamic-SLSQP
python -m pip install -r requirements.txt
python -m pip check
python verify_release.py
```

Use a separate Python environment and install the PyTorch build appropriate for the target hardware. The full server snapshot contains machine-specific installation paths; use the root `requirements.txt` for installation. Actual package versions are recorded in each training run's `run_metadata.json`.

## Dataset

The original dataset was downloaded from [TOXRIC](https://toxric.bioinforai.tech/download) on 21 November 2025 for the `mouse_intravenous_LD50` endpoint. The workflow reads the supplied pLD50 labels directly, without an additional negative-logarithmic transformation or dose-unit conversion.

Required CSV columns:

```text
Canonical SMILES
mouse_intravenous_LD50
```

Data preparation removes missing SMILES, exact duplicate SMILES strings, and invalid molecular structures. Greedy redundancy filtering uses radius-2, 2048-bit Morgan fingerprints and retains a molecule only when its maximum Tanimoto similarity to previously retained molecules is below 0.80. Label cleaning follows this step. Two seed-42 random splits produce an 80%/10%/10% partition.

The archived study contains 11,454 training, 1,432 validation, and 1,432 held-out test samples after preparation. `preparation_audit.json` records processing counts and file hashes. Source order and software versions affect preprocessing reproducibility.

## Model Architecture

| Component | Implementation |
|---|---|
| ChemBERTa branch | Frozen `DeepChem/ChemBERTa-77M-MTR`; masked mean pooling produces 384 features, followed by a 384→256 encoder. |
| Descriptor/fingerprint branch | 10 physicochemical descriptors, 167 MACCS keys, and 1,024 Morgan fingerprint bits; 1,201→256 linear encoding with batch normalization and GELU. |
| 3D branch | 10 geometric descriptors, 12 USR descriptors, and one conformer-validity flag; 23→128→64 MLP encoder. |
| Feature fusion | Concatenation of 256+256+64 features, followed by a Linear–Sigmoid Fusion Gate that scales the 576 channels. |
| Neural regression | 576→256→128 MLP backbone and a 128→1 linear regression head. |
| Traditional regression | RidgeCV, BayesianRidge, ElasticNetCV, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, and KNeighborsRegressor. |
| Final ensemble | Eight candidates: `DL_mean` and seven regression outputs, combined using nonnegative SLSQP weights that sum to one. |

Within each outer fold, five networks use seeds 42, 1024, 2026, 8888, and 9999. Their 128-dimensional hidden representations are concatenated into 640 features for each traditional regression pipeline, which fits its own StandardScaler on that fold's training features.

An internal stopping subset selects training epochs. The networks are then reinitialized and trained on the full outer-training subset. Five-fold cross-fitting produces the training out-of-fold (OOF) predictions used to learn one global set of SLSQP weights. All eight candidates enter optimization; no error-threshold candidate filtering is applied.

Neural networks and regressors share a fold-specific standardized Yeo–Johnson target transformation. Each prediction is inverse-transformed to pLD50 before averaging. For validation and test data, predictions of the same candidate are averaged across five folds and combined with the fixed ensemble weights. The final predictor uses 25 neural networks and 35 fitted regressors.

## Training

### Start from a raw CSV

Place the raw dataset at `data/raw_mouse.csv` and run:

```bash
python run.py --raw-csv data/raw_mouse.csv --data-dir data/splits --cache-dir cache/main --output-dir results/main --save-ml-models
```

### Use existing splits

Place `train.csv`, `val.csv`, and `test.csv` in `data/fixed/`:

```bash
python run.py --data-dir data/fixed --cache-dir cache/main --output-dir results/main --save-ml-models
```

The `--save-ml-models` option is required for subsequent complete-ensemble prediction. Saved regressors can require substantial disk space. Use a new output directory for a new run.

Feature caches are reused after provenance checks. For verified historical `v40_etkdg_3dmlp_*.pt` caches without the current provenance records, explicitly add `--trust-legacy-cache` after checking sample order and extraction settings. Use `--chemberta-model /path/to/local/model` to select a local pretrained model; add `--offline` for offline execution.

## Prediction

Provide a CSV containing the same SMILES column. Toxicity labels are not required.

```bash
python predict.py --run-dir results/main --input-csv data/new_molecules.csv --cache-dir cache/new --output-csv results/new_predictions.csv
```

The output contains the eight cross-fold candidate predictions and the final `pred_pLD50`. These are continuous regression values, not probabilities. Keep the complete saved training directory, including preprocessors and traditional regressors, for inference.

## Reference Results

| Model | Validation RMSE | Validation R² | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| DL-E2E | 0.446210 | 0.628184 | 0.453925 | 0.609872 |
| Static-Ensemble | 0.448387 | 0.624547 | 0.454010 | 0.609726 |
| Dynamic-SLSQP | 0.445187 | 0.629886 | 0.452606 | 0.612135 |

These tables reproduce the archived final-model run; they are not results from a new training run of this release. DL-E2E averages direct neural predictions, Static-Ensemble averages the seven traditional candidates, and Dynamic-SLSQP combines all eight candidates. The test split comes from the same source dataset and is not an external validation cohort. The OOF matrix is used to fit ensemble weights, so the fitted ensemble's OOF error is not an independent test estimate.

## Resources

- [TOXRIC dataset download](https://toxric.bioinforai.tech/download)
- [ChemBERTa pretrained model](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
- [RDKit](https://www.rdkit.org/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
