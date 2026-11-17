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
| `environment/` | Server dependency snapshot and extracted core-package versions. |
| `requirements.txt` | Dependency ranges for installing the main-model workflow. |
| `verify_release.py` | CPU checks for data preparation, cache consistency, and saved-predictor replay. |
| `verification.json` | Recorded release checks and their scope. |
| `SHA256SUMS.txt` | SHA256 checksums of the release files. |
| `CHANGELOG.md` | Changes relative to the archived research code. |

## ⚙️ Python Environment

| Library | Version |
|---|---|
| Python | CPython 3.12 |
| PyTorch | 2.8.0+cu128 |
| NumPy | 2.3.2 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| RDKit | 2026.3.2 |
| Transformers | 5.9.0 |
| joblib | 1.5.3 |


## Dataset

The original dataset was downloaded from [TOXRIC](https://toxric.bioinforai.tech/download) on 21 November 2025 for the `mouse_intravenous_LD50` endpoint. The workflow reads the supplied pLD50 labels directly, without an additional negative-logarithmic transformation or dose-unit conversion.

Required CSV columns:

```text
Canonical SMILES
mouse_intravenous_LD50
```

Dataset split: 11,454 training, 1,432 validation, and 1,432 held-out test samples.

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

Use `--save-ml-models` to save the fitted regressors required by `predict.py`.

## Prediction

Provide a CSV containing the same SMILES column. Toxicity labels are not required.

```bash
python predict.py --run-dir results/main --input-csv data/new_molecules.csv --cache-dir cache/new --output-csv results/new_predictions.csv
```


## Resources

- [TOXRIC dataset download](https://toxric.bioinforai.tech/download)
- [ChemBERTa pretrained model](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
- [RDKit](https://www.rdkit.org/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
