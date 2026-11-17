from runtime_artifacts import (artifact_context, record_epoch, save_preprocessors,
    save_inner_transformer, save_neural_model, save_ml_models, predict_ml_pld50,
    record_slsqp, save_candidates, export_predictions, save_classical_baseline)

import gc
import json
import logging
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, MACCSkeys, rdMolDescriptors
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ImportError("Please install transformers before running this script.") from exc


# =============================================================================
# 1. Configuration
# =============================================================================
# Network/offline behavior is controlled by the launcher or user environment.
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Config:
    DATA_DIR = "."
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    VAL_CSV = os.path.join(DATA_DIR, "val.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    LABEL_COL = "mouse_intravenous_LD50"
    SMILES_COL = "Canonical SMILES"

    CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MTR"
    CHEMBERTA_DIM = 384
    PHYS_DIM = 10
    MACCS_DIM = 167
    FP_DIM = 1024
    TABULAR_DIM = PHYS_DIM + MACCS_DIM + FP_DIM  # 1201

    # 10 scalar 3D descriptors + 12 USR descriptors + one conformer-valid flag.
    DESC3D_DIM = 23
    CHEM_HIDDEN_DIM = 256
    TAB_HIDDEN_DIM = 256
    SPATIAL_HIDDEN_DIM = 64
    DEEP_FEATURE_DIM = 128

    NUM_CONFORMERS = 10
    CONFORMER_SEED = 42
    CONFORMER_MAX_ITERATIONS = 1000
    CONFORMER_PRUNE_RMS = 0.5

    SEEDS = [42, 1024, 2026, 8888, 9999]
    KFOLD_SPLITS = 5
    INNER_STOP_FRACTION = 0.10
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2

    SLSQP_REG = 0.015

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = os.path.join(DATA_DIR, "dynamic_3dmlp_linear_tab_yj_results")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    CACHE_VERSION = "v40_etkdg_3dmlp"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_files():
    for path in (Config.TRAIN_CSV, Config.VAL_CSV, Config.TEST_CSV):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing input file: {os.path.abspath(path)}")


# =============================================================================
# 2. Offline feature extraction
# =============================================================================
PHYS_FUNCTIONS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.NumAromaticRings,
    Descriptors.FractionCSP3,
    Descriptors.BertzCT,
    Descriptors.NumHeteroatoms,
]


def finite_array(values, length):
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(array) != length or not np.all(np.isfinite(array)):
        return np.zeros(length, dtype=np.float32)
    return array


def conformer_energy(mol, conf_id, use_mmff):
    try:
        if use_mmff:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            field = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(conf_id))
        else:
            field = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
        return float(field.CalcEnergy()) if field is not None else np.inf
    except Exception:
        return np.inf


def generate_lowest_energy_conformer(mol):
    """ETKDG multi-conformer generation followed by MMFF/UFF optimization."""
    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = Config.CONFORMER_SEED
    params.maxIterations = Config.CONFORMER_MAX_ITERATIONS
    params.pruneRmsThresh = Config.CONFORMER_PRUNE_RMS
    params.useRandomCoords = True

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=Config.NUM_CONFORMERS, params=params))
    if not conf_ids:
        return None

    use_mmff = bool(AllChem.MMFFHasAllMoleculeParams(mol_h))
    try:
        if use_mmff:
            AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=500, mmffVariant="MMFF94s")
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=500)
        else:
            return None
    except Exception:
        return None

    energies = [conformer_energy(mol_h, cid, use_mmff) for cid in conf_ids]
    if not np.any(np.isfinite(energies)):
        return None
    best_conf = int(conf_ids[int(np.nanargmin(energies))])

    # Keep only the selected conformer so all descriptor functions use one geometry.
    selected = Chem.Mol(mol_h)
    best = Chem.Conformer(selected.GetConformer(best_conf))
    selected.RemoveAllConformers()
    selected.AddConformer(best, assignId=True)
    return selected


def calculate_3d_descriptors(mol):
    """Return 22 conformer descriptors plus a validity indicator."""
    conformer_mol = generate_lowest_energy_conformer(mol)
    if conformer_mol is None:
        return np.zeros(Config.DESC3D_DIM, dtype=np.float32), False

    try:
        scalar = [
            Descriptors3D.Asphericity(conformer_mol, confId=0),
            Descriptors3D.Eccentricity(conformer_mol, confId=0),
            Descriptors3D.InertialShapeFactor(conformer_mol, confId=0),
            Descriptors3D.NPR1(conformer_mol, confId=0),
            Descriptors3D.NPR2(conformer_mol, confId=0),
            Descriptors3D.PMI1(conformer_mol, confId=0),
            Descriptors3D.PMI2(conformer_mol, confId=0),
            Descriptors3D.PMI3(conformer_mol, confId=0),
            Descriptors3D.RadiusOfGyration(conformer_mol, confId=0),
            Descriptors3D.SpherocityIndex(conformer_mol, confId=0),
        ]
        usr = list(rdMolDescriptors.GetUSR(conformer_mol, confId=0))
        descriptor = finite_array(scalar + usr, 22)
        if not np.any(descriptor) or not np.all(np.isfinite(descriptor)):
            return np.zeros(Config.DESC3D_DIM, dtype=np.float32), False
        return np.concatenate([descriptor, np.ones(1, dtype=np.float32)]), True
    except Exception:
        return np.zeros(Config.DESC3D_DIM, dtype=np.float32), False


def get_offline_features(smiles_list, split_name):
    cache_path = os.path.join(Config.DATA_DIR, f"{Config.CACHE_VERSION}_{split_name}.pt")
    if os.path.isfile(cache_path):
        logging.info("Loading cached features: %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    logging.info("Extracting frozen ChemBERTa features for %s...", split_name)
    tokenizer = AutoTokenizer.from_pretrained(Config.CHEMBERTA_MODEL)
    bert = AutoModel.from_pretrained(Config.CHEMBERTA_MODEL).to(Config.DEVICE)
    bert.eval()
    chem_features = []
    for start in tqdm(range(0, len(smiles_list), Config.BATCH_SIZE), desc=f"ChemBERTa-{split_name}"):
        batch = [str(value) for value in smiles_list[start : start + Config.BATCH_SIZE]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(Config.DEVICE)
        with torch.no_grad():
            hidden = bert(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
            pooled = torch.sum(hidden * mask, dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        chem_features.append(pooled.cpu().numpy())
    chem_features = np.vstack(chem_features).astype(np.float32)

    del bert, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    phys_features, maccs_features, fp_features, desc3d_features = [], [], [], []
    invalid_smiles = 0
    conformer_failures = 0
    logging.info("Extracting RDKit 2D/fingerprint/3D descriptors for %s...", split_name)
    for smiles in tqdm(smiles_list, desc=f"RDKit-{split_name}"):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid_smiles += 1
            phys_features.append(np.zeros(Config.PHYS_DIM, dtype=np.float32))
            maccs_features.append(np.zeros(Config.MACCS_DIM, dtype=np.float32))
            fp_features.append(np.zeros(Config.FP_DIM, dtype=np.float32))
            desc3d_features.append(np.zeros(Config.DESC3D_DIM, dtype=np.float32))
            continue

        try:
            phys = finite_array([function(mol) for function in PHYS_FUNCTIONS], Config.PHYS_DIM)
        except Exception:
            phys = np.zeros(Config.PHYS_DIM, dtype=np.float32)
        try:
            maccs = finite_array(list(MACCSkeys.GenMACCSKeys(mol)), Config.MACCS_DIM)
        except Exception:
            maccs = np.zeros(Config.MACCS_DIM, dtype=np.float32)
        try:
            fp = finite_array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=Config.FP_DIM)), Config.FP_DIM)
        except Exception:
            fp = np.zeros(Config.FP_DIM, dtype=np.float32)

        desc3d, valid = calculate_3d_descriptors(mol)
        if not valid:
            conformer_failures += 1
        phys_features.append(phys)
        maccs_features.append(maccs)
        fp_features.append(fp)
        desc3d_features.append(desc3d)

    result = (
        chem_features,
        np.asarray(phys_features, dtype=np.float32),
        np.asarray(maccs_features, dtype=np.float32),
        np.asarray(fp_features, dtype=np.float32),
        np.asarray(desc3d_features, dtype=np.float32),
    )
    metadata = {
        "split": split_name,
        "samples": int(len(smiles_list)),
        "invalid_smiles": int(invalid_smiles),
        "conformer_failures": int(conformer_failures),
        "conformer_success_rate": float((len(smiles_list) - invalid_smiles - conformer_failures) / max(len(smiles_list), 1)),
        "num_conformers_requested": Config.NUM_CONFORMERS,
        "descriptor_dimension": Config.DESC3D_DIM,
    }
    torch.save(result, cache_path)
    with open(cache_path + ".json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    logging.info("%s 3D metadata: %s", split_name, metadata)
    return result


# =============================================================================
# 3. Deep model: ChemBERTa + linear table encoder + 3D descriptor MLP
# =============================================================================
class Descriptor3DMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, descriptors):
        return self.network(descriptors)


class MSformer3DMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chem_proj = nn.Sequential(
            nn.Linear(config.CHEMBERTA_DIM, config.CHEM_HIDDEN_DIM),
            nn.BatchNorm1d(config.CHEM_HIDDEN_DIM),
            nn.GELU(),
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(config.TABULAR_DIM, config.TAB_HIDDEN_DIM),
            nn.BatchNorm1d(config.TAB_HIDDEN_DIM),
            nn.GELU(),
        )
        self.spatial_mlp = Descriptor3DMLP(config.DESC3D_DIM, config.SPATIAL_HIDDEN_DIM)

        fused_dim = config.CHEM_HIDDEN_DIM + config.TAB_HIDDEN_DIM + config.SPATIAL_HIDDEN_DIM
        self.fusion_gate = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
        self.backbone = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, config.DEEP_FEATURE_DIM),
            nn.BatchNorm1d(config.DEEP_FEATURE_DIM),
            nn.GELU(),
        )
        self.regressor = nn.Linear(config.DEEP_FEATURE_DIM, 1)

    def extract_deep_features(self, chem, phys, maccs, fp, desc3d):
        chem_hidden = self.chem_proj(chem)
        tab_hidden = self.tab_proj(torch.cat([phys, maccs, fp], dim=-1))
        spatial_hidden = self.spatial_mlp(desc3d)
        fused = torch.cat([chem_hidden, tab_hidden, spatial_hidden], dim=-1)
        return self.backbone(fused * self.fusion_gate(fused))

    def forward(self, chem, phys, maccs, fp, desc3d):
        return self.regressor(self.extract_deep_features(chem, phys, maccs, fp, desc3d)).squeeze(-1)


class StableLogCoshLoss(nn.Module):
    def forward(self, prediction, target):
        error = prediction - target
        return torch.mean(error + F.softplus(-2.0 * error) - np.log(2.0))


class ToxicityDataset(Dataset):
    def __init__(self, chem, phys, maccs, fp, desc3d, y=None):
        self.chem = chem
        self.phys = phys
        self.maccs = maccs
        self.fp = fp
        self.desc3d = desc3d
        self.y = y

    def __len__(self):
        return len(self.chem)

    def __getitem__(self, index):
        item = {
            "chem": torch.tensor(self.chem[index], dtype=torch.float32),
            "phys": torch.tensor(self.phys[index], dtype=torch.float32),
            "maccs": torch.tensor(self.maccs[index], dtype=torch.float32),
            "fp": torch.tensor(self.fp[index], dtype=torch.float32),
            "desc3d": torch.tensor(self.desc3d[index], dtype=torch.float32),
        }
        if self.y is not None:
            item["y"] = torch.tensor(self.y[index], dtype=torch.float32)
        return item


def move_inputs(batch):
    return (
        batch["chem"].to(Config.DEVICE),
        batch["phys"].to(Config.DEVICE),
        batch["maccs"].to(Config.DEVICE),
        batch["fp"].to(Config.DEVICE),
        batch["desc3d"].to(Config.DEVICE),
    )


def make_loader(data, y=None, shuffle=False, drop_last=False):
    return DataLoader(
        ToxicityDataset(*data, y=y),
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def weighted_validation_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["y"].to(Config.DEVICE)
            prediction = model(*move_inputs(batch))
            loss = criterion(prediction, target)
            batch_count = len(target)
            total_loss += float(loss.item()) * batch_count
            total_count += batch_count
    return total_loss / max(total_count, 1)


def train_for_epochs(model, loader, epochs):
    criterion = StableLogCoshLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*move_inputs(batch))
            loss = criterion(prediction, batch["y"].to(Config.DEVICE))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite deep-training loss encountered.")
            loss.backward()
            optimizer.step()


def select_epoch_without_outer_holdout(inner_train_data, inner_train_y, inner_stop_data, inner_stop_y, seed):
    """The outer OOF split is never used for epoch selection."""
    set_seed(seed)
    model = MSformer3DMLP(Config).to(Config.DEVICE)
    criterion = StableLogCoshLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    train_loader = make_loader(inner_train_data, inner_train_y, shuffle=True, drop_last=True)
    stop_loader = make_loader(inner_stop_data, inner_stop_y, shuffle=False)

    best_epoch = 1
    best_loss = np.inf
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*move_inputs(batch))
            loss = criterion(prediction, batch["y"].to(Config.DEVICE))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite model-selection loss encountered.")
            loss.backward()
            optimizer.step()
        stop_loss = weighted_validation_loss(model, stop_loader, criterion)
        record_epoch(epoch, stop_loss, seed)
        if stop_loss < best_loss:
            best_loss = stop_loss
            best_epoch = epoch

    del model, optimizer, train_loader, stop_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch, best_loss


def extract_features_predictions(loader, models, label_transformer):
    features, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = move_inputs(batch)
            seed_features = [model.extract_deep_features(*inputs).cpu().numpy() for model in models]
            seed_scaled_predictions = [model(*inputs).cpu().numpy() for model in models]
            seed_original_predictions = [
                label_transformer.inverse_transform(values.reshape(-1, 1)).ravel()
                for values in seed_scaled_predictions
            ]
            features.append(np.concatenate(seed_features, axis=1))
            predictions.append(np.column_stack(seed_original_predictions))
    return np.vstack(features), np.vstack(predictions)


def transform_data(raw_data, indices, phys_scaler, desc3d_scaler):
    chem, phys, maccs, fp, desc3d = raw_data
    transformed_desc = desc3d.copy()
    # Standardize only the 22 continuous descriptors; preserve the final validity flag.
    transformed_desc[:, :22] = desc3d_scaler.transform(desc3d[:, :22])
    # A failed conformer is represented by a zero hidden input plus an explicit
    # validity flag, rather than by standardized artificial zero descriptors.
    transformed_desc[desc3d[:, -1] < 0.5, :22] = 0.0
    return (
        chem[indices],
        phys_scaler.transform(phys[indices]),
        maccs[indices],
        fp[indices],
        transformed_desc[indices],
    )


def fit_fold_preprocessors(raw_data, indices, y_original):
    phys_scaler = StandardScaler().fit(raw_data[1][indices])
    valid = raw_data[4][indices, -1] > 0.5
    descriptor_fit_indices = indices[valid]
    if len(descriptor_fit_indices) == 0:
        raise RuntimeError("No valid 3D conformer is available in the current training subset.")
    desc_scaler = StandardScaler().fit(raw_data[4][descriptor_fit_indices, :22])
    label_transformer = PowerTransformer(method="yeo-johnson", standardize=True).fit(y_original[indices].reshape(-1, 1))
    return phys_scaler, desc_scaler, label_transformer


# =============================================================================
# 4. Consistent fold-local ML pipelines and original-scale OOF optimization
# =============================================================================
MODEL_NAMES = ["DL_mean", "Ridge", "Bayes", "Elastic", "GBR", "RF", "ET", "KNN"]


def make_ml_pipelines():
    """Identical pipelines are used for repeated meta-OOF and final fitting."""
    estimators = [
        RidgeCV(alphas=np.logspace(-3, 5, 100)),
        BayesianRidge(max_iter=500),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], max_iter=5000, cv=5, n_jobs=-1),
        HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.015, l2_regularization=5.0, max_depth=6, random_state=42),
        RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, max_features=0.15, n_jobs=-1, random_state=42),
        ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, max_features="log2", n_jobs=-1, random_state=42),
        KNeighborsRegressor(n_neighbors=12, weights="distance"),
    ]
    return [Pipeline([("scale", StandardScaler()), ("model", estimator)]) for estimator in estimators]


def predict_ml_pld50(model, features, label_transformer):
    """Restore one fold-local ML prediction to pLD50 before aggregation."""
    transformed = np.asarray(model.predict(features), dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("ML model produced non-finite transformed predictions.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        restored = label_transformer.inverse_transform(transformed).ravel()
    if not np.isfinite(restored).all():
        raise FloatingPointError(
            "ML Yeo-Johnson inverse transform produced non-finite pLD50 values. "
            "Predictions may lie outside the invertible range; no clipping was applied."
        )
    return restored


def optimize_weights(meta_oof, y_train):
    candidate_mse = np.mean((meta_oof - y_train[:, None]) ** 2, axis=0)
    # All eight candidates participate; OOF errors are reported, not used for filtering.
    kept = np.arange(len(MODEL_NAMES), dtype=int)

    prior = np.full(len(kept), 0.5 / max(len(kept) - 1, 1), dtype=np.float64)
    prior[np.where(kept == 0)[0][0]] = 0.5
    if len(kept) == 1:
        prior[0] = 1.0

    def objective(weights):
        mse = mean_squared_error(y_train, meta_oof[:, kept] @ weights)
        penalty = Config.SLSQP_REG * np.var(y_train) * np.sum((weights - prior) ** 2)
        record_slsqp(weights, mse, penalty)
        return mse + penalty

    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(kept),
        constraints={"type": "eq", "fun": lambda values: values.sum() - 1.0},
    )
    selected = result.x if result.success else prior
    selected = selected / selected.sum()
    weights = np.zeros(len(MODEL_NAMES), dtype=np.float64)
    weights[kept] = selected

    logging.info("Original-scale outer-OOF candidate MSE: %s", {name: round(float(value), 6) for name, value in zip(MODEL_NAMES, candidate_mse)})
    logging.info("Ensemble candidates (no error-threshold filtering): %s", [MODEL_NAMES[index] for index in kept])
    logging.info("SLSQP success: %s; message: %s", result.success, result.message)
    logging.info("Optimal weights:")
    for name, weight in zip(MODEL_NAMES, weights):
        logging.info("   %-8s: %.4f", name, weight)
    return weights, candidate_mse, kept, result


def metrics(y_true, y_pred, prefix):
    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix}_MSE": float(mse),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(np.sqrt(mse)),
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
    }


def log_metrics(title, y_val, pred_val, y_test, pred_test):
    val = metrics(y_val, pred_val, "Val")
    test = metrics(y_test, pred_test, "Test")
    logging.info("\n%s", "=" * 72)
    logging.info("%s", title)
    logging.info("[Val]  MAE %.4f | MSE %.4f | RMSE %.4f | R2 %.4f", val["Val_MAE"], val["Val_MSE"], val["Val_RMSE"], val["Val_R2"])
    logging.info("[Test] MAE %.4f | MSE %.4f | RMSE %.4f | R2 %.4f", test["Test_MAE"], test["Test_MSE"], test["Test_RMSE"], test["Test_R2"])
    return {**val, **test}


# =============================================================================
# 5. Main: leakage-controlled cross-fitting
# =============================================================================
def main():
    set_seed(42)
    validate_files()
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    logging.info("Device: %s", Config.DEVICE)
    logging.info("DL and all seven ML models share fold-local Yeo-Johnson "
                 "standardized targets; predictions and ensemble optimization use pLD50.")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        missing = {Config.SMILES_COL, Config.LABEL_COL} - set(frame.columns)
        if missing:
            raise KeyError(f"{name}.csv is missing columns: {sorted(missing)}")

    y_train_original = train_df[Config.LABEL_COL].to_numpy(dtype=np.float64)
    y_val_original = val_df[Config.LABEL_COL].to_numpy(dtype=np.float64)
    y_test_original = test_df[Config.LABEL_COL].to_numpy(dtype=np.float64)
    train_raw = get_offline_features(train_df[Config.SMILES_COL].astype(str).to_numpy(), "train")
    val_raw = get_offline_features(val_df[Config.SMILES_COL].astype(str).to_numpy(), "val")
    test_raw = get_offline_features(test_df[Config.SMILES_COL].astype(str).to_numpy(), "test")

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    feature_dim = Config.DEEP_FEATURE_DIM * len(Config.SEEDS)
    oof_dl_predictions = np.zeros((n_train, len(Config.SEEDS)), dtype=np.float64)
    oof_candidates = np.zeros((n_train, len(MODEL_NAMES)), dtype=np.float64)
    val_candidate_sum = np.zeros((n_val, len(MODEL_NAMES)), dtype=np.float64)
    test_candidate_sum = np.zeros((n_test, len(MODEL_NAMES)), dtype=np.float64)

    outer_cv = KFold(n_splits=Config.KFOLD_SPLITS, shuffle=True, random_state=42)
    all_indices = np.arange(n_train)
    fold_records = []

    for fold, (outer_train_idx, outer_oof_idx) in enumerate(outer_cv.split(all_indices), start=1):
        logging.info("\n%s OUTER FOLD %d/%d %s", "=" * 18, fold, Config.KFOLD_SPLITS, "=" * 18)

        # The split is fixed within an outer fold and shared by all seeds.
        artifact_context(Config.OUTPUT_DIR, "MSformer3DMLP", fold)
        inner_train_idx, inner_stop_idx = train_test_split(
            outer_train_idx,
            test_size=Config.INNER_STOP_FRACTION,
            random_state=1000 + fold,
            shuffle=True,
        )

        # Stage A preprocessors are fitted only on inner-train data.
        inner_phys_scaler, inner_desc_scaler, inner_label_transformer = fit_fold_preprocessors(
            train_raw, inner_train_idx, y_train_original
        )
        inner_train_data = transform_data(train_raw, inner_train_idx, inner_phys_scaler, inner_desc_scaler)
        inner_stop_data = transform_data(train_raw, inner_stop_idx, inner_phys_scaler, inner_desc_scaler)
        inner_train_y = inner_label_transformer.transform(y_train_original[inner_train_idx].reshape(-1, 1)).ravel()
        inner_stop_y = inner_label_transformer.transform(y_train_original[inner_stop_idx].reshape(-1, 1)).ravel()

        save_inner_transformer(inner_label_transformer, inner_train_idx, inner_stop_idx)
        selected_epochs = {}
        for seed in Config.SEEDS:
            best_epoch, best_loss = select_epoch_without_outer_holdout(
                inner_train_data, inner_train_y, inner_stop_data, inner_stop_y, seed
            )
            selected_epochs[seed] = int(best_epoch)
            logging.info("Fold %d seed %d | inner-stop best epoch %d | loss %.6f", fold, seed, best_epoch, best_loss)

        # Stage B preprocessors are refitted on all outer-training samples.
        # The untouched outer OOF split remains unavailable until final extraction.
        fold_phys_scaler, fold_desc_scaler, fold_label_transformer = fit_fold_preprocessors(
            train_raw, outer_train_idx, y_train_original
        )
        outer_train_data = transform_data(train_raw, outer_train_idx, fold_phys_scaler, fold_desc_scaler)
        outer_oof_data = transform_data(train_raw, outer_oof_idx, fold_phys_scaler, fold_desc_scaler)
        fold_val_data = transform_data(val_raw, np.arange(n_val), fold_phys_scaler, fold_desc_scaler)
        fold_test_data = transform_data(test_raw, np.arange(n_test), fold_phys_scaler, fold_desc_scaler)
        outer_train_y = fold_label_transformer.transform(y_train_original[outer_train_idx].reshape(-1, 1)).ravel()
        save_preprocessors(fold_label_transformer, fold_phys_scaler, fold_desc_scaler, outer_train_idx, outer_oof_idx)

        outer_train_loader = make_loader(outer_train_data, outer_train_y, shuffle=True, drop_last=True)
        outer_train_eval_loader = make_loader(outer_train_data, shuffle=False)
        outer_oof_loader = make_loader(outer_oof_data, shuffle=False)
        val_loader = make_loader(fold_val_data, shuffle=False)
        test_loader = make_loader(fold_test_data, shuffle=False)

        fold_models = []
        for seed in Config.SEEDS:
            set_seed(seed)
            final_model = MSformer3DMLP(Config).to(Config.DEVICE)
            train_for_epochs(final_model, outer_train_loader, selected_epochs[seed])
            final_model.eval()
            checkpoint = os.path.join(Config.CHECKPOINT_DIR, f"fold{fold}_seed{seed}_epoch{selected_epochs[seed]}.pth")
            torch.save(final_model.state_dict(), checkpoint)
            save_neural_model(final_model, seed, selected_epochs[seed])
            fold_models.append(final_model)

        fold_train_features, _ = extract_features_predictions(
            outer_train_eval_loader, fold_models, fold_label_transformer
        )
        fold_oof_features, fold_oof_predictions = extract_features_predictions(
            outer_oof_loader, fold_models, fold_label_transformer
        )
        fold_val_features, fold_val_predictions = extract_features_predictions(
            val_loader, fold_models, fold_label_transformer
        )
        fold_test_features, fold_test_predictions = extract_features_predictions(
            test_loader, fold_models, fold_label_transformer
        )

        oof_dl_predictions[outer_oof_idx] = fold_oof_predictions

        # Each ML pipeline is fitted and evaluated inside one coherent fold-local
        # hidden space. Hidden coordinates from independently trained outer folds
        # are never mixed into a global feature matrix.
        # Share exactly the outer-fold Yeo-Johnson transformer and standardized
        # training targets used by all five formally trained deep networks.
        # Inverse-transform every ML prediction BEFORE fold averaging/ensembling.
        fold_ml_models = [
            clone(template).fit(fold_train_features, outer_train_y)
            for template in make_ml_pipelines()
        ]
        fold_ml_oof = np.column_stack([
            predict_ml_pld50(model, fold_oof_features, fold_label_transformer)
            for model in fold_ml_models
        ])
        fold_ml_val = np.column_stack([
            predict_ml_pld50(model, fold_val_features, fold_label_transformer)
            for model in fold_ml_models
        ])
        fold_ml_test = np.column_stack([
            predict_ml_pld50(model, fold_test_features, fold_label_transformer)
            for model in fold_ml_models
        ])
        oof_candidates[outer_oof_idx] = np.column_stack([
            fold_oof_predictions.mean(axis=1), fold_ml_oof
        ])
        val_candidate_sum += np.column_stack([
            fold_val_predictions.mean(axis=1), fold_ml_val
        ])
        test_candidate_sum += np.column_stack([
            fold_test_predictions.mean(axis=1), fold_ml_test
        ])

        fold_records.append({
            "fold": fold,
            "outer_train_size": int(len(outer_train_idx)),
            "outer_oof_size": int(len(outer_oof_idx)),
            "inner_train_size": int(len(inner_train_idx)),
            "inner_stop_size": int(len(inner_stop_idx)),
            "selected_epochs": {str(seed): int(epoch) for seed, epoch in selected_epochs.items()},
            "shared_target_yeojohnson_lambda": float(fold_label_transformer.lambdas_[0]),
        })
        save_ml_models(fold_ml_models)
        del fold_models, fold_ml_models, outer_train_loader, outer_train_eval_loader, outer_oof_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_candidates = val_candidate_sum / Config.KFOLD_SPLITS
    test_candidates = test_candidate_sum / Config.KFOLD_SPLITS
    dl_val = val_candidates[:, 0]
    dl_test = test_candidates[:, 0]

    dl_metrics = log_metrics("DL-E2E (rigorous cross-fitted training)", y_val_original, dl_val, y_test_original, dl_test)

    static_val = val_candidates[:, 1:].mean(axis=1)
    static_test = test_candidates[:, 1:].mean(axis=1)
    static_metrics = log_metrics("Static ensemble (seven ML pipelines)", y_val_original, static_val, y_test_original, static_test)

    logging.info("Optimizing weights from untouched outer-OOF pipeline predictions...")
    artifact_context(Config.OUTPUT_DIR, "MSformer3DMLP", "ensemble")
    weights, candidate_mse, kept, slsqp_result = optimize_weights(oof_candidates, y_train_original)
    save_candidates(oof_candidates, val_candidates, test_candidates, weights, y_train_original, y_val_original, y_test_original)
    dynamic_val = val_candidates @ weights
    dynamic_test = test_candidates @ weights
    export_predictions(val_df, dynamic_val, Config.LABEL_COL, Config.SMILES_COL, Config.OUTPUT_DIR, "val")
    export_predictions(test_df, dynamic_test, Config.LABEL_COL, Config.SMILES_COL, Config.OUTPUT_DIR, "test")
    dynamic_metrics = log_metrics("Dynamic SLSQP ensemble (complete model)", y_val_original, dynamic_val, y_test_original, dynamic_test)

    result_table = pd.DataFrame([
        {"Model": "DL-E2E", **dl_metrics},
        {"Model": "Static-Ensemble", **static_metrics},
        {"Model": "Dynamic-SLSQP", **dynamic_metrics},
    ])
    result_table.to_csv(os.path.join(Config.OUTPUT_DIR, "table_model_results.csv"), index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame([{"Model": name, "Weight": weight, "Repeated_OOF_MSE": mse} for name, weight, mse in zip(MODEL_NAMES, weights, candidate_mse)]).to_csv(
        os.path.join(Config.OUTPUT_DIR, "dynamic_weights.csv"), index=False, encoding="utf-8-sig", float_format="%.8f"
    )
    np.savez_compressed(
        os.path.join(Config.OUTPUT_DIR, "cross_fitted_arrays.npz"),
        oof_dl_predictions=oof_dl_predictions,
        oof_candidates=oof_candidates,
        val_candidates=val_candidates,
        test_candidates=test_candidates,
        dynamic_val=dynamic_val,
        dynamic_test=dynamic_test,
    )
    diagnostics = {
        "device": str(Config.DEVICE),
        "target_processing": {
            "input_and_output_scale": "pLD50",
            "dl_and_ml_training_target": "Yeo-Johnson with standardize=True",
            "transformer_fit_scope": "outer-training labels only for formal training",
            "shared_dl_ml_transformer": True,
            "inverse_transform_before_fold_averaging": True,
            "ensemble_optimization_scale": "pLD50",
        },
        "feature_dimensions": {"tabular": Config.TABULAR_DIM, "descriptor3d": Config.DESC3D_DIM, "single_deep": Config.DEEP_FEATURE_DIM, "five_seed_deep": feature_dim},
        "outer_folds": fold_records,
        "model_names": MODEL_NAMES,
        "candidate_mse_original_scale": {name: float(value) for name, value in zip(MODEL_NAMES, candidate_mse)},
        "retained_candidates": [MODEL_NAMES[index] for index in kept],
        "weights": {name: float(value) for name, value in zip(MODEL_NAMES, weights)},
        "slsqp_success": bool(slsqp_result.success),
        "slsqp_message": str(slsqp_result.message),
        "results": result_table.to_dict(orient="records"),
    }
    with open(os.path.join(Config.OUTPUT_DIR, "diagnostics.json"), "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)

    logging.info("\nFinal result table:\n%s", result_table.round(4).to_string(index=False))
    logging.info("All outputs saved to: %s", os.path.abspath(Config.OUTPUT_DIR))


if __name__ == "__main__":
    main()
