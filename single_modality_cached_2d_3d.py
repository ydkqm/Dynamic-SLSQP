#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two single-modality controls for Section 3.1; existing raw caches ONLY.

Runs exactly two configurations, not the previous nine fusion configurations:
  Tabular-Only: phys10 + MACCS167 + Morgan1024 = 1201 dimensions.
  Conformer-Only: 22 continuous conformer descriptors + 1 validity flag = 23.

The feature encoders and prediction backbone follow the fusion experiment:
  Tabular-Only: 1201 -> 256 encoder -> 256 -> 128 backbone -> 1 output.
  Conformer-Only: 23 -> 128 -> 64 encoder -> 256 -> 128 backbone -> 1 output.
Every encoder/backbone hidden linear layer uses BatchNorm1d then GELU.
No Fusion Gate, no ChemBERTa input, no classical regressors, no SLSQP.

Reads train.csv, val.csv, test.csv without changing the fixed data split.
Requires v40_etkdg_3dmlp_train.pt / val.pt / test.pt produced by the main code.
Validates shapes, finiteness, and rowwise raw physical descriptors/fingerprints.
Does not generate conformers, load a language model, or access the network.

Five outer folds (seed 42), five network seeds, 10% inner stopping split,
1-50 epoch search, fresh final training, shared fold-local Yeo-Johnson labels,
AdamW(lr=3e-4, weight_decay=1e-2), batch=64, StableLogCosh loss.
Inverse-transform EACH prediction before averaging seeds and folds.

Run:
  python single_modality_cached_2d_3d.py --data-dir /root/autodl-tmp
  python single_modality_cached_2d_3d.py --data-dir /root/autodl-tmp --main-cache-dir /root/autodl-tmp --output-dir /root/autodl-tmp/single_modality_results
  python single_modality_cached_2d_3d.py --data-dir /root/autodl-tmp --output-dir /root/autodl-tmp/single_modality_results --resume

Dependencies: torch, numpy, pandas, scikit-learn, rdkit, joblib, tqdm.
Only this .py file is needed; no other custom modules are imported.
"""
import argparse
import csv
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import logging
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class Config:
    SEEDS=[42,1024,2026,8888,9999]
    KFOLD_SPLITS=5
    INNER_STOP_FRACTION=0.10
    NUM_EPOCHS=50
    BATCH_SIZE=64
    LEARNING_RATE=3e-4
    WEIGHT_DECAY=1e-2
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAIN_CACHE_DIR="."
    DATA_DIR="."
    REQUIRE_MAIN_CACHE=True

_context=None


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def record_epoch(epoch, loss, seed):
    path = _context / f"inner_search_seed_{seed}.csv"
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(["epoch", "inner_stop_logcosh"])
        writer.writerow([epoch, loss])
    logging.info("%s | seed %d | search epoch %d/%d | inner-stop LogCosh %.6f", _context.parent.name+"/"+_context.name, seed, epoch, Config.NUM_EPOCHS, loss)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def load_main_model_cache(smiles_list, split_name):
    path=Path(Config.MAIN_CACHE_DIR)/f"v40_etkdg_3dmlp_{split_name}.pt"
    if not path.is_file():
        if Config.REQUIRE_MAIN_CACHE:
            raise FileNotFoundError(f"Required main-model raw feature cache is absent: {path}")
        return None
    logging.info("Loading existing MAIN MODEL feature cache: %s",path)
    data=torch.load(path,map_location="cpu",weights_only=False)
    expected=[384,10,167,1024,23]
    if not isinstance(data,(tuple,list)) or len(data)!=5:
        raise ValueError(f"Expected five raw feature arrays in {path}")
    data=tuple(x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x) for x in data)
    for array,dim in zip(data,expected):
        if array.shape!=(len(smiles_list),dim) or not np.isfinite(array).all():
            raise ValueError(f"Raw cache shape/finiteness mismatch: {path}; actual={array.shape}, expected={(len(smiles_list),dim)}")
    metadata_path=Path(str(path)+".json")
    metadata=json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.is_file() else {}
    if "samples" in metadata and int(metadata["samples"])!=len(smiles_list):
        raise ValueError(f"Cache metadata sample count mismatch: {path}")
    flags=data[4][:,-1]
    if not np.isin(flags,[0,1]).all():
        raise ValueError(f"Invalid 3D validity indicators: {path}")
    # Old v40 caches did not record ordered SMILES. Check EVERY row against the
    # input's raw 10 descriptors and binary fingerprints. Do not verify by shape
    # alone or silently treat a mismatched cache as reusable. This check is not
    # a reconstruction of the historical ChemBERTa/3D extraction provenance.
    for row,smiles in enumerate(tqdm(smiles_list,desc=f"Verify raw 2D/cache order-{split_name}")):
        mol=Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid input SMILES at row {row}; cache reuse stopped")
        phys=finite_array([function(mol) for function in PHYS_FUNCTIONS],10)
        maccs=finite_array(list(MACCSkeys.GenMACCSKeys(mol)),167)
        fp=finite_array(list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)),1024)
        if not np.allclose(phys,data[1][row],rtol=1e-5,atol=1e-5) or not np.array_equal(maccs,data[2][row]) or not np.array_equal(fp,data[3][row]):
            raise ValueError(f"Cache and current CSV differ at {split_name} row {row}. Check sample order and RDKit version; old raw features must not be paired with different rows.")
    signature=hashlib.sha256("\n".join(map(str,smiles_list)).encode("utf-8")).hexdigest()
    save_json(Path(Config.DATA_DIR)/f"main_cache_reuse_{split_name}.json",{
        "source_cache":str(path.resolve()),"cache_sha256":file_hash(path),
        "current_ordered_smiles_sha256":signature,"samples":len(smiles_list),
        "dimensions":expected,"historical_metadata":metadata,
        "verification":"all rows: raw physical descriptors and MACCS/Morgan fingerprints matched current CSV",
        "historical_ordered_smiles_record_available":False,
        "scope":"reuse original raw ChemBERTa/2D/3D features from the same data preparation run; fold-specific scalers are fitted anew"})
    logging.info("Verified and reused %s; skipped ALL ChemBERTa and conformer extraction",path)
    return data

def encoder(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.GELU())

def backbone(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
                         nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())

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


class TabularOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tab_proj=encoder(1201,256)
        self.backbone=backbone(256)
        self.regressor=nn.Linear(128,1)
    def forward(self, x):
        return self.regressor(self.backbone(self.tab_proj(x))).squeeze(-1)

class ConformerOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_mlp=Descriptor3DMLP(23,64)
        self.backbone=backbone(64)
        self.regressor=nn.Linear(128,1)
    def forward(self, x):
        return self.regressor(self.backbone(self.spatial_mlp(x))).squeeze(-1)

MODEL_SPECS=[
    ("Tabular-Only",TabularOnlyModel,"二维表格",False),
    ("Conformer-Only",ConformerOnlyModel,"三维构象",True),
]

class ToxicityDataset(Dataset):
    def __init__(self, x, y=None):
        self.x=x
        self.y=y
    def __len__(self):return len(self.x)
    def __getitem__(self,index):
        result={"x":torch.tensor(self.x[index],dtype=torch.float32)}
        if self.y is not None:
            result["y"]=torch.tensor(self.y[index],dtype=torch.float32)
        return result

def move_inputs(batch):
    return (batch["x"].to(Config.DEVICE),)

def fit_preprocessors(raw,indices,target,use_3d):
    label=PowerTransformer(method="yeo-johnson",standardize=True).fit(target[indices,None])
    phys=None
    descriptor=None
    if use_3d:
        valid_indices=indices[raw[4][indices,-1]>0.5]
        if len(valid_indices)==0:
            raise RuntimeError("No valid 3D conformer in the current training subset")
        descriptor=StandardScaler().fit(raw[4][valid_indices,:22])
    else:
        phys=StandardScaler().fit(raw[1][indices])
    return phys,descriptor,label

def prepare_features(raw,indices,phys,descriptor):
    if descriptor is not None:
        selected=raw[4][indices].copy()
        selected[:,:22]=descriptor.transform(selected[:,:22])
        selected[selected[:,-1]<0.5,:22]=0
    else:
        if phys is None:raise RuntimeError("Missing 2D physical-descriptor scaler")
        selected=np.concatenate([phys.transform(raw[1][indices]),raw[2][indices],raw[3][indices]],axis=1)
    if not np.isfinite(selected).all():raise ValueError("Non-finite transformed model input")
    return (selected.astype(np.float32),)

def save_tables(results,output_dir):
    frame=pd.DataFrame(results)
    frame.to_csv(Path(output_dir)/"single_modality_results_all_metrics.csv",index=False,encoding="utf-8-sig",float_format="%.8f")
    frame[["Model","Features","Val_RMSE","Val_R2","Test_RMSE","Test_R2"]].to_csv(
        Path(output_dir)/"Table_Single_Modality_RMSE_R2.csv",index=False,encoding="utf-8-sig",float_format="%.4f")

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",default=".",help="Existing train.csv / val.csv / test.csv directory")
    parser.add_argument("--main-cache-dir",default=None,help="Existing v40 cache directory; defaults to --data-dir")
    parser.add_argument("--output-dir",default=None,help="Defaults to a new directory with a timestamp")
    parser.add_argument("--smiles-col",default="Canonical SMILES")
    parser.add_argument("--label-col",default="mouse_intravenous_LD50")
    parser.add_argument("--device",default=None,help="cpu or cuda")
    parser.add_argument("--resume",action="store_true",help="Skip completed models under the explicit same output directory; not epoch-level checkpoint recovery")
    parser.add_argument("--no-checkpoints",action="store_true")
    args=parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir pointing to an existing run")
    output=Path(args.output_dir or ("single_modality_cached_"+datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True,exist_ok=True)
    args.output_dir=str(output)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(output/"training.log",encoding="utf-8")])
    RDLogger.DisableLog("rdApp.*")
    if args.device is not None:Config.DEVICE=torch.device(args.device)
    Config.MAIN_CACHE_DIR=str(Path(args.main_cache_dir or args.data_dir).resolve())
    Config.DATA_DIR=str(output/"cache_validation")
    Path(Config.DATA_DIR).mkdir(exist_ok=True)
    cache_paths=[Path(Config.MAIN_CACHE_DIR)/f"v40_etkdg_3dmlp_{split}.pt" for split in ["train","val","test"]]
    for path in cache_paths:
        if not path.is_file():raise FileNotFoundError(f"Required precomputed cache not found: {path}; this script never extracts features")
    versions={name:importlib.metadata.version(name) for name in ["torch","numpy","pandas","scikit-learn","joblib"]}
    versions["rdkit"]=rdBase.rdkitVersion
    paths=[Path(args.data_dir).resolve()/f"{split}.csv" for split in ["train","val","test"]]
    frames=[]
    for path in paths:
        frame=pd.read_csv(path)
        if not {args.smiles_col,args.label_col}.issubset(frame.columns):
            raise ValueError(f"Missing required columns in {path}")
        if frame[args.smiles_col].isna().any():raise ValueError(f"Missing SMILES: {path}")
        y=pd.to_numeric(frame[args.label_col],errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():raise ValueError(f"Invalid labels: {path}")
        if len(frame)<2:raise ValueError(f"Too few rows in {path}")
        frame[args.label_col]=y
        frames.append(frame)
    if len(frames[0])<150:raise ValueError("Too few training samples for the original batch-64 five-fold procedure")
    settings={"script_sha256":file_hash(__file__),"input_sha256":{str(p):file_hash(p) for p in paths},
        "cache_sha256":{str(p):file_hash(p) for p in cache_paths},"versions":versions,
        "smiles_col":args.smiles_col,"label_col":args.label_col,"device":str(Config.DEVICE),
        "seeds":Config.SEEDS,"outer_folds":Config.KFOLD_SPLITS,
        "inner_stop_fraction":Config.INNER_STOP_FRACTION,"epoch_search":Config.NUM_EPOCHS,
        "batch_size":Config.BATCH_SIZE,"lr":Config.LEARNING_RATE,"weight_decay":Config.WEIGHT_DECAY,
        "loss":"StableLogCosh","labels":"existing pLD50; fold-local Yeo-Johnson standardize=True",
        "save_checkpoints":not args.no_checkpoints}
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode("utf-8")).hexdigest()
    manifest=output/"experiment_manifest.json"
    if manifest.is_file():
        old=json.loads(manifest.read_text(encoding="utf-8"))
        if not args.resume or old["run_signature"]!=signature:
            raise FileExistsError("Output directory already belongs to a run; use matching --resume or a new --output-dir")
    save_json(manifest,{"run_signature":signature,"settings":settings,
        "sample_counts":dict(zip(["train","val","test"],map(len,frames))),
        "architectures":{"Tabular-Only":"1201->256 (BN/GELU), backbone 256->256->128 (BN/GELU), Linear128->1",
                         "Conformer-Only":"23->128->64 (BN/GELU), backbone 64->256->128 (BN/GELU), Linear128->1"},
        "evaluation":"Each prediction inverse-transformed first; five seeds averaged within each fold, then five folds averaged; no ML or SLSQP",
        "comparison":"Use as two new single-modality rows beside the previous fusion results; model capacities are not matched"})
    logging.info("ONLY TWO SINGLE-MODALITY MODELS | device %s | samples %s",Config.DEVICE,[len(f) for f in frames])
    set_seed(42)
    raws=[load_main_model_cache(f[args.smiles_col].astype(str).to_numpy(),split) for f,split in zip(frames,["train","val","test"])]
    folds=list(KFold(n_splits=Config.KFOLD_SPLITS,shuffle=True,random_state=42).split(np.arange(len(frames[0]))))
    results=[]
    for spec in MODEL_SPECS:
        results.append(execute_model(spec,frames,raws,folds,args,signature,versions))
        save_tables(results,output)
    logging.info("FINAL SINGLE-MODALITY RESULTS:\n%s",pd.DataFrame(results).to_string(index=False))
    logging.info("All results saved to %s",output)



class StableLogCoshLoss(nn.Module):
    def forward(self, prediction, target):
        error = prediction - target
        return torch.mean(error + F.softplus(-2.0 * error) - np.log(2.0))

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

def select_epoch_without_outer_holdout(model_class, inner_train_data, inner_train_y, inner_stop_data, inner_stop_y, seed):
    """The outer OOF split is never used for epoch selection."""
    set_seed(seed)
    model = model_class(Config).to(Config.DEVICE)
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

def predict_original(model, loader, transformer):
    model.eval()
    values=[]
    with torch.no_grad():
        for batch in loader:
            z=model(*move_inputs(batch)).detach().cpu().numpy().reshape(-1,1)
            restored=transformer.inverse_transform(z).ravel()
            if not np.isfinite(restored).all():
                raise FloatingPointError("Non-finite inverse-transformed prediction; no clipping applied")
            values.append(restored.astype(np.float64))
    return np.concatenate(values)

def experiment_metrics(y,p,prefix):
    mse=float(mean_squared_error(y,p))
    return {prefix+"_MSE":mse,prefix+"_MAE":float(mean_absolute_error(y,p)),
            prefix+"_RMSE":float(np.sqrt(mse)),prefix+"_R2":float(r2_score(y,p))}

def export_prediction(frame,pred,args,path):
    pd.DataFrame({"SMILES":frame[args.smiles_col].astype(str),
                  "y_true":frame[args.label_col].to_numpy(dtype=float),
                  "y_pred":pred}).to_csv(path,index=False,encoding="utf-8-sig")

def execute_model(spec,frames,raws,folds,args,run_signature,versions):
    global _context
    name,model_class,features,use_3d=spec
    directory=Path(args.output_dir)/name
    done=directory/"completed.json"
    if args.resume and done.is_file():
        saved=json.loads(done.read_text(encoding="utf-8"))
        if saved["run_signature"]!=run_signature:
            raise ValueError(f"Run settings or input files changed: {directory}")
        logging.info("Completed result reused: %s",name)
        return saved["result"]
    if directory.exists() and any(directory.iterdir()):
        if args.resume:
            raise RuntimeError(f"Incomplete model directory: {directory}. Use a NEW --output-dir or select the remaining models with --models. Existing results are not overwritten.")
        raise FileExistsError(f"Non-empty model directory: {directory}; use --resume for completed models")
    directory.mkdir(parents=True,exist_ok=True)
    train_frame,val_frame,test_frame=frames
    train_raw,val_raw,test_raw=raws
    y=train_frame[args.label_col].to_numpy(dtype=np.float64)
    yv=val_frame[args.label_col].to_numpy(dtype=np.float64)
    yt=test_frame[args.label_col].to_numpy(dtype=np.float64)
    oof=np.full(len(y),np.nan,dtype=np.float64)
    val_sum=np.zeros(len(yv),dtype=np.float64)
    test_sum=np.zeros(len(yt),dtype=np.float64)
    fold_metrics=[]
    parameter_count=None
    for fold,(train_idx,oof_idx) in enumerate(folds,start=1):
        logging.info("========== %s | OUTER FOLD %d/%d ==========",name,fold,len(folds))
        _context=directory/f"fold_{fold}"
        _context.mkdir()
        inner_idx,stop_idx=train_test_split(train_idx,test_size=Config.INNER_STOP_FRACTION,random_state=1000+fold,shuffle=True)
        ip,id3,il=fit_preprocessors(train_raw,inner_idx,y,use_3d)
        inner_data=prepare_features(train_raw,inner_idx,ip,id3)
        stop_data=prepare_features(train_raw,stop_idx,ip,id3)
        inner_y=il.transform(y[inner_idx,None]).ravel()
        stop_y=il.transform(y[stop_idx,None]).ravel()
        selected={}
        for seed in Config.SEEDS:
            epoch,loss=select_epoch_without_outer_holdout(model_class,inner_data,inner_y,stop_data,stop_y,seed)
            selected[seed]=int(epoch)
            logging.info("%s | fold %d | seed %d | selected epoch %d | loss %.6f",name,fold,seed,epoch,loss)
        phys,desc,label=fit_preprocessors(train_raw,train_idx,y,use_3d)
        formal=prepare_features(train_raw,train_idx,phys,desc)
        holdout=prepare_features(train_raw,oof_idx,phys,desc)
        vd=prepare_features(val_raw,np.arange(len(yv)),phys,desc)
        td=prepare_features(test_raw,np.arange(len(yt)),phys,desc)
        train_y=label.transform(y[train_idx,None]).ravel()
        joblib.dump({"phys_scaler":phys,"desc_scaler":desc,"label_transformer":label,
                     "outer_train_indices":train_idx,"outer_oof_indices":oof_idx,
                     "inner_train_indices":inner_idx,"inner_stop_indices":stop_idx,
                     "inner_phys_scaler":ip,"inner_desc_scaler":id3,
                     "inner_label_transformer":il},_context/"preprocessors.joblib")
        np.savez_compressed(_context/"split_indices.npz",outer_train=train_idx,outer_oof=oof_idx,
                            inner_train=inner_idx,inner_stop=stop_idx)
        seed_oof=[];seed_val=[];seed_test=[]
        train_loader=make_loader(formal,train_y,shuffle=True,drop_last=True)
        oof_loader=make_loader(holdout,shuffle=False)
        val_loader=make_loader(vd,shuffle=False)
        test_loader=make_loader(td,shuffle=False)
        for seed in Config.SEEDS:
            set_seed(seed)
            model=model_class(Config).to(Config.DEVICE)
            parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
            train_for_epochs(model,train_loader,selected[seed])
            if not args.no_checkpoints:
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},_context/f"seed_{seed}_epoch_{selected[seed]}.pth")
            seed_oof.append(predict_original(model,oof_loader,label))
            seed_val.append(predict_original(model,val_loader,label))
            seed_test.append(predict_original(model,test_loader,label))
            logging.info("%s | fold %d | seed %d | formal training and prediction completed",name,fold,seed)
            del model
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        po=np.column_stack(seed_oof);pv=np.column_stack(seed_val);pt=np.column_stack(seed_test)
        oof[oof_idx]=po.mean(axis=1)
        val_fold=pv.mean(axis=1);test_fold=pt.mean(axis=1)
        val_sum+=val_fold;test_sum+=test_fold
        np.savez_compressed(_context/"predictions_pld50.npz",oof_indices=oof_idx,
            seeds=np.array(Config.SEEDS),oof_by_seed=po,val_by_seed=pv,test_by_seed=pt)
        record={"Model":name,"Fold":fold,"Train_n":len(train_idx),"OOF_n":len(oof_idx),
                "Inner_train_n":len(inner_idx),"Inner_stop_n":len(stop_idx),
                "YJ_lambda":float(label.lambdas_[0]),
                **experiment_metrics(yv,val_fold,"Val"),**experiment_metrics(yt,test_fold,"Test")}
        fold_metrics.append(record)
        save_json(_context/"training.json",{"selected_epochs":selected,"metrics":record,
                  "trainable_parameters":parameter_count,"use_3d":use_3d})
        del train_loader,oof_loader,val_loader,test_loader,inner_data,stop_data,formal,holdout,vd,td
        gc.collect()
    if not np.isfinite(oof).all():raise RuntimeError("OOF coverage or finiteness check failed")
    final_val=val_sum/len(folds);final_test=test_sum/len(folds)
    result={"Model":name,"Features":features,"Uses_3D":use_3d,"Trainable_parameters":parameter_count,
            **experiment_metrics(yv,final_val,"Val"),**experiment_metrics(yt,final_test,"Test")}
    export_prediction(val_frame,final_val,args,directory/"val_predictions.csv")
    export_prediction(test_frame,final_test,args,directory/"test_predictions.csv")
    export_prediction(train_frame,oof,args,directory/"oof_predictions.csv")
    pd.DataFrame(fold_metrics).to_csv(directory/"per_fold_val_test_metrics.csv",index=False,encoding="utf-8-sig")
    save_json(done,{"run_signature":run_signature,"result":result,"versions":versions})
    logging.info("COMPLETED %s | Val RMSE %.6f R2 %.6f | Test RMSE %.6f R2 %.6f",name,result["Val_RMSE"],result["Val_R2"],result["Test_RMSE"],result["Test_R2"])
    return result

if __name__=="__main__":
    main()
