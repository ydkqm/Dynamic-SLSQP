"""Train the final model from fixed splits, or prepare raw data then train."""
import argparse, hashlib, importlib.metadata, json, os, sys
from pathlib import Path
from prepare_data import prepare,sha256

def ordered_smiles_hash(smiles):
 return hashlib.sha256(json.dumps([str(s) for s in smiles],ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()

def configure_cache(model, cache_dir, trust_legacy=False):
 original=model.get_offline_features
 model.Config.DATA_DIR=str(Path(cache_dir).resolve());Path(cache_dir).mkdir(parents=True,exist_ok=True)
 def checked(smiles,split):
  path=Path(model.Config.DATA_DIR)/f'{model.Config.CACHE_VERSION}_{split}.pt'
  stamp=Path(str(path)+'.provenance.json')
  expected={'ordered_smiles_sha256':ordered_smiles_hash(smiles),'rows':len(smiles),
            'chemberta_model':model.Config.CHEMBERTA_MODEL,'cache_version':model.Config.CACHE_VERSION}
  if path.exists():
   if stamp.exists():
    if json.loads(stamp.read_text(encoding='utf-8'))!=expected:raise ValueError(f'Cache provenance mismatch: {path}')
   elif not trust_legacy:
    raise ValueError(f'Legacy cache has no ordered-SMILES provenance: {path}. Rebuild it in a new directory, or explicitly use --trust-legacy-cache after checking the original data/order.')
  arrays=original(smiles,split)
  if len(arrays)!=5:raise ValueError('Expected five cached arrays')
  for array,dim in zip(arrays,[384,10,167,1024,23]):
   if array.shape!=(len(smiles),dim):raise ValueError(f'Wrong feature shape: {array.shape}')
   if not model.np.isfinite(array).all():raise ValueError('Non-finite cache features')
  if not stamp.exists():stamp.write_text(json.dumps(expected,ensure_ascii=False,indent=2),encoding='utf-8')
  return arrays
 model.get_offline_features=checked

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--raw-csv');p.add_argument('--data-dir',required=True)
 p.add_argument('--cache-dir',required=True);p.add_argument('--output-dir',required=True)
 p.add_argument('--smiles-col',default='Canonical SMILES');p.add_argument('--label-col',default='mouse_intravenous_LD50')
 p.add_argument('--chemberta-model',default='DeepChem/ChemBERTa-77M-MTR')
 p.add_argument('--offline',action='store_true');p.add_argument('--trust-legacy-cache',action='store_true')
 p.add_argument('--save-ml-models',action='store_true',help='Required for subsequent predict.py; can use many GB')
 a=p.parse_args()
 if a.offline:os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 os.environ['SAVE_LARGE_ML_MODELS']='1' if a.save_ml_models else '0'
 if a.raw_csv:prepare(a.raw_csv,a.data_dir,a.smiles_col,a.label_col)
 out=Path(a.output_dir).resolve()
 if out.exists() and any(out.iterdir()):raise FileExistsError('Output is nonempty; choose a new output directory. This script does not resume partial runs.')
 import main_model as model
 data=Path(a.data_dir).resolve()
 for name in ['train','val','test']:
  setattr(model.Config,name.upper()+'_CSV',str(data/f'{name}.csv'))
 model.Config.SMILES_COL=a.smiles_col;model.Config.LABEL_COL=a.label_col
 model.Config.CHEMBERTA_MODEL=a.chemberta_model
 model.Config.OUTPUT_DIR=str(out);model.Config.CHECKPOINT_DIR=str(out/'checkpoints')
 model.validate_files()
 # Verify finite numeric labels, valid structures and split overlap before training.
 seen=set()
 for name in ['train','val','test']:
  frame=model.pd.read_csv(data/f'{name}.csv')
  if frame[a.smiles_col].isna().any():raise ValueError(f'Missing SMILES in {name}')
  ss=frame[a.smiles_col].astype(str).tolist()
  if len(set(ss))!=len(ss) or seen.intersection(ss):raise ValueError(f'Duplicate/overlapping exact SMILES in {name}')
  seen.update(ss)
  if any(model.Chem.MolFromSmiles(s) is None for s in ss):raise ValueError(f'Invalid SMILES in {name}')
  y=model.pd.to_numeric(frame[a.label_col],errors='raise').to_numpy(dtype=float)
  if not model.np.isfinite(y).all():raise ValueError(f'Invalid labels in {name}')
 configure_cache(model,a.cache_dir,a.trust_legacy_cache)
 out.mkdir(parents=True,exist_ok=True)
 versions={}
 for n in ['torch','numpy','pandas','scipy','scikit-learn','rdkit','transformers','joblib']:
  try:versions[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:versions[n]='unknown'
 metadata={'python':sys.version,'packages':versions,'configuration':{k:v for k,v in vars(model.Config).items() if k.isupper() and isinstance(v,(str,int,float,list))},
           'input_sha256':{n:sha256(data/f'{n}.csv') for n in ['train','val','test']},
           'save_ml_models':a.save_ml_models,'legacy_cache_trust':a.trust_legacy_cache}
 (out/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 model.main()

if __name__=='__main__':main()
