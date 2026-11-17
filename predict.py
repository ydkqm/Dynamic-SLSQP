"""Inference through all five saved fold pipelines; no label is required."""
import argparse,json,os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def predict_arrays(model, raw, run_dir):
 root=Path(run_dir);meta=json.loads((root/'run_metadata.json').read_text(encoding='utf-8'))
 diag=json.loads((root/'diagnostics.json').read_text(encoding='utf-8'))
 config=meta['configuration']
 for k in ['CHEMBERTA_DIM','PHYS_DIM','MACCS_DIM','FP_DIM','TABULAR_DIM','DESC3D_DIM','CHEM_HIDDEN_DIM','TAB_HIDDEN_DIM','SPATIAL_HIDDEN_DIM','DEEP_FEATURE_DIM','BATCH_SIZE','SEEDS']:
  if k in config:setattr(model.Config,k,config[k])
 n=len(raw[0]);total=np.zeros((n,8),dtype=np.float64)
 for rec in diag['outer_folds']:
  directory=root/'saved_run'/'MSformer3DMLP'/f"fold_{rec['fold']}"
  ml_path=directory/'ml_models.joblib'
  if not ml_path.exists():raise FileNotFoundError(f'{ml_path}: training must use --save-ml-models; inference does not silently refit regressors.')
  prep=joblib.load(directory/'preprocessors.joblib')
  data=model.transform_data(raw,np.arange(n),prep['phys_scaler'],prep['desc_scaler'])
  networks=[]
  for seed in model.Config.SEEDS:
   epoch=rec['selected_epochs'][str(seed)]
   net=model.MSformer3DMLP(model.Config).to(model.Config.DEVICE)
   net.load_state_dict(model.torch.load(directory/f'seed_{seed}_epoch_{epoch}.pth',map_location=model.Config.DEVICE,weights_only=True));net.eval();networks.append(net)
  features,dl=model.extract_features_predictions(model.make_loader(data),networks,prep['label_transformer'])
  ml=joblib.load(ml_path)
  if len(ml)!=7:raise ValueError('Expected seven saved ML pipelines')
  pred=np.column_stack([dl.mean(axis=1)]+[model.predict_ml_pld50(m,features,prep['label_transformer']) for m in ml])
  total+=pred
  del networks,ml
  if model.torch.cuda.is_available():model.torch.cuda.empty_cache()
 candidates=total/len(diag['outer_folds']);weights=np.array([diag['weights'][k] for k in model.MODEL_NAMES])
 return candidates,candidates@weights

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--input-csv',required=True)
 p.add_argument('--output-csv',required=True);p.add_argument('--cache-dir',required=True);p.add_argument('--offline',action='store_true')
 a=p.parse_args()
 if a.offline:os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 if Path(a.output_csv).exists():raise FileExistsError(a.output_csv)
 import main_model as model
 from run import configure_cache,ordered_smiles_hash
 meta=json.loads((Path(a.run_dir)/'run_metadata.json').read_text(encoding='utf-8'))
 model.Config.CHEMBERTA_MODEL=meta['configuration']['CHEMBERTA_MODEL']
 column=meta['configuration']['SMILES_COL'];frame=pd.read_csv(a.input_csv)
 if frame.empty or column not in frame or frame[column].isna().any():raise ValueError('Nonempty input with valid SMILES column required')
 smiles=frame[column].astype(str).tolist()
 if any(model.Chem.MolFromSmiles(s) is None for s in smiles):raise ValueError('Invalid SMILES in prediction input')
 configure_cache(model,a.cache_dir)
 raw=model.get_offline_features(smiles,'predict_'+ordered_smiles_hash(smiles)[:16])
 candidates,pred=predict_arrays(model,raw,a.run_dir)
 result=frame.copy()
 for i,name in enumerate(model.MODEL_NAMES):result[f'pred_{name}']=candidates[:,i]
 result['pred_pLD50']=pred
 Path(a.output_csv).parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output_csv,index=False)
 print(f'Predictions saved: {a.output_csv}')

if __name__=='__main__':main()
