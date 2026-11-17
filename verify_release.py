"""CPU-only structural, preprocessing parity, and saved-predictor replay checks."""
import ast, hashlib, importlib.util, json, tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import main_model as m
from prepare_data import prepare
from predict import predict_arrays
from run import configure_cache

def main():
 torch.set_num_threads(1)
 with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
  root=Path(td)
  # Small genuine molecules: verify preparation counts, deterministic splits and no overlap.
  raw=root/'raw.csv'
  smiles=['C','CC','CCC','CCCC','CCCCC','CCCCCC','O','CO','CCO','CCCO','N','CN','CCN','CCCN','C#N','CC#N','C=C','CC=C','C#C','CC#C','c1ccccc1','c1ccncc1','C1CCCCC1','C1CCOC1','CC(=O)O','CC(=O)N','Cl','Br','P','S','CS','CCS','CCCl','CCBr','C(F)(F)F','CC(F)F','C=O','CC=O']
  pd.DataFrame({'Canonical SMILES':smiles+[smiles[0],None,'not_smiles'],
                'mouse_intravenous_LD50':list(np.linspace(1,5,len(smiles)))+[1,2,3]}).to_csv(raw,index=False)
  a=prepare(raw,root/'one');b=prepare(raw,root/'two')
  assert a['splits']==b['splits']
  sets=[set(pd.read_csv(root/'one'/f'{n}.csv')['Canonical SMILES']) for n in ['train','val','test']]
  assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
  # Compare unchanged archive logic using the same small dataset if archive paths are supplied.
  import argparse
  parser=argparse.ArgumentParser();parser.add_argument('--archive-dir');parser.add_argument('--original-preparation-dir');args=parser.parse_args()
  if args.archive_dir:
   old=ast.parse((Path(args.archive_dir)/'03_ensemble.py').read_text(encoding='utf-8'))
   new=ast.parse(Path(m.__file__).read_text(encoding='utf-8'))
   select=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
   old_nodes=select(old);new_nodes=select(new)
   # The only authorized scientific change is removing the error-threshold rule.
   for name in old_nodes:
    if name not in ['Config','optimize_weights']:
     assert old_nodes[name]==new_nodes[name],f'Unexpected change: {name}'
   assert not hasattr(m.Config,'PRUNE_RATIO')
  if args.original_preparation_dir:
   base=Path(args.original_preparation_dir)
   def load(name,file):
    sp=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);return module
   clean=load('original_ry',base/'ry.py');clean.Config.INPUT_CSV=str(raw);clean.Config.OUTPUT_CSV=str(root/'original_clean.csv')
   clean.clean_redundant_molecules(clean.Config)
   split=load('original_split',base/'split_data.py');split.RAW_CSV_PATH=str(root/'original_clean.csv');split.OUTPUT_DIR=str(root/'original_split');split.main()
   for n in ['train','val','test']:
    assert (root/'one'/f'{n}.csv').read_bytes()==(root/'original_split'/f'{n}.csv').read_bytes(),n
  # Cache check must reject a reused feature filename with different ordered SMILES.
  original=m.get_offline_features
  m.get_offline_features=lambda ss,sp: tuple(np.ones((len(ss),d),dtype=np.float32) for d in [384,10,167,1024,23])
  configure_cache(m,root/'cache');m.get_offline_features(['C','CC'],'train')
  (root/'cache'/'v40_etkdg_3dmlp_train.pt').write_bytes(b'fixture')
  try:m.get_offline_features(['CC','C'],'train')
  except ValueError:pass
  else:raise AssertionError('Cache mismatch was not rejected')
  m.get_offline_features=original
  # Replay all five saved pipelines on synthetic features without expensive training.
  m.Config.DEVICE=torch.device('cpu');torch.manual_seed(42);rng=np.random.default_rng(42)
  dims=[384,10,167,1024,23]
  train=tuple(rng.normal(size=(20,d)).astype(np.float32) for d in dims)
  test=tuple(rng.normal(size=(7,d)).astype(np.float32) for d in dims)
  train[-1][:,-1]=1;test[-1][:,-1]=1
  y=np.linspace(1,5,20);reference=np.zeros((7,8));records=[]
  run=root/'saved_fixture';run.mkdir()
  for fold in range(1,6):
   directory=run/'saved_run'/'MSformer3DMLP'/f'fold_{fold}';directory.mkdir(parents=True)
   phys,desc,label=m.fit_fold_preprocessors(train,np.arange(20),y)
   joblib.dump({'phys_scaler':phys,'desc_scaler':desc,'label_transformer':label},directory/'preprocessors.joblib')
   train_data=m.transform_data(train,np.arange(20),phys,desc);test_data=m.transform_data(test,np.arange(7),phys,desc)
   nets=[]
   for seed in m.Config.SEEDS:
    torch.manual_seed(seed+fold);net=m.MSformer3DMLP(m.Config);net.eval();nets.append(net)
    torch.save(net.state_dict(),directory/f'seed_{seed}_epoch_1.pth')
   tf,_=m.extract_features_predictions(m.make_loader(train_data),nets,label)
   ef,dl=m.extract_features_predictions(m.make_loader(test_data),nets,label)
   target=label.transform(y[:,None]).ravel()
   regressors=[Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=1+i))]).fit(tf,target) for i in range(7)]
   joblib.dump(regressors,directory/'ml_models.joblib')
   reference+=np.column_stack([dl.mean(axis=1)]+[m.predict_ml_pld50(q,ef,label) for q in regressors])/5
   records.append({'fold':fold,'selected_epochs':{str(seed):1 for seed in m.Config.SEEDS}})
  config={k:v for k,v in vars(m.Config).items() if k.isupper() and isinstance(v,(str,int,float,list))}
  weights=np.array([.5]+[.5/7]*7)
  (run/'run_metadata.json').write_text(json.dumps({'configuration':config}),encoding='utf-8')
  (run/'diagnostics.json').write_text(json.dumps({'outer_folds':records,'weights':dict(zip(m.MODEL_NAMES,weights.tolist()))}),encoding='utf-8')
  candidates,final=predict_arrays(m,test,run)
  np.testing.assert_allclose(candidates,reference,rtol=0,atol=1e-12)
  np.testing.assert_allclose(final,reference@weights,rtol=0,atol=1e-12)
  m.artifact_context(root/'optimization_fixture','self_test','ensemble')
  w,_,_,_=m.optimize_weights(reference,np.linspace(2,4,7))
  assert np.all(w>=0) and np.isclose(w.sum(),1)
  # A deliberately poor candidate must still participate in optimization.
  target=np.linspace(2,4,7)
  deliberately_poor=np.tile(target[:,None],(1,8))
  deliberately_poor[:,7]+=100
  _,errors,kept,_=m.optimize_weights(deliberately_poor,target)
  assert list(kept)==list(range(8)) and errors[7]>1000
 print('PASS: deterministic preparation, overlap protection, cache-order rejection, five-fold saved prediction replay, weight constraints, all-eight retention including a deliberately poor candidate.')
 print('Synthetic fixture checks only. This is not a new full model training or an accuracy benchmark.')

if __name__=='__main__':main()
