"""Original cleaning, ordered greedy deduplication, and fixed 80/10/10 split."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def prepare(raw_csv, output_dir, smiles_col='Canonical SMILES', label_col='mouse_intravenous_LD50'):
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 names=['cleaned_sim080.csv','train.csv','val.csv','test.csv','preparation_audit.json']
 if any((out/n).exists() for n in names): raise FileExistsError('Choose a new output directory; existing splits will not be overwritten.')
 df=pd.read_csv(raw_csv)
 if not {smiles_col,label_col}<=set(df): raise ValueError(f'Required columns: {smiles_col!r}, {label_col!r}; found {list(df)}')
 audit={'raw_sha256':sha256(raw_csv),'raw_rows':len(df),'smiles_column':smiles_col,'label_column':label_col,
        'fingerprint_radius':2,'fingerprint_bits':2048,'retention_rule':'max Tanimoto to previously retained molecules < 0.80',
        'split_seed':42,'label_conversion':'none; input labels must already be pLD50'}
 df=df.dropna(subset=[smiles_col]).drop_duplicates(subset=[smiles_col],keep='first')
 audit['after_missing_smiles_and_string_duplicates']=len(df)
 valid_indices=[];fps=[]
 for i,s in enumerate(tqdm(df[smiles_col].tolist(),desc='Parse molecules')):
  mol=Chem.MolFromSmiles(str(s))
  if mol is not None:
   valid_indices.append(i);fps.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048))
 audit['valid_structures']=len(fps)
 kept=[];kept_fps=[]
 for i,fp in enumerate(tqdm(fps,desc='Ordered greedy Tc filtering')):
  if not kept_fps or max(DataStructs.BulkTanimotoSimilarity(fp,kept_fps))<0.80:
   kept.append(valid_indices[i]);kept_fps.append(fp)
 clean=df.iloc[kept].copy();clean.to_csv(out/'cleaned_sim080.csv',index=False,encoding='utf-8-sig')
 audit['after_greedy_filter']=len(clean)
 # Match original split_data.py: label cleaning occurs AFTER greedy filtering.
 clean=clean.dropna(subset=[smiles_col,label_col]).drop_duplicates(subset=[smiles_col],keep='first')
 clean[label_col]=pd.to_numeric(clean[label_col],errors='coerce');clean=clean.dropna(subset=[label_col])
 if not np.isfinite(clean[label_col].to_numpy()).all(): raise ValueError('Non-finite labels; inspect source data rather than silently replacing labels.')
 train,temp=train_test_split(clean,test_size=0.2,random_state=42)
 val,test=train_test_split(temp,test_size=0.5,random_state=42)
 audit['splits']={}
 for name,frame in [('train',train),('val',val),('test',test)]:
  p=out/f'{name}.csv';frame.to_csv(p,index=False)
  audit['splits'][name]={'rows':len(frame),'sha256':sha256(p)}
 (out/'preparation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(audit,ensure_ascii=False,indent=2));return audit

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-csv',required=True);p.add_argument('--output-dir',required=True)
 p.add_argument('--smiles-col',default='Canonical SMILES');p.add_argument('--label-col',default='mouse_intravenous_LD50')
 a=p.parse_args();prepare(a.raw_csv,a.output_dir,a.smiles_col,a.label_col)
