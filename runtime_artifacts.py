"""Target restoration and passive experiment recording (no RNG operations)."""
from pathlib import Path
import csv
import json
import hashlib
import re
import os
import joblib
import numpy as np

_context = None


def artifact_context(output_dir, experiment, fold):
    global _context
    safe = re.sub(r'[^\w.-]+', '_', str(experiment), flags=re.UNICODE)
    _context = Path(output_dir) / 'saved_run' / safe / f'fold_{fold}'
    _context.mkdir(parents=True, exist_ok=True)
    return _context


def _directory():
    if _context is None:
        raise RuntimeError('Artifact context must be set before recording.')
    return _context


def record_epoch(epoch, loss, seed):
    path = _directory() / f'epoch_search_seed_{seed}.csv'
    fresh = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(['epoch', 'inner_stop_loss'])
        writer.writerow([int(epoch), float(loss)])


def save_preprocessors(label_transformer, phys_scaler=None, desc_scaler=None,
                       train_indices=None, oof_indices=None):
    joblib.dump({'label_transformer': label_transformer,
                 'phys_scaler': phys_scaler, 'desc_scaler': desc_scaler,
                 'outer_train_indices': train_indices, 'outer_oof_indices': oof_indices,
                 'training_target': 'Yeo-Johnson, standardize=True',
                 'prediction_scale': 'pLD50'}, _directory() / 'preprocessors.joblib')


def save_inner_transformer(transformer, train_indices, stop_indices):
    joblib.dump({'label_transformer': transformer, 'inner_train_indices': train_indices,
                 'inner_stop_indices': stop_indices}, _directory() / 'inner_preprocessors.joblib')


def save_neural_model(model, seed, epochs):
    import torch
    # Raw state_dict remains directly loadable with model.load_state_dict(...).
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, _directory() / f'seed_{seed}_epoch_{epochs}.pth')
    with (_directory() / f'seed_{seed}_training.json').open('w', encoding='utf-8') as stream:
        json.dump({'seed': int(seed), 'selected_epochs': int(epochs),
                   'class': model.__class__.__name__}, stream, indent=2)


def save_ml_models(models):
    if os.environ.get('SAVE_LARGE_ML_MODELS', '0') == '1':
        joblib.dump(models, _directory() / 'ml_models.joblib', compress=3)
        return
    # Tree ensembles are very large when repeated for every fold and ablation.
    # They have already produced the OOF/validation/test candidate predictions
    # used by SLSQP, so omit their joblib serialization while retaining all
    # neural checkpoints, preprocessors, predictions, weights and traces.
    manifest = {
        'saved': False,
        'reason': 'joblib serialization disabled to control disk usage',
        'models': [type(model.named_steps.get('model', model)).__name__
                   if hasattr(model, 'named_steps') else type(model).__name__
                   for model in models],
    }
    (_directory() / 'ml_models_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )


def predict_ml_pld50(model, features, transformer):
    transformed = np.asarray(model.predict(features), dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(transformed).all():
        raise FloatingPointError('Non-finite ML prediction in transformed target scale.')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        restored = transformer.inverse_transform(transformed).ravel()
    if not np.isfinite(restored).all():
        raise FloatingPointError('Yeo-Johnson inverse transform is not finite; no clipping was applied.')
    return restored


def record_slsqp(weights, mse, penalty):
    path = _directory() / 'slsqp_objective_evaluations.csv'
    fresh = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        if fresh:
            writer.writerow(['mse_pld50', 'regularization', 'objective', 'weights'])
        writer.writerow([float(mse), float(penalty), float(mse+penalty), json.dumps(list(map(float, weights)))])


def save_candidates(oof, val, test, weights, y_train, y_val, y_test):
    directory = _directory()
    np.savez_compressed(directory / 'candidate_predictions_pld50.npz',
                        oof_candidates=oof, val_candidates=val, test_candidates=test,
                        weights=weights, y_train=y_train, y_val=y_val, y_test=y_test,
                        dynamic_val=val @ weights, dynamic_test=test @ weights)
    with (directory / 'ensemble_weights.json').open('w', encoding='utf-8') as stream:
        json.dump({'model_names': ['DL_mean','Ridge','Bayes','Elastic','GBR','RF','ET','KNN'],
                   'weights': list(map(float, weights)), 'scale': 'pLD50'}, stream, indent=2)


def export_predictions(frame, predictions, label_col, smiles_col, output_dir, split):
    import pandas as pd
    result = pd.DataFrame({'SMILES': frame[smiles_col].astype(str).to_numpy(),
                           'y_true': frame[label_col].to_numpy(dtype=float),
                           'y_pred': np.asarray(predictions, dtype=float)})
    result.to_csv(Path(output_dir) / f'{split}_predictions.csv', index=False, encoding='utf-8-sig')
    manifest = {'scale': 'pLD50', 'n': len(result), 'label_col': label_col,
                'ordered_smiles_sha256': hashlib.sha256(
                    '\n'.join(result['SMILES']).encode('utf-8')).hexdigest()}
    (Path(output_dir) / f'{split}_predictions_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')


def save_classical_baseline(model, name, fold, seed, transformer, args):
    directory = artifact_context(args.output_dir, name, fold)
    if os.environ.get('SAVE_LARGE_ML_MODELS', '0') != '1':
        joblib.dump({'label_transformer': transformer,
                     'training_target': 'Yeo-Johnson, standardize=True',
                     'prediction_scale': 'pLD50', 'model_saved': False},
                    directory / f'seed_{seed}_preprocessor.joblib')
        return
    joblib.dump({'model': model, 'label_transformer': transformer,
                 'training_target': 'Yeo-Johnson, standardize=True',
                 'prediction_scale': 'pLD50'}, directory / f'seed_{seed}.joblib', compress=3)
