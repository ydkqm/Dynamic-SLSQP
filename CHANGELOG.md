# Packaging changes relative to archived 03_ensemble.py

- The neural architecture, feature calculations, training loops, target transforms,
  fold definitions and SLSQP objective are preserved.
  The OOF-MSE candidate threshold was removed at the author's request; all eight candidates are optimized.
- Renamed archived script to main_model.py; removed forced proxy deletion and
  forced offline mode so new users can download ChemBERTa normally.
- Merged the cleaning and split behavior of ry.py and split_data.py into
  prepare_data.py, preserving greedy source order, 2048-bit radius-2 fingerprints,
  strict <0.80 retention, label-cleaning order and the two seed-42 splits.
- Added launchers for paths, input validation, cache provenance, package versions,
  output collision protection and optional serialization of all ML regressors.
- Added prediction using saved networks, preprocessors, regressors and weights.
- No ablations, unrelated experiments, personal data, weights or training data
  have been bundled. No public license is imposed on the user's code.

## Environment documentation update

- Added the user-supplied server dependency snapshot and an explicit provenance table.
- Corrected the broad pandas compatibility bound from <3 to <4 so it does not exclude the supplied pandas 3.0.3 record.
- No model Python source or algorithm was modified by this documentation update.
- The server profile was not installed or full-data trained in this update.
