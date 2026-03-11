# Model Artifacts

This directory contains curated release checkpoints for `PolyGraphMT`.

Only one seed is retained for each model group. For each group, the smallest available checkpoint was kept to reduce upload size while preserving one working artifact pair.

## Layout

```text
models/
├── best_model_map.json
├── multitask_models/
└── single_models/
```

## File Naming

Single-task artifacts:

- `<property>_single_model_<seed>.pt`
- `<property>_single_scalar_<seed>.pt`

Multitask artifacts:

- `<family>_model_<seed>.pt`
- `<family>_scalar_<seed>.pt`

`*_model_*.pt` stores the trained checkpoint.

`*_scalar_*.pt` stores the target-scaling metadata needed to invert predictions back to original units.

## Model Selection Map

`best_model_map.json` defines which released model family should be used for each property.

Preserved multitask mappings:

- `cp -> multitask:g2`
- `poisson -> multitask:mechanical`
- `alpha -> multitask:g1`
- `rho -> multitask:g2`

All other mapped properties use single-task checkpoints.

## Recommended Usage

Use the model-selection wrapper:

```bash
python3 scripts/predict_from_models.py \
  --property cp \
  --input_csv my_smiles.csv \
  --fidelity exp \
  --out_csv predictions_cp.csv
```

This wrapper:

- reads `best_model_map.json`
- resolves the correct checkpoint/scaler pair from `single_models/` or `multitask_models/`
- reuses the repository inference code
- writes a property-level prediction CSV by default

By default, the wrapper keeps only the requested property output. For properties backed by multitask models, use `--keep_all_outputs` to retain all predictions from the resolved multitask family.

Example:

```bash
python3 scripts/predict_from_models.py \
  --property cp \
  --input_csv my_smiles.csv \
  --fidelity exp \
  --keep_all_outputs \
  --out_csv predictions_g2_full.csv
```

## Direct Usage

The base predictor can also load a specific artifact pair directly:

```bash
python3 scripts/predict.py \
  --ckpt_path models/single_models/tg_single_model_42.pt \
  --scaler_path models/single_models/tg_single_scalar_42.pt \
  --input_csv my_smiles.csv \
  --fidelity exp \
  --out_csv predictions_tg.csv
```

This mode is useful when selecting a checkpoint manually rather than going through `best_model_map.json`.

## Notes

- Property names are lowercase keys such as `cp`, `rho`, `alpha`, `tg`, and `poisson`.
- `--fidelity` should match one of the fidelities used during training for the selected checkpoint.
- These files are curated release artifacts, not the complete training archive of all seeds and all intermediate runs.
