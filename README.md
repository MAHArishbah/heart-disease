# disease prediction project

`ClassificationPrediction.ipynb` (90 cells), refactored into a package, three
notebooks and a service, following the *Notebook to production - cell by cell*
plan. (Save that PDF as `docs/notebook-refactor-map.pdf` to keep it alongside;
`docs/cell_map.md` reproduces its mapping table.)

The original notebook is untouched and stays where it is.

## Layout

```
params.yaml                every constant that used to be a literal in cell 0 or 23
src/disease_pred/
    config.py              cell 0        paths, params loader, seed, cv
    data.py                cells 1,8,13,23,24
    features.py            cell 28       build_preprocessor
    train.py               cells 37,51,56
    evaluate.py            cells 34,72,76
    serve.py               (new)         FastAPI endpoint
notebooks/
    01-eda.ipynb           cells 2-7, 9-13, 14-22
    02-diagnostics.ipynb   cells 25-27, 29-36, 38-55, 57-61, 63-71, 73-75, 77-82
    03-shap.ipynb          cells 83-88
models/                    model.joblib, feature_spec.json     (generated)
reports/                   metrics.json, model_card.md         (generated + written)
tests/                     71 tests, including the acceptance test
data/raw/heart+disease/    the four UCI processed.*.data files
```

## Run it

```bash
pip install -e ".[notebooks,dev]"        # or just set PYTHONPATH=src

python -m disease_pred.train             # ~13s -> models/model.joblib, feature_spec.json
python -m disease_pred.evaluate          # -> reports/metrics.json
python -m pytest                         # 71 passed
uvicorn disease_pred.serve:app --reload  # http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
       "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3}'
# {"probability":0.6813,"threshold":0.2,"flag":true,"label":"disease"}
```

Notebooks find `src/` on their own - just open and run.

## The sorting rule

For every cell: *does this need to run again when new data arrives?*

- **No** - it ran once to justify a decision -> stays in `notebooks/`
- **Yes, at retraining** -> `src/disease_pred/`
- **Yes, on every prediction** -> inside the fitted `Pipeline`

The notebook already passed the third test. Cell 28 built a
`ColumnTransformer`, cells 36-37 wrapped it with the estimator in a `Pipeline`,
so `model.joblib` takes raw columns and returns a probability. Most of this was
a move, not a rewrite.

**Notebooks import from `src/`, never the reverse.** `tests/test_notebooks.py`
enforces it: re-defining any moved symbol in a notebook fails the suite.

## What changed on the way

**1. A real leakage bug in cell 23.** `high_miss` was computed on the full
frame, before the split at cell 24, so which columns survived was decided
partly by test rows. `data.select_features()` now takes the training frame
only, and `train.py` persists the result to `models/feature_spec.json`. The
dropped list is unchanged in practice (`thal, slope, ca`) - but now that is a
fact rather than a hope.

**2. The threshold is an artifact, not a constant.** `THRESHOLD = 0.2` was
derived from out-of-fold predictions under a 4:1 cost ratio and rests on a
calibration assumption. It is written into `feature_spec.json` and travels with
the model; the cost ratio itself lives in `params.yaml`. Hardcoding 0.2 in
`serve.py` would mean a retrain that shifts the optimal cut silently keeps
serving the old one.

**3. `remainder='passthrough'` -> `remainder='drop'`.** No difference in the
notebook (X was already subset to the selected features), but in a service
passthrough silently appends an unexpected payload column to the design matrix
and breaks coefficient alignment. `serve.py` also reindexes to the saved column
list, so the guard is doubled.

**4. Two designs, one serving path.** `final_model` (sklearn `Pipeline`) is
prediction and goes to production. `Zc_inf` (VA-battery-collapsed, for
statsmodels ORs / p-values / LR tests / Cook's distance) is inference and stays
strictly inside `02-diagnostics.ipynb`. The collapse at cell 31 exists to fix
VIF for interpretable standard errors; regularised prediction does not need it.

Three further changes came out of writing `serve.py` and its tests - each one a
train/serve skew the notebook could not have surfaced, because in a notebook
every row arrives through the same `pd.read_csv`:

**5. `apply_zero_rules` runs at predict time too.** Cell 8's `chol`/`trestbps`
zero -> NaN rule is deterministic, so it was easy to file under "cleaning" and
leave in `data.clean()`. But the model never saw `chol = 0` in training - they
all became NaN - so a payload carrying a literal 0 would be scaled as a real
cholesterol of zero instead of imputed. Split out of `clean()` and called by
`serve.to_frame()`.

**6. Payloads are coerced to float.** An absent JSON field arrives as `None`,
and in a pandas *object* column that is not what `SimpleImputer` looks for. It
sailed past the imputer and reached the encoder as an unknown category -
silently, scoring the patient against a design matrix nobody intended.

**7. `MissingIndicator(error_on_new=False)`.** `features='missing-only'` fits
indicators for columns that had missing values in training; by default a column
that was complete in training but absent in a payload raises. Useful in a
notebook, a 500 in a service for a patient whose age was not recorded.

## SHAP

**Report only**, per the plan. Cells 83-88 live in `notebooks/03-shap.ipynb` as
a dev artifact; `serve.py` returns a probability and a decision, nothing more.

If per-patient explanations are ever wanted from the API, cell 86 already
proves the values are exactly `beta_j * (z_ij - zbar_j)`, so the second
versioned artifact is just the coefficient vector and the training column
means - both already inside the fitted pipeline.

## Verification

Every number the original notebook quotes in its own markdown reproduces
exactly:

| | original markdown | here |
|---|---|---|
| best params | C=0.316 (cell 44) | C=0.316, l1_ratio=0.0 |
| dropped columns | thal, slope, ca | same |
| t_f1 / t_recall / t_cost | 0.43 / 0.405 / 0.2 (cell 53) | 0.431 / 0.405 / 0.200 |
| OOF Brier | 0.1364 (cell 71) | 0.136445 |
| no-skill baseline Brier | 0.247 (cell 55) | 0.247 |
| base_logreg Brier | 0.130 (cell 55) | 0.130 |
| test vs OOF | auc .912/.881, ap .920/.883, brier .118/.136 (cell 73) | identical |
| LOGO AUC range | ~0.70-0.89 (cell 73), swz AP ~0.97 (cell 50) | 0.704-0.889, swz AP 0.961 |
| site mix | swz .152/.128, va .187/.227 (cell 75) | identical |
| pseudo R2 / LLR p | 0.399 / ~2e-70 (cell 58) | 0.399 / 2.11e-70 |
| Cook's + leverage flags | 19/688 (cell 69) | 19 of 688 |
| per-class @ t=0.2 | recall .984 / .476, tn=49 fp=54 fn=2 tp=125 (cell 78) | identical |

**The acceptance test** from the plan - load `model.joblib`, call
`predict_proba()` on raw rows straight from the CSV, get the same numbers the
fitted pipeline gave - is `tests/test_serve.py`, run over all 230 test rows,
not just five.

## Cell coverage

All 90 original cells are accounted for; see `docs/cell_map.md`. Cells 62, 85
and 89 were empty and are deleted (noted in place in the notebooks). Cells 34
and 51 each split across two destinations; everything else moved whole.

CI: every pull request reruns the DVC pipeline, the tests and a metrics diff against main.
