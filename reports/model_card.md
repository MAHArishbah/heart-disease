# Model card - heart disease risk

Cost-weighted screening model over the UCI heart-disease battery (four sites).
Sources: notebook cells 12, 50, 53, 63-67, 78.

| | |
|---|---|
| Task | binary classification, `target = num > 0` |
| Estimator | `LogisticRegression(penalty='elasticnet', solver='saga')` inside a `Pipeline` |
| Selected hyperparameters | `C = 0.316`, `l1_ratio = 0.0` (i.e. ridge; the grid searched L1 too and chose none) |
| Decision threshold | **0.20**, not 0.5 - see *Threshold* below |
| Training rows | 688 | 
| Held-out rows | 230 |
| Training prevalence | 0.554 |
| Artifacts | `models/model.joblib`, `models/feature_spec.json` |

## Intended use

Screening triage: rank patients by risk and flag those who warrant further
workup. It is a coursework/portfolio model built on a 1988 four-site research
dataset of 920 rows. **Not** a clinical device, and not validated for use on
any live patient population.

## Inputs

Ten raw clinical fields: `age, trestbps, chol, thalach, oldpeak` (continuous),
`sex, fbs` (binary), `cp, restecg, exang` (nominal).

`thal`, `slope` and `ca` are **dropped** - each is missing in more than 30% of
training rows. The threshold lives in `params.yaml` (`features.missing_threshold`).

Missing values are expected and handled by the fitted pipeline (median for
continuous, most-frequent for categorical), plus a `MissingIndicator` block on
the numeric and binary columns. `chol = 0` and `trestbps = 0` are treated as
missing, not as measurements - they are physiologically impossible.

## Performance

| metric | out-of-fold (n=688) | test (n=230) |
|---|---|---|
| ROC AUC | 0.881 | 0.912 |
| Average precision | 0.883 | 0.920 |
| Brier | 0.136 | 0.118 |
| Recall @ t=0.20 | 0.969 | 0.984 |
| Precision @ t=0.20 | 0.690 | 0.698 |

**Report the out-of-fold number (~0.88 AUC), not the test number (0.91).** Test
beats OOF on every metric, which is unusual. Site composition was checked and
does not explain it (Switzerland is *over*-represented in test, 15.2% vs 12.8%,
which should make test harder). The likelier explanation is sampling variance
on a smaller single split - nested CV showed AUC std ~0.02 across folds of
similar size. OOF is the lower-variance estimate of the two.

Nested CV log-loss 0.4406 +/- 0.0368 against a single-split 0.4376, so the
tuning estimate is not optimistically biased and is reportable.

## Threshold: 0.20, and why

`0.20 = C_FP / (C_FP + C_FN)` for a stated **4:1** cost ratio - a missed
diagnosis is assumed four times costlier than an unnecessary follow-up. This is
the Bayes-optimal cut under that cost, not the symmetric F1 peak (which was
0.43). A recall target of 0.9 would have given 0.405, but 0.9 was a round
number rather than a cost estimate.

The rule is only meaningful if the probabilities are calibrated, not merely
well-ranked. That was checked twice: OOF Brier 0.136 against a no-skill
baseline of 0.247 (~45% Brier skill score), and reliability diagrams on both
OOF and test that track the diagonal - tightest in the high-probability bins,
which are the ones the decision actually depends on.

The threshold is a **fitted quantity** and is stored in
`models/feature_spec.json`, not hardcoded in `serve.py`. The cost ratio lives
in `params.yaml`; changing it re-triggers training.

### What t=0.20 costs, concretely (test split)

| | precision | recall | support |
|---|---|---|---|
| no disease | 0.961 | **0.476** | 103 |
| disease | 0.698 | **0.984** | 127 |

Confusion matrix: TN=49, FP=54, FN=2, TP=125.

Two of 127 diseased patients are missed; 54 of 103 healthy patients are
flagged. That asymmetry *is* the 4:1 cost ratio, working as specified. The
macro average (0.730) hides it - **quote the per-class numbers**, not the
average, to a non-technical audience.

## Limitations

### Cross-site generalisation is the main one

Leave-one-site-out CV:

| held-out site | AUC | AP |
|---|---|---|
| cleveland | 0.851 | 0.851 |
| hungary | 0.889 | 0.872 |
| switzerland | 0.704 | 0.961 |
| va | 0.711 | 0.860 |

Cleveland and Hungary are in line with the 5-fold nested estimate. Switzerland
and VA drop to ~0.70-0.71, outside the nested-CV std band - a real
generalisation gap, not noise. (Switzerland's AP of 0.961 looks strong but is a
base-rate artifact of 93% prevalence; AUC is the fair read there.)

**So the headline ~0.88 AUC is an in-distribution estimate for a
Cleveland/Hungary-like workup. Expect closer to 0.70-0.75 at a genuinely new
site whose data is collected like Switzerland's or VA's.**

### Missingness is informative, and partly encodes site

Missingness rates and prevalence both vary strongly by site and move together
at Switzerland and VA, so the data is not MCAR. `missingindicator_chol` and
`missingindicator_fbs` carry large coefficients - the model is partly learning
"this field is absent -> probably a site that only enrolled diseased patients".
Three independent signals agree on this:

1. the LOGO gap above;
2. OR confidence intervals - `missingindicator_fbs` has the largest point
   estimate in the table (OR=3.64) but by far the widest CI [1.38, 9.61], a ~7x
   span against ~3x for `sex`/`exang` at a similar point estimate;
3. influence diagnostics - 19 of 688 rows exceed both the Cook's distance and
   leverage cutoffs, and several of them are rows missing several fields at
   once.

Report those two odds ratios with the site-confound caveat attached, never as
standalone clinical findings.

### On the inference numbers

Coefficients, ORs, p-values and LR tests come from a separate statsmodels
design (`Zc_inf`), which collapses the three VA-battery missingness indicators
into one to fix VIF. It exists only in `notebooks/02-diagnostics.ipynb` and
never goes to production; the served model is the regularised sklearn pipeline,
which does not need the collapse.

- HC3 robust SEs: all ratios to model-based SEs fall in 0.944-1.125, no
  coefficient flips significance. The p-values are not an artifact of
  heteroscedasticity or generic misspecification.
- **Cluster-robust SEs by site are reported but should not be believed.** With
  G=4 clusters the sandwich "meat" matrix is built from four score
  contributions; ratios swing 0.18 to 2.22 in both directions with no
  consistent story. Cluster-robust SEs need many clusters to be asymptotically
  valid. LOGO CV is the trustworthy answer to "does this transfer across
  sites"; that SE comparison is a negative result and should not be repeated on
  this dataset.
- No separation detected (no |coef| or SE above 8).
- Pseudo R² = 0.399, LR chi2 = 377.25 on 16 df, p = 2.1e-70.

Clinically interpretable effects worth trusting: `sex` (OR 3.58), `exang`
(OR 3.37), `oldpeak` (OR 1.69/SD), `thalach` (OR 0.69/SD, protective),
`chol` (OR 1.27/SD).

### Other

- 920 rows before deduplication, from 1988. Small, old, and not representative
  of any current population.
- The train/test split is stratified by target only, not by site.
- No fairness analysis beyond site. `sex` is the strongest single predictor and
  the cohort is male-skewed; subgroup performance by sex has not been measured.

## Ethical considerations

At t=0.20 roughly half of healthy patients are flagged. That is deliberate, but
it is a real cost borne by real people - anxiety and unnecessary follow-up
procedures. The 4:1 ratio is an assumption written in `params.yaml`, not a
measured clinical fact, and anyone deploying this should replace it with their
own.

## Reproducing

```
python -m disease_pred.train      # -> models/, reports/train_metrics.json
python -m disease_pred.evaluate   # -> reports/metrics.json
```

Seed 42 throughout. Every number above came from those two commands and
`notebooks/02-diagnostics.ipynb`.
