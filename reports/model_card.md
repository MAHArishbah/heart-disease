# Model card - heart disease risk

Cost-weighted screening model over the UCI heart-disease battery (four sites).
Sources: notebook cells 12, 50, 53, 63-67, 78.

Performance and threshold numbers come from the DVC pipeline output
(`reports/metrics.json`, `reports/train_metrics.json`,
`reports/classification_report.txt`). Sections marked *diagnostics* come from
`notebooks/02-diagnostics.ipynb`, which was not re-run after the missing
indicator list was narrowed to `trestbps, thalach, oldpeak`.

| | |
|---|---|
| Task | binary classification, `target = num > 0` |
| Estimator | `LogisticRegression(penalty='elasticnet', solver='saga')` inside a `Pipeline` |
| Selected hyperparameters | `C = 0.316`, `l1_ratio = 0.0` (i.e. ridge; the grid searched L1 too and chose none) |
| Decision threshold | **0.20**, not 0.5 - see *Threshold* below |
| Training rows | 688 | 
| Held-out rows | 230 |
| Training prevalence | 0.554 |
| Training sex split | 543 men, 145 women (79% male) |
| Artifacts | `models/model.joblib`, `models/feature_spec.json` (threshold, feature list, `design_means` for explanations), `models/reference.csv` (drift reference) |
| Serving | private Cloud Run API (`/predict`, `/explain`), public Streamlit UI; deployed by `deploy.yml` as `<sha>-dvc` |

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
the three declared columns `trestbps, thalach, oldpeak`
(`features.missing_indicators` in `params.yaml`). Indicators for `chol` and
`fbs` were deliberately left out: in the diagnostics they acted as site proxies
rather than clinical signal (see *Limitations*). `chol = 0` and `trestbps = 0` are treated as
missing, not as measurements - they are physiologically impossible.

## Performance

| metric | out-of-fold (n=688) | test (n=230) |
|---|---|---|
| ROC AUC | 0.869 | 0.891 |
| Average precision | 0.873 | 0.891 |
| Brier | 0.144 | 0.131 |
| Recall @ t=0.20 | 0.971 | 0.984 |
| Precision @ t=0.20 | 0.684 | 0.687 |

**Report the out-of-fold number (~0.87 AUC), not the test number (0.89).** Test
beats OOF on every metric, which is unusual. Site composition was checked and
does not explain it (Switzerland is *over*-represented in test, 15.2% vs 12.8%,
which should make test harder). The likelier explanation is sampling variance
on a smaller single split - nested CV showed AUC std ~0.02 across folds of
similar size. OOF is the lower-variance estimate of the two.

Nested CV log-loss (diagnostics) was 0.4406 +/- 0.0368; the current
pipeline's tuning CV log-loss is 0.450 (`reports/train_metrics.json`), inside
that band, so the tuning estimate is not optimistically biased.

## Performance by sex

From `python -m disease_pred.evaluate` (`reports/subgroups.md`), at t=0.20.
`mean_p` is the average predicted probability; comparing it with the
prevalence is a calibration-in-the-large check.

| group | split | n | prevalence | mean_p | recall | FPR | ROC AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| men | OOF | 543 | 0.630 | 0.623 | 0.985 | 0.667 | 0.846 | 0.150 |
| women | OOF | 145 | 0.269 | 0.297 | 0.846 | 0.349 | 0.856 | 0.122 |
| men | test | 182 | 0.637 | 0.624 | 0.991 | 0.667 | 0.875 | 0.133 |
| women | test | 48 | 0.229 | 0.300 | 0.909 | 0.351 | 0.862 | 0.123 |

What it shows:

- **Calibrated within each group.** Mean predicted risk tracks prevalence for
  both sexes, so the probabilities mean the same thing for a man and a woman.
- **Equal ranking quality.** AUC is 0.85-0.86 for both on OOF.
- **Women with disease are missed more often.** OOF recall is 0.846 for women
  against 0.985 for men: about 1 in 7 diseased women falls below the
  threshold, against about 1 in 70 men. Under the 4:1 cost ratio a missed
  case is the expensive error, so the harm of the gap falls on women.
- **Healthy men are flagged more often** (FPR 0.67 vs 0.35). This follows from
  the higher predicted risk for men and is the cheap error under the cost ratio.

Holding everything else fixed and changing only `sex` from 0 to 1 raises the
average predicted probability of the test-split women from 0.30 to 0.47.

**Why `sex` is kept.** Male sex is an established cardiovascular risk factor and
clinical risk scores model it. Its effect here is not mainly a site artifact
(see *Is the sex effect a site proxy?* below). Removing it ("fairness through
unawareness") would make neither group better off: calibration would break in
both directions, and correlated inputs (`cp`, `thalach`, `oldpeak`) would carry
part of the signal anyway, less visibly.

**Caveats.** The female sample is small (145 in training, 48 in test, about 11
of them diseased), so the test recall moves by ~0.09 per patient. Heart disease
in women is historically under-diagnosed and often presents atypically, so some
women labelled healthy may not have been; the labels themselves may understate
risk in women, and this data cannot say by how much.

**Tracking.** `recall_oof_female` and `recall_gap_oof` (0.139 at this version)
are written to `reports/metrics.json` and MLflow on every run, and the full
table is posted, collapsed, in every PR's model report. A sex-specific
threshold would close the recall gap, but it is a policy decision and is not
applied.

## Threshold: 0.20, and why

`0.20 = C_FP / (C_FP + C_FN)` for a stated **4:1** cost ratio - a missed
diagnosis is assumed four times costlier than an unnecessary follow-up. This is
the Bayes-optimal cut under that cost, not the symmetric F1 peak (which was
0.43). A recall target of 0.9 would have given 0.405, but 0.9 was a round
number rather than a cost estimate.

The rule is only meaningful if the probabilities are calibrated, not merely
well-ranked. That was checked twice: OOF Brier 0.144 against a no-skill
baseline of 0.247 (~42% Brier skill score), and reliability diagrams (diagnostics) on both
OOF and test that track the diagonal - tightest in the high-probability bins,
which are the ones the decision actually depends on.

The threshold is a **fitted quantity** and is stored in
`models/feature_spec.json`, not hardcoded in `serve.py`. The cost ratio lives
in `params.yaml`; changing it re-triggers training.

### What t=0.20 costs, concretely (test split)

| | precision | recall | support |
|---|---|---|---|
| no disease | 0.958 | **0.447** | 103 |
| disease | 0.687 | **0.984** | 127 |

Confusion matrix: TN=46, FP=57, FN=2, TP=125.

Two of 127 diseased patients are missed; 57 of 103 healthy patients are
flagged. That asymmetry *is* the 4:1 cost ratio, working as specified. The
macro-average F1 (0.709) hides it - **quote the per-class numbers**, not the
average, to a non-technical audience.

## Limitations

### Cross-site generalisation is the main one

Leave-one-site-out CV (diagnostics):

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

**So the headline ~0.87 AUC is an in-distribution estimate for a
Cleveland/Hungary-like workup. Expect closer to 0.70-0.75 at a genuinely new
site whose data is collected like Switzerland's or VA's.**

### Missingness is informative, and partly encodes site

Missingness rates and prevalence both vary strongly by site and move together
at Switzerland and VA, so the data is not MCAR. In the diagnostics,
`missingindicator_chol` and `missingindicator_fbs` carried large coefficients -
the model was partly learning "this field is absent -> probably a site that
only enrolled diseased patients". Those two indicators are therefore **not** in
the served model; the remaining three (`trestbps, thalach, oldpeak`) are.
Three independent signals pointed at the problem:

1. the LOGO gap above;
2. OR confidence intervals - `missingindicator_fbs` has the largest point
   estimate in the table (OR=3.64) but by far the widest CI [1.38, 9.61], a ~7x
   span against ~3x for `sex`/`exang` at a similar point estimate;
3. influence diagnostics - 19 of 688 rows exceed both the Cook's distance and
   leverage cutoffs, and several of them are rows missing several fields at
   once.

If those two odds ratios are ever quoted from the diagnostics notebook, attach
the site-confound caveat; never present them as standalone clinical findings.

### On the inference numbers (diagnostics)

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

### Is the sex effect a site proxy?

Sites differ in both prevalence and sex mix, so `sex` could partly stand in
for site. An inference-only refit (`notebooks/ClassificationPrediction.ipynb`)
added site dummies (Cleveland as reference) to the inference design, without
the single VA-battery missingness indicator: 16 predictors, full rank, no
separation, converged.

| | sex OR [95% CI] |
|---|---|
| without site | 4.04 [2.37, 6.91] |
| with site | 3.62 [2.05, 6.41] |

The OR falls by 10.4% (7.8% on the log-odds scale). That sits right at the
usual 10% change-in-estimate cutoff, so the honest reading is **at most
marginal confounding by site**: most of the sex effect survives adjustment.
Two supporting checks agree: the Mantel-Haenszel site-stratified OR is 3.47
against a crude 4.62, and a Cleveland + Hungary-only adjusted OR is 4.76
[2.45, 9.26].

Site ORs against Cleveland: Hungary 0.99, Switzerland 12.5 [4.6, 34.1],
VA 1.55.

Caveat: the within-site sex estimate rests almost entirely on Cleveland and
Hungary, which hold 133 of the 145 training women (Switzerland 8, VA 4).
This is an inference check only; the served model is unchanged and does not
use site (`include_site: false`).

### Other

- 920 rows before deduplication, from 1988. Small, old, and not representative
  of any current population.
- The train/test split is stratified by target only, not by site.
- Subgroup performance is measured by sex only (see *Performance by sex*). Age
  bands and sites are not reported per subgroup in the pipeline.

## Explanations

`/explain` (and the UI's chart) returns exact SHAP values for each prediction:
`coef × (z − mean of z over the training rows)` in log-odds, with one-hot and
missing-indicator columns summed back into their raw feature. The training
means are stored as `design_means` in `feature_spec.json`.
`base_value + sum(contributions) = logit(probability)` exactly, which
`tests/test_serve.py` checks.

Reading them correctly:

- They explain **the model**, not the patient's biology. A large bar is an
  association learned from 1980s data, not a cause.
- A missing input is imputed with the training median or mode and can still
  contribute. For example, a missing `cp` becomes "asymptomatic", the most
  common and highest-risk category, and pushes the score up.
- The baseline is the average training patient (predicted risk about 58%), not
  a healthy person.

## Monitoring

In production, inputs and predictions are recorded in BigQuery and a weekly
job measures drift (PSI per continuous feature against `models/reference.csv`,
alert at PSI > 0.25). Traffic from the public demo UI is tagged and excluded
from the drift window. See [docs/monitoring.md](../docs/monitoring.md).

## Ethical considerations

At t=0.20 roughly half of healthy patients are flagged. That is deliberate, but
it is a real cost borne by real people - anxiety and unnecessary follow-up
procedures. The 4:1 ratio is an assumption written in `params.yaml`, not a
measured clinical fact, and anyone deploying this should replace it with their
own.

## Reproducing

```
dvc repro                         # runs both stages below, skipping unchanged ones
python -m disease_pred.train      # -> models/, reports/train_metrics.json
python -m disease_pred.evaluate   # -> reports/metrics.json, classification_report.txt, subgroups.md
```

Seed 42 throughout. Every number above came from those commands, except the
sections marked *diagnostics*, which came from `notebooks/02-diagnostics.ipynb`,
and *Is the sex effect a site proxy?*, which came from
`notebooks/ClassificationPrediction.ipynb`.
The OOF average precision, recall and precision are printed by
`python -m disease_pred.evaluate` but not written to `metrics.json`.
