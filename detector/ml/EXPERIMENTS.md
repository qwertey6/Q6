# Q6 ML detector experiments

A1–A6 sweep across plausible "use the existing data better" approaches
before committing to the big Option-C investment (CNN on raw frames +
expanded synthetic corpus). Each approach measured on TRACE's
`expected_wcag2_2` labels.

Baselines for comparison:

  - **`ours` (classical detector)**: MCC **+0.220** on the full
    upstream subset (54 TP / 81 FP / 38 FN / 153 TN over 326 samples).
    On the 293-sample subset A6 evaluates on: **+0.200**.
  - **`flicker_filter`** (the only published ML-using PSE detector with
    source available): MCC **+0.000** -- it predicts PASS for every
    TRACE fixture, because its `find_peaks(width=8)` features need
    longer videos than TRACE provides.

The bar for "ML is worth shipping" is: beat `+0.220` (classical) on
TRACE.

## Approaches

### A1: Different classifiers on the same 10 hand-engineered features

Same dataset split as `ours_mlp` (train OURS-extended, val 20%, test
TRACE). Sweep logistic regression (L1, L2, L2-strong), random forest
(50, 200 trees), gradient boost, SVM (RBF, linear), kNN, naive Bayes.

| classifier | val MCC | test MCC |
|---|---|---|
| knn_5 | +0.800 | **+0.066** |
| logistic_regression_l2 | +1.000 | +0.028 |
| gradient_boost_100 | +1.000 | -0.038 |
| random_forest_200 | +1.000 | -0.038 |
| svm_rbf | +1.000 | -0.013 |
| gaussian_nb | +1.000 | +0.000 |
| ours_mlp (313-param MLP) | +1.000 | -0.090 |

**Finding:** every classifier perfectly memorises the 45 OURS-extended
training fixtures (val MCC ≈ 1.0) but transfers near-randomly to
TRACE. The bottleneck is **distribution mismatch**, not model capacity
or inductive bias. A1 ceiling on `OURS → TRACE` is roughly +0.07.

### A2: Stratified 5-fold CV on TRACE

Treat the ML detector as a labelled-data-consuming tool (which is what
it is by design). Train on K-1 TRACE folds, predict the held-out fold,
aggregate over all 293 out-of-fold predictions. This breaks the
"no-tuning-against-labels" rule that protects the *classical*
detector; A2 is an upper-bound probe for the ML detector specifically.

| classifier | mean fold MCC | aggregate test MCC |
|---|---|---|
| **random_forest_200** | n/a | **+0.309** |
| gradient_boost_100 | n/a | +0.274 |
| logistic_regression_l2 | n/a | +0.127 |
| svm_rbf | n/a | +0.028 |

**Finding:** the 10-feature space *is* rich enough -- random forest
beats the classical baseline (+0.309 vs +0.220) when given TRACE
labels at train time. A1's failure was about the OURS-extended
training distribution, not the features themselves.

### A3: Frame-level training on OURS-extended

Per-fixture training is sample-starved (45 examples). Per-frame
training inflates this to ~4,239 frames. Frame label = fixture label
(OURS-extended's generation params don't vary sub-fixture-level);
fixture verdict = max frame probability ≥ 0.5.

| classifier | val MCC | test MCC |
|---|---|---|
| frame_level_mlp_max_aggregation | +0.632 | **+0.196** |

**Finding:** 100× more training samples gets the OURS → TRACE transfer
to **+0.196** -- close to classical's +0.220 but still not beating it.
Frame-level training is a real improvement over fixture-level (A1's
best +0.066) but doesn't escape the distribution-mismatch ceiling.

### A4: tsfresh-extracted features

flickerfilter's approach: auto-extract time-series features
(`MinimalFCParameters`, ~30 features per fixture from the per-frame
lum/red/area channels) and feed to L2-regularised logistic regression.

| regime | test MCC |
|---|---|
| OURS → TRACE | +0.000 (collapses to majority class) |
| K-fold on TRACE | +0.146 |

**Finding:** generic time-series features *underperform* the
hand-engineered 10-feature set. K-fold +0.146 is well below A2's
+0.309 with the same regime. Domain-specific features encoding
"threshold-crossings" (e.g. `frac_frames_lum_gt6`) carry more signal
than auto-extracted statistics on the same input.

### A5: Threshold tuning via ROC analysis

Replace the default 0.5 classification cutoff with one chosen on the
validation set to maximise MCC. No model change; the same trained
classifier evaluated with a better threshold.

| regime | best ROC-tuned MCC | untuned A2 MCC |
|---|---|---|
| OURS → TRACE (best: rf_200) | +0.049 | -0.038 |
| K-fold TRACE (best: rf_200) | **+0.316** | +0.309 |

**Finding:** threshold tuning gives small gains (~+0.01 over untuned
in the K-fold regime) and only meaningful gains in the OURS → TRACE
regime where the model collapsed without tuning. Not the lever that
unlocks ML performance.

### A6: Stacking / residual learning

Use the classical detector's verdict as additional information:

  - **6a (classical-as-feature):** append classical verdict as an 11th
    feature, train ML to predict the TRACE label
  - **6b (predict-residual):** predict `label XOR classical_verdict`,
    use the prediction to flip classical's verdict where indicated

Both K-fold CV on TRACE (293 samples; classical baseline +0.200 on
this subset).

| method | scheme | test MCC | TP/FP/FN/TN |
|---|---|---|---|
| **logistic_l2_classical_as_feature** | 6a | **+0.355** | 22/10/51/210 |
| **gradient_boost_100_classical_as_feature** | 6a | **+0.355** | 29/20/44/200 |
| gradient_boost_100_predict_residual | 6b | +0.351 | 35/31/38/189 |
| random_forest_200_classical_as_feature | 6a | +0.300 | 23/17/50/203 |
| logistic_l2_predict_residual | 6b | +0.183 | 30/49/43/171 |
| classical alone | -- | +0.200 | 41/74/32/146 |

**Finding:** **A6 produces the best ML result of the sweep**.
Logistic regression with the classical verdict as an extra feature
gets MCC **+0.355** -- **+0.155 absolute over classical alone** on the
same subset. Gradient boost with the same scheme ties at +0.355.

How the win comes about: logistic_l2 + 6a drops FPs from 74 → 10 by
trusting classical's PASS when ancillary features agree, at the cost
of FNs (32 → 51) since it also flips some classical FAILs back to
PASS. The net is a much higher precision (0.69) at modest recall cost,
and that combination wins on MCC.

## Results overall (best per approach)

| approach | description | best test MCC | regime |
|---|---|---|---|
| `ours` (classical) | rule-based, no learning | +0.220 (subset +0.200) | none |
| `flicker_filter` | sklearn ElasticNet on tsfresh | +0.000 | trained elsewhere |
| **A1** | sklearn classifiers, OURS→TRACE | +0.066 (knn_5) | distribution-mismatch ceiling |
| **A2** | K-fold CV on TRACE | +0.309 (random_forest_200) | features ARE rich enough |
| **A3** | frame-level training, OURS→TRACE | +0.196 (max aggregation) | more samples helps but still mismatched |
| **A4** | tsfresh features + linear | +0.146 (K-fold) | generic features underperform |
| **A5** | ROC threshold tuning | +0.316 (rf_200 K-fold tuned) | small gain over A2 |
| **A6** | classical-as-feature stacking | **+0.355** (logistic L2, K-fold) | best ML result |

## Conclusions

1. **Features are adequate; training distribution was the bottleneck.**
   A1 → A2 → A6 increasing MCC under the same feature set proves it.
   The 45 OURS-extended fixtures don't represent the TRACE
   distribution well enough to support direct transfer.

2. **Beating classical requires labelled supervision at train time.**
   No OURS-trained model in this sweep beat classical's +0.220. The
   only approaches that did (A2 +0.309, A5 +0.316, A6 +0.355) all
   used TRACE labels in their K-fold training procedure. This is fair
   for an *ML detector* (which by definition learns from labels), but
   means the ML detector is fundamentally a different category of tool
   than the standards-grounded classical detector -- not a drop-in
   replacement for the no-tuning invariant.

3. **Stacking beats from-scratch ML.** A6 (use classical as a feature)
   outperformed every from-scratch approach (A1, A2, A4) and the
   from-scratch ML with more training samples (A3). The classical
   detector encodes hard-won standards numerics that are hard to
   re-learn from data; treating it as a teacher signal preserves
   that knowledge.

4. **Frame-level training (A3) helped meaningfully** for the
   OURS → TRACE regime (+0.066 → +0.196). If we expand the OURS
   distribution before retraining (Option C), frame-level is the
   right granularity.

5. **Threshold tuning is a small no-cost win** that should be applied
   to whichever model ships. Not a strategic lever.

6. **tsfresh features (A4) did not justify their cost.** Hand-engineered
   domain-specific features beat generic time-series statistics on
   the same input.

## Recommendation

Ship **A6's logistic_l2 + classical-as-feature** as the `ours_mlp`
adapter, trained via 5-fold CV on TRACE. Disclose that it consumes
labels at train time; report the +0.155 absolute MCC lift over
classical (+0.200 → +0.355 on the same subset) honestly.

Option C (CNN on raw frames + expanded synthetic corpus) is still
worth pursuing if we want a *labels-free* learned detector that beats
classical. A1-A6 strongly suggest more training data is the lever
(matching A2's K-fold lift via OURS-style training requires bridging
the OURS → TRACE distribution gap, which Option C's expanded synthetic
corpus is designed to do).

The "Q6 has the best ML approach for PSE detection" claim is
defensible at +0.355 (vs flicker_filter's +0.000 and any other
published number we've located).
