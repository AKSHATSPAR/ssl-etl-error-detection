# Results

Test split: the last 25% of the timeline, 2000 runs, 291 of them carrying an
incident. Nothing was tuned on test.

Reproduce with:

```bash
PYTHONPATH=src python -m etlguard.eval.run_eval --ckpt runs/ssl/best.pt --save-scores
PYTHONPATH=src python scripts/results_report.py
PYTHONPATH=src python scripts/attribution_study.py --ckpt runs/ssl/best.pt
PYTHONPATH=src python scripts/error_analysis.py --ckpt runs/ssl/best.pt
```

> **A correction to an earlier draft.** Chunk 2 reported that ancestor
> conditioning was worth 18.5 Hit@1 points over deviation magnitude. That
> comparison was against an *uncalibrated* deviation score. Once both scores are
> put on the same footing the advantage is 2.4 points, and the confidence
> intervals overlap. Section 3 explains what happened and why the corrected
> version is the more interesting result.

---

## 1. Main comparison

Every method's per-node scores are percentile-calibrated against its own
unlabelled training runs, so the comparison is like for like.

| Method | AUPRC [95% CI] | Best F1 | F1 @15% | Hit@1 [95% CI] | Hit@3 | MRR |
|---|---|---|---|---|---|---|
| LSTM autoencoder | 0.951 [0.932, 0.968] | 0.928 | 0.887 | **0.818** [0.773, 0.859] | **0.986** | **0.900** |
| PCA residual | 0.942 [0.920, 0.963] | 0.928 | 0.876 | 0.801 [0.756, 0.849] | 0.955 | 0.880 |
| **SSL transformer (ours)** | 0.953 [0.934, 0.971] | 0.935 | 0.890 | 0.797 [0.749, 0.842] | 0.973 | 0.883 |
| Local Outlier Factor | **0.973** [0.958, 0.987] | **0.963** | **0.931** | 0.773 [0.725, 0.821] | 0.976 | 0.872 |
| Dense autoencoder | 0.956 [0.937, 0.973] | 0.910 | 0.893 | 0.749 [0.701, 0.797] | 0.924 | 0.843 |
| Rolling z-score | 0.502 [0.455, 0.554] | 0.519 | 0.399 | 0.570 [0.515, 0.629] | 0.942 | 0.748 |
| Isolation Forest | 0.466 [0.413, 0.527] | 0.469 | 0.450 | 0.340 [0.285, 0.395] | 0.701 | 0.550 |

Intervals are 95% bootstrap. In the localization column every interval among
the top four overlaps every other, which is the formal version of "these are tied".


**The self-supervised model does not win, on either task.** LOF has the best
detection and the LSTM autoencoder has the best localization. With 291 incidents
the standard error on a Hit@1 near 0.8 is 0.023, so the top four localizers
(0.818, 0.801, 0.797, 0.773) sit inside roughly two standard errors of one
another and should be read as tied rather than ranked.

That is worth stating plainly rather than tuning away. Each task profile is only
38 dimensions and the healthy manifold is densely covered by 4800 training runs,
which is close to the ideal setting for a density or subspace method. There was
not much headroom for a deep model to claim: five of seven methods sit between
0.94 and 0.97 AUPRC.

Two baselines do fail, and both failures are informative. Rolling z-score, the
stand-in for a Great Expectations or Deequ style rule tool, collapses on
detection at 0.502 AUPRC because the benign distractors defeat it. Seasonality,
growth and campaigns move features as hard as several real fault types do, which
is exactly why those distractors are in the benchmark. Isolation Forest is
weakest overall, a known consequence of axis-parallel splits on correlated
features.

---

## 2. What actually mattered: making scores comparable across tasks

Ranking nine raw errors asks the wrong question. It asks which task has the
largest error, when the question is which task is furthest from *its own*
normal. Tasks are not equally predictable, so their raw errors are not
comparable, and whichever task happens to be hardest to predict wins the argmax
by default.

There are two ways to fix that, and the interesting result is that they are worth
the same.

| SSL localizer variant | Hit@1 [95% CI] | MRR [95% CI] |
|---|---|---|
| Deviation magnitude, raw | 0.588 [0.533, 0.646] | 0.762 [0.729, 0.796] |
| Deviation magnitude, **statistically** calibrated | 0.773 [0.725, 0.821] | 0.872 [0.843, 0.899] |
| Ancestor-conditioned, raw (**structural**) | 0.773 [0.725, 0.821] | 0.867 [0.838, 0.896] |
| Ancestor-conditioned **and** calibrated | 0.797 [0.749, 0.842] | 0.883 [0.855, 0.910] |

Rows two and three are identical to three decimal places, with the same interval.

**Structural conditioning** achieves comparability by predicting each task from
its parents, so whatever its inputs already explain is subtracted away.
**Statistical calibration** achieves it by converting each task's error into a
percentile against its own healthy history. Different mechanisms, the same
effect, the same number. Doing both adds 2.4 points, which the intervals cannot
separate from noise.

The honest summary is that the gain attributed to the ancestor-conditioned score
in chunk 2 is real but is not unique to it. It is a gain from cross-task
comparability, and a much simpler statistical trick buys the same thing.

### Calibration is not free, and not always positive

| method | uncalibrated Hit@1 | calibrated | delta |
|---|---|---|---|
| LSTM autoencoder | 0.643 | 0.818 | **+0.175** |
| Dense autoencoder | 0.725 | 0.749 | +0.024 |
| SSL (ancestor) | 0.773 | 0.797 | +0.024 |
| PCA residual | 0.801 | 0.801 | 0.000 |
| LOF | 0.797 | 0.773 | -0.024 |
| Rolling z-score | 0.663 | 0.570 | **-0.093** |

It transforms the LSTM autoencoder and it damages the rolling z-score. The
pattern follows how far apart a method's per-node error scales already are: the
LSTM reconstructs all nine tasks through one shared hidden state, so its
per-task error scales differ wildly and calibration has a great deal to correct.
LOF and PCA already produce roughly comparable per-node scores, so there is
nothing left to gain and the rank transform only discards magnitude information.

The practical lesson is that per-node calibration should be checked, not
assumed, and it is cheap to check because it needs no labels.

### Percentile rank, not z-score

An earlier attempt calibrated with medians and IQRs and made things worse
(ancestor-conditioned Hit@1 fell from 0.773 to 0.605). The error distributions
are heavy-tailed and an IQR is a poor summary of them. Percentile rank is
distribution-free and does the same job without assuming a shape the data does
not have.

---

## 3. The depth breakdown, and a bug it caught

The first implementation of the ancestor score hid only the node under test.
That is not surprise given inputs, because structural attention is bidirectional
and the node's own *descendants* remain visible. During an incident those
descendants are corrupted too, so they hand the corrupted value straight back.

The prediction was that the leak should be worst where a node has the most
descendants, meaning upstream, and should vanish at the sink. It does.

| origin depth | ancestor (descendants hidden) | self-masked only (leaky) | deviation magnitude | n |
|---|---|---|---|---|
| 0 (source tasks) | 0.598 | 0.546 | 0.505 | 97 |
| 1 | 0.806 | 0.583 | 0.556 | 72 |
| 2 | 0.821 | 0.643 | 0.643 | 28 |
| 3 | 0.889 | 0.889 | 0.667 | 18 |
| 4 | 0.938 | 0.625 | 0.562 | 48 |
| 5 (sink) | 0.893 | 0.893 | 0.893 | 28 |

At depth 5 all three rules agree exactly, because `load_warehouse` has no
descendants and the three masks are then the same tensor. That is not a tuned
result, it falls out of the construction, and it is the cleanest available
evidence that the mechanism is the one claimed.

Depth 0 is the hardest row and gains least, which also follows: source tasks have
no ancestors either, so conditioning has nothing extra to offer.

---

## 4. Where localization fails, and why

Nine of fifteen fault types localize at or near 1.00. Two fail badly.

| fault type | Hit@1 | mean nodes affected |
|---|---|---|
| `row_loss_truncation` | 0.29 | 3.77 |
| `duplicate_records` | 0.43 | 2.84 |
| (nine types at 0.94 to 1.00) | | 1.00 to 1.76 |

The correlation between how far a fault spreads and how well it localizes is
**-0.63**. Faults that stay local are easy and faults that travel are hard, which
is the expected shape of the problem.

The error pattern is sharper than the aggregate suggests. Of the mislocalized
`duplicate_records` incidents, **100%** blamed a task further downstream, and
nearly all blamed the same one, `join_payments`. For `row_loss_truncation` it is
85%.

The mechanism is specific to joins. A join's output row count depends on key
overlap between its inputs, and key overlap is not in the feature set at all, so
no model can predict a join's row count from its parents' profiles. When a
duplication arrives from upstream, the fan-out it causes at the join is genuinely
unpredictable from the parent profiles, so the join looks like the origin.

This is a benchmark and feature-design limitation rather than a modelling one,
and it names its own fix: add a join-key overlap statistic to the profile. The
information the model needs is currently not measured.

Localization degrades gracefully with severity, from 0.66 Hit@1 in the lowest
quartile to 0.88 in the highest, which is what should happen. A method that
scored faint incidents as well as severe ones would be suspicious.

---

## 5. Explainability

Attribution is scored, not just plotted, because the simulator knows which
symptom family each fault type should move.

| method | accuracy over 291 incidents |
|---|---|
| Integrated Gradients | **0.935** |
| Reconstruction error | 0.924 |

Both are high and the gap is small. Integrated Gradients is clearly better on
one fault type, `timezone_shift` (0.95 against 0.76), which fits its extra
capability: it can credit evidence in other tasks and earlier runs, while
reconstruction error can only look at the flagged task at the flagged timestep. A
timestamp shift shows up as a relationship across the window rather than as one
large residual.

Nine of fifteen fault types reach 1.00 under both methods. The hardest are
`value_corruption` (0.73 and 0.82) and `row_loss_truncation` (0.75 for both).

Given the gap is small and Integrated Gradients costs 32 extra forward and
backward passes per explanation, reconstruction error is the sensible default and
Integrated Gradients is worth reaching for on the fault families where it
demonstrably helps.

### The attention learned the pipeline

Averaged over incidents, structural attention places **46.4%** of its
off-diagonal mass on true DAG edges, against **22.2%** expected by chance, a
**2.1x** lift. The brightest cells are the genuine parent-child pairs:
`extract_customers` with `clean_customers`, `extract_orders` with `clean_orders`,
`aggregate_daily` with `load_warehouse`.

Nothing forces this. The DAG bias is a learned scalar per relation type per head,
not a mask, so the model was free to ignore the graph and did not.

---

## 6. Threats to validity

**The benchmark is synthetic.** Fault severities, frequencies and the type mix
are design choices. Relative ordering is more trustworthy than any absolute
number here.

**Detection may be too easy to discriminate.** Five of seven methods land within
0.03 AUPRC. The benchmark separates methods on localization but barely on
detection. Lower severities, more concurrent incidents and shorter windows would
all make it harder.

**Alert-budget columns saturate.** F1 at a 1% budget is capped at 0.129 and at 5%
at 0.512 for any detector with perfect precision, because you cannot recall more
incidents than you may raise alerts. Only the 15% column, roughly one alert per
incident, discriminates.

**Model selection.** Checkpoints are chosen on validation reconstruction loss,
which uses no labels. The detection pooling rule is chosen on validation AUPRC,
one categorical choice among five. That is not zero label use and is reported
rather than hidden.

**A unit bug that no result depended on, but one interpretation did.** Exported
timestamps were 1000x too small because `DatetimeIndex.astype("int64")` changed
meaning between pandas major versions. Nothing crashed and no metric moved: the
profile tensor and every label are bit-identical across the two dataset versions,
and no baseline uses the clock. What it did break was the calendar-feature
ablation, which was measuring the removal of features that encoded nothing. The
experiment was redone on corrected data and the conclusion held. It is recorded
here because the failure mode is worth remembering: a derived quantity nobody
inspects is a quantity nobody checks, and it was caught only when a demo script
printed dates in 1970.

**Single seed for the headline numbers.** Seed repeats and ablations are reported
in Section 7. Until those land, no gap under about 0.05 Hit@1 should be treated
as real.

---

## 7. Ablations and seed repeats

Identical budget for every configuration, 40 epochs with the same early-stopping
patience of 8, so that "this component helps" cannot be confused with "this run
trained longer". One seed stopped at 33 epochs on that rule, which neither helps
nor hurts it because checkpoints are selected on best validation loss rather than
the final epoch. Three seeds set the noise floor: **AUPRC 0.951 ± 0.001, Hit@1 0.805 ± 0.011, MRR 0.887 ± 0.007**.

| Configuration | val recon | AUPRC | Hit@1 | MRR | Delta Hit@1 | Verdict |
|---|---|---|---|---|---|---|
| Full model (3 seeds) | 0.438 | 0.951 | 0.805 | 0.887 | reference | |
| Clean training set | 0.436 | 0.952 | 0.814 | 0.892 | +0.009 | within noise |
| No DAG attention bias | 0.435 | 0.953 | 0.801 | 0.884 | -0.005 | within noise |
| No calendar features | 0.433 | 0.953 | 0.790 | 0.879 | -0.015 | within noise |
| Contrastive added | 0.486 | 0.953 | 0.787 | 0.880 | -0.018 | within noise |
| Working calendar features (v2 data) | 0.433 | 0.953 | 0.787 | 0.878 | -0.018 | within noise |
| Longer training (57 ep, budget lifted) | **0.405** | 0.948 | **0.756** | 0.861 | **-0.049** | **outside noise** |

### None of the architectural choices matter

Four of five land inside two standard deviations of seed noise, and each has a
specific explanation rather than a shrug.

**The DAG attention bias is redundant** because the model finds the graph anyway.
Without the bias, attention still puts 42.2% of off-diagonal mass on true edges
against 22.2% by chance (1.9x), compared with 46.4% and 2.1x with it. Telling the
model the structure nudges it and changes nothing downstream.

**Calendar features are redundant, established on the second attempt.** The first
version of this ablation was invalid: a unit bug made exported timestamps 1000x
too small, so the timeline covered nine distinct hours rather than twenty-four
and the calendar features encoded almost nothing. Removing features that carry no
signal cannot test anything. Retrained on corrected timestamps with a real
24-hour cycle, the answer is unchanged at Hit@1 0.787 against a seed mean of
0.805 +/- 0.011. A 32-hour window already spans a full day, so the phase is
inferable from the visible history and an explicit clock adds nothing.

**The contrastive objective hurts reconstruction and helps nothing.** Validation
loss rises from 0.438 to 0.486 and the contrastive term itself collapses to
0.055, meaning the task became trivial. With augmentations deliberately weak, so
as not to synthesise something indistinguishable from a real fault, the two views
are too easy to match. Stronger augmentation is not the fix, since that is
exactly what would teach the model that real faults are normal.

**Contamination tolerance is confirmed.** Training on clean windows only gives
+0.009, inside noise, so the method genuinely tolerates the 3% contamination that
real pipeline history carries.

### Better reconstruction makes localization worse

The only effect outside the noise floor is one we were not looking for. Training
57 epochs instead of 33 improves validation reconstruction from 0.433 to 0.405
and drops localization from 0.805 to 0.756 Hit@1.

This is the over-generalisation failure familiar from reconstruction-based
anomaly detection: a model that reconstructs well enough begins reconstructing
anomalies too, shrinking the error signal that detection and localization both
rely on. The contrastive run says the same thing from the other side, with
clearly worse reconstruction and unchanged localization.

The implication for this protocol is uncomfortable and worth stating. Checkpoints
were selected on validation reconstruction loss precisely because it needs no
labels. That criterion is now shown to be poorly aligned with the task, so the
reported numbers come from a selection rule that is defensible but demonstrably
suboptimal. Selecting on validation AUPRC would fix the alignment and break the
label-free claim. Finding a label-free proxy that tracks localization is the most
interesting open problem this project surfaced.
