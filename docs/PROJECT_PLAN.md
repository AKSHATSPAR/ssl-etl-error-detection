# Project Plan — Self-Supervised AI for ETL Error Detection and Root-Cause Localization

## 1. Problem formulation

An ETL pipeline is a DAG of tasks `V = {v_1 ... v_N}`. Every scheduled run *t*
produces, for each task, a data-quality profile `x_t^i ∈ R^F` (F = 38). A run is
therefore a matrix `X_t ∈ R^{N×F}`, and the observable history is a tensor
`X ∈ R^{T×N×F}`.

Given a window `X_{t-W+1:t}` the system must output:

| Output | Type | Meaning |
|---|---|---|
| `s_t ∈ R` | scalar | run-level anomaly score |
| `r_t ∈ R^N` | vector | per-task root-cause score |
| `a_t ∈ R^{N×F}` | matrix | feature attribution — the *symptom* |
| `c_t` | class | fault-type (probed with few labels, not required) |

**Constraint that defines the project:** none of the training signal may come
from labels. `s_t`, `r_t` and `a_t` must all fall out of a self-supervised
objective.

### Why this is not just anomaly detection

Under an incident at node `v*`, the set of *anomalous* nodes is
`{v*} ∪ descendants(v*)`. Detection asks "is any node anomalous"; localization
asks "which one is `v*`". Because downstream symptoms are frequently **larger**
than the originating one — a 5% null spike at extraction becomes a 30% revenue
drop after aggregation — simply ranking nodes by anomaly score is a poor
localizer. This is the core difficulty, and the reason the model must reason over
the DAG rather than over each node independently.

---

## 2. Method

### 2.1 Backbone

A **factorised spatio-temporal Transformer encoder** over the `[W, N, F]` tensor:

* per-node feature embedding `R^F → R^d`
* alternating **temporal attention** (across `W`, causal) and **structural
  attention** (across `N`, biased by the DAG adjacency so a task attends mainly to
  its ancestors and descendants)
* learned node embeddings, sinusoidal time encodings, plus calendar features
  (hour-of-day, day-of-week) so seasonality is explained away rather than flagged

Factorising the attention keeps cost at `O(W² + N²)` per layer instead of
`O((WN)²)`, which matters because the DAG axis is short and the time axis is not.

### 2.2 Self-supervised objectives

**(a) Masked profile modelling (primary).** Mask a random subset of
`(timestep, node, feature-group)` cells and reconstruct them from context. Masking
whole *feature groups* at a node — rather than scattered scalars — forces the model
to use cross-node and cross-time structure instead of interpolating a feature from
its neighbours in the same vector. This is the ETL analogue of masked autoencoding.

**(b) Ancestor-consistency / next-run prediction (auxiliary).** Predict a node's
profile at time *t* from its **ancestors'** profiles at time *t* plus its own
history. This is the objective that actually teaches causal direction along the
DAG, and it is what the root-cause scorer is built on (§2.3).

**(c) Contrastive augmentation view (auxiliary).** Two augmented views of the same
window (jitter, scaling, temporal crop) are pulled together; different windows are
pushed apart. Augmentations are chosen to be *fault-preserving-free* — i.e. they
must never synthesise something that looks like a real fault.

### 2.3 Root-cause localization

Three signals, combined and ablated against each other:

1. **Residual magnitude** `‖x_t^i − x̂_t^i‖` — the naive baseline. Included
   precisely to show it is not enough.
2. **Ancestor-conditioned surprise.** A node whose profile is *predictable from its
   parents* is behaving consistently with its inputs — it is a victim, not a cause.
   A node whose profile is surprising *even given its parents* is the origin.
   Score: `r_t^i = ‖x_t^i − x̂_t^i(parents)‖ − ‖x_t^i − x̂_t^i(history only)‖`.
   This is the main methodological contribution.
3. **Counterfactual upstream repair.** Replace node *i*'s profile with its
   model-predicted healthy value, re-run the decoder, and measure how much the
   downstream anomaly dissolves. The node whose repair explains the most downstream
   anomaly is the root cause.

### 2.4 Explainability

* **Feature attribution** via Integrated Gradients on the anomaly score, aggregated
  into the eight named feature families, so the output reads
  *"`clean_orders`, completeness family, `null_rate_max`"* rather than a number.
* **DAG attention rollout** to visualise which upstream path the model blamed.
* Attribution is evaluated quantitatively, not just shown: each fault type has a
  known ground-truth symptom family, so we can report **attribution accuracy**.

---

## 3. Baselines

| Class | Methods | Purpose |
|---|---|---|
| Classical | z-score / EWMA thresholds, PCA residual, Isolation Forest, LOF | what a data-quality tool (Great Expectations, Deequ) effectively does today |
| Shallow deep | Dense autoencoder, LSTM autoencoder | the standard reconstruction baseline |
| Time-series SOTA | USAD-style adversarial AE, TranAD-style transformer | published multivariate anomaly detectors |
| Supervised upper bound | Same backbone, trained with labels | the ceiling the SSL method is trying to approach without labels |
| Ablations | no-DAG-attention, no-ancestor-objective, no-contrastive, residual-only RCA | isolates where the gain comes from |

The supervised model is an **upper bound, not a competitor** — its point is to
quantify how much is lost by dropping labels.

---

## 4. Evaluation

### Detection
* AUROC, AUPRC (AUPRC is the honest one — anomalies are ~10–14% of runs)
* Best-F1, and F1 at a fixed alert budget (e.g. 1 alert per 100 runs), which is
  what an on-call rotation actually cares about
* Detection delay for sustained incidents
* **No point-adjustment.** Point-adjusted F1 is known to inflate scores to the
  point where random noise beats real models, so raw per-run scores are reported.
  Point-adjusted numbers may be shown *alongside*, labelled as such.

### Localization
* **Hit@1 / Hit@3** and **MRR** over the N candidate nodes
* Localization accuracy broken down **by fault type** and **by severity quartile** —
  reporting only the mean hides that low-severity incidents are near-impossible
* **Attribution accuracy**: does the top feature family match the fault's known
  symptom family?

### Robustness
* Performance vs. training-set contamination rate (0%, 3%, 10%)
* Performance on benign distractors — the false-positive rate during campaigns and
  drift periods, reported separately

---

## 5. Phase plan

| Chunk | Contents | Status |
|---|---|---|
| 1 | Simulator, fault taxonomy, profiler, dataset builder, windowing + robust normalisation, test suite, repo scaffold | **done** |
| 2 | Backbone + SSL objectives, training loop, classical & deep baselines, literature review | next |
| 3 | Root-cause localization, Integrated Gradients attribution, ablations | |
| 4 | Full results, figures, error analysis, final written report | |

---

## 6. Design decisions worth defending in a viva

**Counterfactual propagation labels.** Labelling the whole descendant set as
anomalous is an over-approximation that some faults falsify outright. Re-running
each faulty batch with the fault removed, under identical seeds, turns
propagation from an assumption into a measurement. The topological label is kept
alongside so the two can be compared directly in the results.

**Row-level simulation instead of feature-level.** Injecting a fault into the
*features* would mean hand-writing how it propagates, and the model would then be
learning our propagation rules rather than a data pipeline. Corrupting the rows and
letting the real transforms carry the damage downstream removes that circularity.

**Chronological split, robust normalisation.** A random split leaks the future.
Mean/std normalisation would let the contaminated training batches define "normal";
median/IQR does not.

**Contaminated training data.** Training on a guaranteed-clean set is the standard
shortcut in the anomaly-detection literature and it is unavailable in practice —
nobody knows which historical runs were clean. 3% contamination is the honest setting.

**Benign distractors are labelled normal.** Seasonality, growth and campaigns all
move row counts hard. Without them, a threshold on `log_rows_out` would score well
and the deep model would be unjustifiable.

**No point-adjusted F1 as the headline metric.** It is the standard metric in this
literature and it is known to be badly broken; using it would make the results
look better and mean less.
