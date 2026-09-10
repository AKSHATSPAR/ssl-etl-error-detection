# Project Plan

*Self-Supervised AI for ETL Error Detection and Root-Cause Localization*

## 1. Problem formulation

An ETL pipeline is a DAG of tasks. Every scheduled run *t* produces, for each
task, a data-quality profile of 38 numbers. A run is therefore a matrix of shape
`[N, F]` with N = 9 tasks and F = 38 features, and the observable history is a
tensor `[T, N, F]`.

Given a window of recent runs, the system has to produce four things:

| Output | Shape | Meaning |
|---|---|---|
| `s` | scalar | run-level anomaly score |
| `r` | `[N]` | per-task root-cause score |
| `a` | `[N, F]` | feature attribution, meaning the symptom |
| `c` | class | fault type, probed with few labels rather than required |

The constraint that defines the project is that none of the training signal may
come from labels. All of `s`, `r` and `a` have to fall out of a self-supervised
objective.

### Why this is not just anomaly detection

During an incident originating at task *v*, the set of anomalous tasks is *v*
together with everything downstream of it. Detection asks whether any task is
anomalous. Localization asks which one started it.

The difficulty is that downstream symptoms are often larger than the original
one. A five percent null rate at extraction can become a thirty percent revenue
drop after aggregation. So ranking tasks by anomaly magnitude, which is what the
structure-aware detectors in the literature do, systematically blames the task
furthest from the fault. This is the core problem, and it is why the model has to
reason over the DAG rather than treating each task independently.

## 2. Method

### 2.1 Backbone

A factorised spatio-temporal transformer encoder over the `[W, N, F]` window, at
roughly 1.1M parameters.

Each layer embeds every task profile into `d` channels, then attends along time
(each task over its own history), then across the DAG (each timestep over the
other tasks), then applies a feed-forward block. Factorising keeps the cost at
`O(W² + N²)` per layer rather than `O((WN)²)`, which matters because the DAG axis
is short and the time axis is not.

Structural attention carries a learned bias per DAG relationship. Every ordered
pair of tasks is classified as self, direct parent, direct child, distant
ancestor, distant descendant or unrelated, and the model learns one bias scalar
per relationship per head. This lets it decide how much a task should attend to
its immediate upstream rather than having a hard mask imposed on it, and it keeps
the attention maps interpretable for the explainability pass.

Learned task embeddings, sinusoidal time encodings and cyclic calendar features
complete the input. The calendar features matter: volume swings by roughly three
times across a day, so without a clock the dominant reconstruction error is
simply the time of night.

### 2.2 Self-supervised objectives

**Masked profile modelling** is the primary objective. Hide a subset of entries
and reconstruct them from context, with the mask given to the model as an input
rather than only used as a loss weight, so filling it in is a well-posed question
instead of a guess about whether a zero means healthy or missing.

Masking whole symptom families rather than scattered scalars is deliberate.
`null_rate_mean` sits next to `null_rate_max` in the same vector and is trivially
recoverable from it, so scattered masking can be solved by within-vector
interpolation without ever looking at the pipeline.

**Node masking** hides a task's entire profile at a timestep. Filling that in
means predicting what a task should have produced given what its parents
produced, which is the ancestor-consistency objective the root-cause scorer is
built on.

**Contrastive views** are treated as an ablation rather than a default, because
they cost roughly four times the compute per epoch and their benefit here is an
open question rather than an assumption. The augmentations are kept weak on
purpose: strong amplitude scaling is indistinguishable from a unit change, and
heavy dropout is indistinguishable from a null spike, so an aggressive
augmentation would train the model to treat real faults as normal.

### 2.3 Root-cause localization

Scoring mirrors training, using two masking configurations at the final timestep.

`e_hist` hides every task, so each prediction rests on history alone. It is large
whenever a run departs from its own past, which makes it the detection signal.

`e_ctx` hides only task *i* and leaves everything else at that timestep visible,
parents included. This is surprise given inputs. A task downstream of a failure
produces anomalous output that follows perfectly from its own anomalous inputs,
so `e_ctx` stays small and it reads correctly as a victim. The originating task
has healthy inputs and anomalous output, so `e_ctx` stays large.

The naive alternative, ranking by `e_hist`, is reported alongside as the
comparison this has to beat. If ancestor conditioning does not improve
localization, the contribution does not exist and the results should say so.

### 2.4 Explainability

Attribution is aggregated into the eight named symptom families, so the output
reads as "clean_orders, completeness family, null_rate_max" rather than as a
number on an unnamed axis. Because the simulator knows which family each fault
type should move, attribution is scored as accuracy rather than merely displayed.
Integrated Gradients and DAG attention rollout come in chunk 3.

## 3. Baselines

| Class | Methods | Purpose |
|---|---|---|
| Classical | rolling z-score, PCA residual, per-node Isolation Forest, per-node LOF | what a data-quality tool such as Deequ or Great Expectations effectively does today |
| Deep reconstruction | dense autoencoder, LSTM autoencoder | the standard baseline in this literature |
| Supervised ceiling | the same backbone trained with labels | quantifies what dropping labels actually costs |
| Ablations | no DAG bias, no calendar, clean training set, residual-only localization, plus contrastive | isolates where any gain comes from |

The classical baselines are given the current profile *and* its deviation from
the window's rolling mean, so they have temporal context too. A baseline crippled
by construction proves nothing. Per-node Isolation Forest and LOF are fitted once
per task rather than once over the concatenated 342-dimensional vector, which is
both the stronger configuration and the only way to get a node-level score out of
them at all.

The supervised model is a ceiling, not a competitor.

## 4. Evaluation

**Detection.** AUROC and AUPRC, with AUPRC the honest one given that incidents
are about 14% of test runs. Best F1 over all thresholds as an oracle upper bound,
and F1 at a fixed alert budget of one alert per hundred runs, which is closer to
what an on-call rotation actually experiences. Detection delay for sustained
incidents.

Point-adjusted F1 is reported but never as the headline. Under point adjustment,
catching one timestep inside an incident credits the whole incident, and
published re-evaluations have shown random scores beating real models on the
standard benchmarks.

**Localization.** Hit@1, Hit@3 and MRR over the nine candidate tasks, computed
only on runs that carry an incident. Broken down by fault type and by severity
quartile, because a single mean hides that low-severity incidents are close to
impossible and high-severity ones close to trivial.

**Attribution.** Whether the top-scoring symptom family matches the fault's known
symptom, scored per fault type.

**Robustness.** Performance against training-set contamination at 0%, 3% and 10%,
and the false-positive rate specifically during campaigns and drift periods,
reported separately from the overall rate.

## 5. Scope decisions

Confirmed with the student on 2026-09-10.

The purpose-built synthetic benchmark is the dataset. No public ETL failure
corpus with root-cause annotations exists, so constructing one is part of the
contribution rather than a way around a missing baseline.

The deliverable is working code together with a written report, so every result
has to be reproducible from a script and every figure regenerable. No number
should exist only in terminal output.

Training runs are sized to finish in minutes on Apple Silicon. This is a
deliberate constraint rather than a limitation, because the project needs many
ablation runs far more than it needs one large model, and a dataset of 8000 by 9
by 38 does not justify a big backbone.

## 6. Phase plan

| Chunk | Contents | Status |
|---|---|---|
| 1 | Simulator, fault taxonomy, profiler, dataset builder, windowing and robust normalisation, test suite, repo scaffold | done |
| 2 | Backbone and SSL objectives, training loop, classical and deep baselines, evaluation harness, literature review | done |
| 3 | Ablations, Integrated Gradients attribution, attention rollout, error analysis | next |
| 4 | Final results, figures, written report | |

## 7. Design decisions worth defending in a viva

**Counterfactual propagation labels.** Labelling the whole descendant set as
anomalous is an over-approximation that several faults falsify outright.
Re-running each faulty batch with the fault removed, under identical seeds, turns
propagation from an assumption into a measurement. The topological label is kept
alongside so the two can be compared directly.

**Row-level simulation rather than feature-level.** Injecting a fault into the
features would mean hand-writing how it propagates, and the model would then be
learning our propagation rules rather than anything about data pipelines.
Corrupting the rows and letting the real transforms carry the damage downstream
removes that circularity.

**Chronological splits and robust normalisation.** A random split leaks the
future. Mean and standard deviation would let the contaminated training batches
define what normal means, so median and IQR are used instead, with a standard
deviation fallback for the features that never vary while healthy.

**Contaminated training data.** Training on a guaranteed-clean set is the
standard shortcut in this literature and it is unavailable in practice, because
nobody knows which historical runs were clean. Three percent contamination is the
honest setting, and the clean-training-set version is kept as an ablation.

**Label-free checkpoint selection.** Selecting on validation AUPRC would make
this weakly supervised while still calling it self-supervised, and would inflate
every number reported afterwards.

**Benign distractors labelled normal.** Seasonality, growth and campaigns all
move row counts hard. Without them a threshold on row count would score well and
the deep model would be difficult to justify.

**No point-adjusted F1 as the headline.** It is the standard metric in this
literature and it is known to be broken. Using it would make the results look
better and mean less.
