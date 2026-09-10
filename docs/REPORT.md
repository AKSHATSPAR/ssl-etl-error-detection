# Self-Supervised Deep Learning for ETL Error Detection and Root-Cause Localization

**B.Tech Deep Learning Project**

---

## Abstract

Extract-Transform-Load pipelines fail quietly. A source system starts emitting
nulls, a currency convention changes upstream, a join key gets reformatted, and
the pipeline keeps running and keeps loading data that is wrong. The failure is
usually noticed days later by someone reading a dashboard. Detecting these
failures automatically is hard because labelled examples barely exist, and
locating their cause is harder still, because by the time a fault reaches a
dashboard every task downstream of it looks broken.

This project addresses both problems with a self-supervised model trained on
unlabelled pipeline telemetry. Because no public benchmark of labelled ETL
failures with root-cause annotations exists, we first construct one: a nine-task
warehouse pipeline simulated at row level, with fifteen fault types injected into
the rows so that propagation through the task graph emerges from the simulation
rather than being scripted. Propagation labels are obtained by re-running every
faulty batch with the fault removed under identical random seeds, which makes the
blast radius a measurement rather than an assumption. Measured reach is 2.48 of 9
tasks per incident against 3.94 under the usual topological assumption.

The model is a factorised spatio-temporal transformer trained by masked
reconstruction over the pipeline's data-quality profiles. On detection it reaches
0.953 AUPRC, which is competitive with but does not beat a Local Outlier Factor
baseline at 0.973. On root-cause localization it reaches 0.797 Hit@1, statistically
tied with three baselines.

The finding that survives scrutiny concerns scoring rather than architecture.
Ranking tasks by raw anomaly magnitude asks which task has the largest error,
when the question is which task is furthest from its own normal. Two independent
corrections for this, conditioning each task's prediction on its parents and
calibrating each task against its own healthy history, prove worth exactly the
same (0.773 Hit@1 each, against 0.588 uncalibrated), and combining them adds
little. An earlier version of this work reported the structural correction as
worth 18.5 points; measured against a fairly calibrated baseline it is worth 2.4.

Explanations are scored rather than displayed, which the benchmark makes possible:
Integrated Gradients names the correct symptom family for 93.5% of incidents.

**Keywords:** self-supervised learning, anomaly detection, root-cause analysis,
data quality, ETL pipelines, transformers, explainable AI

---

## 1. Introduction

### 1.1 Motivation

Modern analytics rests on data pipelines that no one watches closely. A typical
warehouse pipeline extracts from several source systems, cleans and conforms the
records, joins them into facts and dimensions, aggregates them, and publishes the
result to a reporting layer. Each stage runs on a schedule, often hourly, and
each stage can fail in ways that do not raise an exception.

The failures that matter are the quiet ones. A job that crashes gets noticed
within minutes because a scheduler alerts on it. A job that completes
successfully while writing amounts that are a hundred times too large, because an
upstream service switched from rupees to paise, can go unnoticed for weeks, and
every report built on that table is wrong for the whole period.

Two things make this hard to automate.

**Labels do not exist.** Incidents are rare, they are recorded inconsistently
when they are recorded at all, and a team that has just spent a night restoring a
table is not going to sit down and annotate the run history. Any method that
requires a thousand labelled examples of schema drift is a method that cannot be
deployed.

**Symptoms are not causes.** When extraction starts emitting nulls, the cleaning
step propagates them, the join loses rows, the aggregate under-reports revenue,
and the warehouse load looks thin. A detector that flags nine unhealthy tasks has
given the on-call engineer no information they did not already have. Worse, the
symptom often grows as it travels: a five percent null rate at extraction can
become a thirty percent revenue drop after aggregation, so the largest anomaly
frequently sits furthest from the fault.

### 1.2 Objectives

1. Detect broken pipeline runs without using labels for training.
2. Identify which task originated the incident, not merely which tasks are
   affected by it.
3. Explain the finding in terms an engineer can act on, naming the symptom rather
   than a feature index.
4. Evaluate all of the above honestly, on a benchmark that makes the questions
   answerable.

### 1.3 Contributions

**A benchmark.** A nine-task ETL pipeline simulated at row level with fifteen
fault types across six families, seasonality, growth, drift and campaign
distractors labelled normal, and 8000 hourly runs. Faults corrupt rows rather
than features, so downstream tasks consume corrupted data and propagation is an
emergent property of the simulation.

**Measured propagation labels.** Every faulty batch is re-run with the fault
removed and everything else held identical via per-node seeded randomness, so a
task counts as affected only if its profile actually moved. This is a
measurement, not the usual topological over-approximation, and the two differ
substantially.

**A self-supervised architecture** combining masked reconstruction over
data-quality profiles with DAG-structured attention, plus an ancestor-conditioned
scoring rule for localization.

**A negative result and a correction, both reported.** The DAG attention bias
turns out not to help, and we show why. The structural scoring rule is worth much
less than first measured, and we show what the original comparison was actually
capturing.

**Scored explanations.** Because the simulator knows which symptom family each
fault should move, attribution accuracy is a number rather than a picture.

---

## 2. Related Work

A full review with a paper-by-paper matrix is in `docs/LITERATURE_REVIEW.md`.
This section summarises the position.

### 2.1 Reconstruction-based anomaly detection

The dominant approach learns to reproduce normal behaviour and treats what the
model cannot reproduce as anomalous. Malhotra et al. established that LSTM
prediction error carries the signal, then showed reconstruction holds up better
on genuinely unpredictable series. OmniAnomaly made normality probabilistic at
significant computational cost, which USAD then addressed with adversarially
trained autoencoders reaching similar accuracy far faster.

All of these assume a clean training set. In an ETL setting that assumption
fails quietly, because nobody knows which historical runs were healthy, and the
incidents that were never noticed are still in the history.

### 2.2 Structure-aware detection

MTAD-GAT and GDN both add a graph over variables and both improve on flat
baselines, which is evidence that inter-variable structure carries real signal.
GDN in particular scores each sensor by how far it deviates from what its
neighbours predict, and describes this as localization.

In a sensor network that is defensible. In a pipeline it is not, for the reason
given in Section 1.1: magnitude does not survive propagation, so ranking by
deviation systematically blames the wrong end of the graph. This is the specific
gap the present work targets.

### 2.3 Self-supervised pretraining

Masked modelling and contrastive learning have transferred from language and
vision to time series through work such as Zerveas et al. and TS2Vec, and to
operational data through LogBERT. Masked autoencoding supplied the design lesson
that matters most here, which is that masking must be aggressive enough that the
task cannot be solved by local interpolation.

None of this has reached ETL data quality. LogBERT is the nearest neighbour and
it models log key sequences, so it sees that a job ran and in what order but
never what the data looked like. A pipeline can execute flawlessly, emit a perfect
log, and load a table where every amount is wrong.

### 2.4 Data quality systems

Deequ and TensorFlow Data Validation made data quality a production concern and
encode a taxonomy of ETL failures more grounded than anything in the machine
learning literature. Their limitation is that somebody has to write the rules.
Auto-Validate removes much of that burden but works within a single column, so it
cannot see a fault that only appears as a relationship between tables.

### 2.5 Evaluation practice

Kim et al. re-evaluated the standard benchmarks under point-adjusted F1, the
metric most of this literature reports, and found that randomly generated scores
can beat published models. Point adjustment credits an entire anomalous segment
for detecting any single timestep inside it, so on datasets with long anomalous
stretches, alerting at random works. Wu and Keogh reached a related conclusion
from the data side.

This work therefore reports AUPRC and F1 at a fixed alert budget as headline
metrics, and point-adjusted F1 only alongside, labelled as such.

---

## 3. Problem Formulation

An ETL pipeline is a directed acyclic graph of tasks. Each scheduled run *t*
produces, for every task, a data-quality profile of *F* = 38 numbers. A run is
therefore a matrix in R^(N x F) with *N* = 9 tasks, and the observable history is
a tensor in R^(T x N x F).

Given a window of *W* consecutive runs ending at time *t*, the system must
produce:

| Output | Shape | Meaning |
|---|---|---|
| s | scalar | run-level anomaly score |
| r | N | per-task root-cause score |
| a | N x F | feature attribution, the symptom |

Under an incident originating at task *v*, the set of anomalous tasks is *v*
together with those downstream of it that the fault actually reaches. Detection
asks whether any task is anomalous. Localization asks which one is *v*.

The constraint that defines the project is that no training signal may come from
labels. Labels exist only to evaluate, and to train the supervised comparisons.

---

## 4. The Benchmark

### 4.1 Why build one

There is no public benchmark of ETL failures with root-cause annotations. The
detection literature evaluates almost entirely on SMD, SWaT, SMAP and MSL, which
are server and sensor telemetry: no task graph, no data-content quality metrics,
and no label for which component originated a failure. Root-cause localization in
a data pipeline currently cannot be posed as a benchmark task at all.

### 4.2 Pipeline and simulation

The simulated pipeline has nine tasks:

```
extract_customers ─┐
                   ├─ clean_customers ─┐
extract_orders ────┴─ clean_orders ────┴─ join_orders_customers ─┐
                                                                 ├─ join_payments ─ aggregate_daily ─ load_warehouse
extract_payments ────────────────────────────────────────────────┘
```

Every task performs a real transformation over actual dataframe rows: type
casting, deduplication, currency normalisation to a common unit, left joins,
group-by aggregation, and a final publish step. Transforms are written to degrade
rather than crash, as production jobs do, so a task whose join key was removed
upstream produces empty joins rather than an exception.

Source data is generated with a persistent pool of 20,000 customers, so foreign
keys point at something real and referential integrity is a meaningful signal.
Volume follows a daily cycle, a weekly cycle, a growth trend, slow drift in the
customer country mix, and occasional marketing campaigns that raise volume by
about 1.7x for a day or two. All of this variation is labelled normal on purpose,
because without it a threshold on row count would score well and no deep model
would be justifiable.

![**Figure 1.** Order volume over 600 consecutive runs, with labelled incidents shaded red. The daily cycle and campaign surges move the row count as much as several genuine fault types do, which is why a threshold on volume cannot work and why these distractors are in the benchmark.](artifacts/dataset_report/fig1_volume_incidents.png)


### 4.3 Fault taxonomy

Fifteen fault types across six families:

| Family | Faults |
|---|---|
| Schema | column drop, column add, type mismatch |
| Completeness | null spike, primary key null, duplicate records, row loss truncation |
| Value and distribution | unit change, value corruption, category corruption |
| Temporal | timezone shift, late arrival |
| Integrity | foreign key break, join fan-out |
| Operational | resource degradation |

Faults are injected into the rows, never into the features. Downstream tasks then
consume the corrupted rows, so propagation is emergent. This matters more than it
might appear: if propagation were hand-written, the model would be learning our
propagation rules rather than anything about data pipelines.

`resource_degradation` is deliberately awkward. The task blows its service level
objective while its output data stays perfectly valid, so a data-quality detector
must not confuse it with corruption, and it must not propagate.

### 4.4 Measuring propagation instead of assuming it

The obvious way to label which tasks an incident affected is to take the origin
plus every descendant. That is wrong, and provably so. An unexpected extra column
is ignored by every downstream select. A task that runs slowly still emits correct
rows. Labelling those descendants anomalous would penalise a detector for being
right about them.

So the simulator re-runs every faulty batch a second time with the fault removed,
holding the source rows, telemetry noise and scheduling jitter identical. This
works because each task draws its randomness from its own generator seeded on
(seed, batch, task), which guarantees that the two runs are bit-identical
wherever the fault does not reach. A unit test asserts exactly that.

A task counts as affected only when its profile actually moved. Measured reach is
**2.48 of 9 tasks** per incident against **3.94** under the topological
assumption. `schema_column_add` correctly reaches exactly one task where the
topological label claims 4.5.

![**Figure 2.** Blast radius of a single incident, measured against the counterfactual so a healthy run is exactly zero by construction. The incident originates at `extract_customers` and moves 8 of 9 tasks, several of them harder than the origin itself. This is why ranking tasks by anomaly magnitude cannot localize.](artifacts/dataset_report/fig2_propagation.png)


### 4.5 Feature extraction

Each task run is summarised into 38 features aggregated by column role, which is
what keeps the width constant across tasks with completely different schemas:
operational telemetry, schema stability, completeness, primary metric
distribution, secondary numeric distribution, categorical distribution, temporal
freshness, and referential integrity.

A validation experiment confirms every fault type produces a greater than
three-sigma response at the injected task, and that each moves its own expected
symptom family: unit change moves the metric distribution family by 9.5 robust
units, timezone shift moves temporal by 10.7, resource degradation moves
operational by 10.2, and the healthy baseline stays flat.

![**Figure 3.** Response of each symptom family to each fault type, in robust units, measured on the tasks the fault actually reached. The diagonal dominance is what makes attribution scorable: each fault has a known expected symptom. The top row shows healthy runs staying flat.](artifacts/dataset_report/fig4_fault_signature.png)


### 4.6 Dataset statistics

| Property | Value |
|---|---|
| Runs | 8000 hourly, about eleven months |
| Shape | 8000 x 9 x 38 |
| Incidents | 273 spanning 594 runs (7.4%) |
| Training contamination | 3.0% |
| Test incident rate | 14.6% |
| Split | chronological, 60 / 15 / 25 |

![**Figure 4.** Incident mix across the 8000-run timeline. All fifteen fault types are represented, with between 12 and 70 affected runs each.](artifacts/dataset_report/fig3_fault_mix.png)


---

## 5. Methodology

### 5.1 Architecture

The backbone is a factorised spatio-temporal transformer over the `[W, N, F]`
window, with about 1.1 million parameters.

Attending over `W x N` positions jointly would cost O((WN)^2) per layer for no
good reason: the task axis is short at 9, the time axis is long at 32, and the
two carry different kinds of structure. Each encoder block therefore attends
along time first, letting every task look at its own history, then across the
task graph, letting every timestep look at the other tasks. The cost is
O(W^2 + N^2).

Structural attention carries a learned bias per graph relationship. Every ordered
pair of tasks is classified as self, direct parent, direct child, distant
ancestor, distant descendant or unrelated, and the model learns one bias scalar
per relationship per attention head. This lets it decide how much a task should
attend to its immediate upstream, rather than having a hard mask imposed. Section
8.2 reports that this component does not help, and explains why.

Learned task embeddings, sinusoidal time encodings and cyclic calendar features
complete the input. Calendar features matter: volume swings roughly threefold
across a day, so without a clock the largest reconstruction error in the dataset
is simply the time of night.

The mask is concatenated to the input rather than only used as a loss weight, so
the model always knows which entries it is being asked to predict. A test asserts
that rewriting a masked value leaves the output unchanged, which rules out the
model reading the answer it is meant to predict.

### 5.2 Self-supervised objectives

**Feature-group masking** hides one whole symptom family at a task and timestep.
Masking scattered individual numbers would be far too easy, because
`null_rate_mean` sits beside `null_rate_max` in the same vector and is trivially
recoverable from it. The model would learn within-vector correlations and never
look at the pipeline. Hiding an entire family forces it to use other tasks and
other timesteps.

**Task masking** hides a task's entire profile at a timestep. Filling that in
requires predicting what a task should have produced given what its parents
produced, which is the objective the root-cause scorer is built on.

**Contrastive views** were implemented but treated as an ablation rather than a
default, because they cost 4.3 times the compute per epoch. Their augmentations
are deliberately weak: strong amplitude scaling is indistinguishable from a unit
change and heavy dropout from a null spike, so aggressive augmentation would
train the model to treat real faults as normal.

Checkpoints are selected on validation reconstruction loss, which needs no labels.
Selecting on validation AUPRC would make the method weakly supervised while still
calling it self-supervised, and would inflate every number reported afterwards.

### 5.3 Scoring

Scoring mirrors training. The model was trained to fill in hidden entries, so at
inference it is asked the same kind of question. Running a masked autoencoder
unmasked lets it copy its input to its output, and the reconstruction error then
measures nothing.

Two masking configurations at the final timestep:

**e_hist** hides every task, so each prediction rests on history alone. Large
whenever a run departs from its own past, which makes it the detection signal.

**e_anc** hides task *i* together with all of its descendants, so the prediction
rests only on *i*'s ancestors and on history. This is surprise given inputs. A
task downstream of a failure emits anomalous output that follows perfectly from
its own anomalous inputs, so e_anc stays small and it reads correctly as a
victim. The task where the incident began has healthy inputs and anomalous output,
so e_anc stays large.

Hiding the descendants is essential and was missing from the first
implementation. Structural attention is bidirectional, so a hidden task can be
reconstructed from its own descendants, and during an incident those are corrupted
too and leak the value straight back. Section 8.1 quantifies the difference.

**Per-task calibration.** Tasks are not equally predictable, so their raw errors
are not comparable and whichever task is hardest to predict wins the ranking by
default. Each task's score is therefore converted to a percentile against its own
distribution over unlabelled training runs. Percentile rank rather than a z-score,
because the error distributions are heavy-tailed and an interquartile range
summarises them poorly; z-scoring measurably hurt.

### 5.4 Explainability

Attribution is aggregated into the eight named symptom families, so the output
reads as "clean_orders, completeness family, null_rate_max" rather than as an
index. Two methods are compared: per-feature reconstruction error, and Integrated
Gradients of the anomaly score with an all-zero baseline. The zero baseline is
principled here rather than arbitrary, because in the robust-scaled space zero is
the healthy median of every feature, so the baseline literally means a completely
typical run.

Structural attention rollout shows which upstream tasks the model consulted.

---

## 6. Experimental Setup

### 6.1 Protocol

The split is chronological, never random, because a random split over a time
series leaks the future into the past. Training uses the first 60%, validation
the next 15%, and test the final 25%.

Training history carries about 3% incident contamination, as real history does.
Normalisation uses median and interquartile range, robust to that contamination,
fitted on the training split alone, with a standard deviation fallback for
features that never vary while healthy. Without that fallback a schema flag
flipping would register as a one-sigma event, weaker than ordinary volume noise.

At run *t* the model sees only runs *t-W+1* through *t*.

### 6.2 Baselines

| Class | Methods |
|---|---|
| Classical | rolling z-score, PCA residual, per-task Isolation Forest, per-task Local Outlier Factor |
| Deep reconstruction | dense autoencoder, LSTM autoencoder |

Classical baselines receive the current profile together with its deviation from
the window's rolling mean, so they have temporal context too. A baseline crippled
by construction proves nothing. Isolation Forest and Local Outlier Factor are
fitted once per task rather than once over the concatenated 342-dimensional
vector, which is both the stronger configuration and the only way to obtain a
task-level score from them.

Every baseline receives the same percentile calibration as the model, fitted the
same way on unlabelled training scores. Calibration turns out to be worth more
than the choice of detector, so applying it to one side only would have been the
result rather than a detail.

### 6.3 Metrics

Detection is measured by AUROC and AUPRC, with AUPRC the honest one at a 14.6%
base rate, plus best F1 over all thresholds and F1 at fixed alert budgets.
Point-adjusted F1 is reported but never as the headline.

Localization is measured by Hit@1, Hit@3 and mean reciprocal rank over the nine
candidate tasks, computed only on runs carrying an incident, with 95% bootstrap
intervals. Asking which task is at fault on a healthy run is not a meaningful
question.

Attribution is measured by whether the top symptom family matches the fault's
mechanistically expected family.

### 6.4 Implementation

PyTorch 2.14 on Apple Silicon using the Metal backend. Training a single
configuration takes about 18 minutes, deliberately, because the project needs
many ablation runs far more than one large model, and a dataset of 8000 by 9 by
38 does not justify a larger backbone. The full test suite is 54 tests covering
simulator contracts, healthy baselines, fault propagation, counterfactual
isolation, masking behaviour, scoring indexing and metric correctness.

---

## 7. Results

All numbers are on the test split: 2000 runs, 291 carrying an incident. Nothing
was tuned on test.

### 7.1 Main comparison

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


The self-supervised model does not win on either task. Local Outlier Factor has
the best detection and the LSTM autoencoder the best localization. With 291
incidents the standard error on a Hit@1 near 0.8 is 0.023, and the seed-to-seed
standard deviation of our own model is 0.011, so the top four localizers should
be read as tied rather than ranked.

This deserves an explanation rather than an excuse. Each task profile is only 38
dimensions and the healthy manifold is densely covered by 4800 training runs,
which is close to the ideal operating regime for a density or subspace method.
There was little headroom to claim: five of the seven methods sit between 0.94
and 0.97 AUPRC. A benchmark that separates methods sharply on localization can
still fail to separate them on detection, and this one does.

![**Figure 5.** Detection and localization across all seven methods, ours in blue. Two panels rather than one shared axis, because they are different questions and a method can be strong at one and weak at the other.](artifacts/results/fig5_method_comparison.png)


Two failures among the baselines are informative. Rolling z-score, the stand-in
for the rule-based tooling teams actually run today, collapses to 0.502 AUPRC
because the benign distractors defeat it. Seasonality, growth and campaigns move
features as hard as several genuine fault types do, which is precisely why those
distractors are in the benchmark. Isolation Forest is weakest overall, a known
consequence of axis-parallel splits on correlated features.

### 7.2 The main finding: cross-task comparability

Ranking nine raw errors asks which task has the largest error. The question is
which task is furthest from its own normal. Since tasks are not equally
predictable, the two questions have different answers, and whichever task is
intrinsically hardest to predict wins the raw ranking by default.

There are two independent ways to correct this, and the result is that they are
worth the same.

| SSL localizer variant | Hit@1 [95% CI] | MRR [95% CI] |
|---|---|---|
| Deviation magnitude, raw | 0.588 [0.533, 0.646] | 0.762 [0.729, 0.796] |
| Deviation magnitude, **statistically** calibrated | 0.773 [0.725, 0.821] | 0.872 [0.843, 0.899] |
| Ancestor-conditioned, raw (**structural**) | 0.773 [0.725, 0.821] | 0.867 [0.838, 0.896] |
| Ancestor-conditioned and calibrated | 0.797 [0.749, 0.842] | 0.883 [0.855, 0.910] |

Rows two and three are identical to three decimal places with the same interval.

Structural conditioning achieves comparability by predicting each task from its
parents, so whatever a task's inputs already explain is subtracted away.
Statistical calibration achieves it by converting each task's error into a
percentile against its own healthy history. Different mechanisms, the same
effect, the same number. Applying both adds 2.4 points, which the intervals
cannot distinguish from noise.

An earlier version of this work reported the structural correction as worth 18.5
Hit@1 points. That figure compared it against an uncalibrated baseline. The
corrected comparison is the more interesting one, because it identifies what the
gain was actually made of.

### 7.3 Calibration is not universally beneficial

| Method | Uncalibrated Hit@1 | Calibrated | Delta |
|---|---|---|---|
| LSTM autoencoder | 0.643 | 0.818 | **+0.175** |
| Dense autoencoder | 0.725 | 0.749 | +0.024 |
| SSL (ancestor-conditioned) | 0.773 | 0.797 | +0.024 |
| PCA residual | 0.801 | 0.801 | 0.000 |
| Local Outlier Factor | 0.797 | 0.773 | -0.024 |
| Rolling z-score | 0.663 | 0.570 | **-0.093** |

The effect tracks how far apart a method's per-task error scales already are.
The LSTM autoencoder reconstructs all nine tasks through one shared hidden state,
so its per-task error scales differ wildly and calibration has a great deal to
correct. PCA and Local Outlier Factor already produce roughly comparable
per-task scores, so there is nothing to gain and the rank transform only
discards magnitude information.

The practical lesson is that per-task calibration should be checked rather than
assumed, and checking is cheap because it requires no labels.

### 7.4 Localization by graph depth

The first implementation of the ancestor score hid only the task under test,
leaving its descendants visible to leak the corrupted value back. The prediction
was that this leak should hurt most where a task has the most descendants,
meaning upstream, and should vanish at the sink.

| Origin depth | Ancestor (descendants hidden) | Self-masked only | Deviation magnitude | n |
|---|---|---|---|---|
| 0 (source tasks) | 0.598 | 0.546 | 0.505 | 97 |
| 1 | 0.806 | 0.583 | 0.556 | 72 |
| 2 | 0.821 | 0.643 | 0.643 | 28 |
| 3 | 0.889 | 0.889 | 0.667 | 18 |
| 4 | 0.938 | 0.625 | 0.562 | 48 |
| 5 (sink) | 0.893 | 0.893 | 0.893 | 28 |

At depth 5 all three rules agree exactly, because the sink task has no
descendants and the three masks are then the same tensor. That agreement is not
a tuned result, it follows from the construction, and it is the cleanest
available evidence that the mechanism is the one claimed.

Depth 0 is the hardest row and gains least, which also follows: source tasks have
no ancestors either, so conditioning has nothing extra to offer.

### 7.5 Explainability

| Method | Accuracy over 291 incidents |
|---|---|
| Integrated Gradients | **0.935** |
| Reconstruction error | 0.924 |

Both are high and the gap is small. Integrated Gradients is clearly better on
`timezone_shift` (0.95 against 0.76), which fits its extra capability: it can
credit evidence in other tasks and earlier runs, while reconstruction error can
only inspect the flagged task at the flagged timestep. A timestamp shift shows up
as a relationship across the window rather than as one large residual. Nine of
fifteen fault types reach 1.00 under both.

Since Integrated Gradients costs 32 extra forward and backward passes per
explanation for a one-point average gain, reconstruction error is the sensible
default, with Integrated Gradients reserved for the fault families where it
demonstrably helps.

**The attention recovered the pipeline.** Averaged over incidents, structural
attention places 46.4% of its off-diagonal mass on true graph edges against
22.2% expected by chance, a 2.1x lift. The brightest cells are genuine
parent-child pairs. Nothing forces this, since the graph bias is a learned scalar
per relation rather than a mask.

![**Figure 6.** Structural attention averaged over incidents, row attending to column. The brightest cells are true parent-child pairs in the pipeline. Section 8.1 shows this alignment survives removing the DAG bias entirely, which is why the bias turns out to be redundant.](artifacts/results/fig7_attention.png)

![**Figure 7.** Attribution accuracy per fault type for both explanation methods, scored against each fault's mechanistically expected symptom family.](artifacts/results/fig8_attribution.png)


---

## 8. Ablation Study

Every configuration received an identical budget, 40 epochs with the same
early-stopping patience of 8, so "this component helps" cannot be confused with
"this run trained longer". One seed stopped at epoch 33 on that rule; since
checkpoints are selected on best validation loss rather than the final epoch,
that neither helps nor hurts it. Three seeds of the full model establish the
noise floor.

**Baseline, three seeds:** AUPRC 0.951 ± 0.001, Hit@1 0.805 ± 0.011, MRR 0.887 ± 0.007.

| Configuration | val recon | AUPRC | Hit@1 | MRR | Delta Hit@1 | Verdict |
|---|---|---|---|---|---|---|
| Full model (3 seeds) | 0.438 | 0.951 | 0.805 | 0.887 | reference | |
| Clean training set | 0.436 | 0.952 | 0.814 | 0.892 | +0.009 | within noise |
| No DAG attention bias | 0.435 | 0.953 | 0.801 | 0.884 | -0.005 | within noise |
| No calendar features | 0.433 | 0.953 | 0.790 | 0.879 | -0.015 | within noise |
| Contrastive objective added | 0.486 | 0.953 | 0.787 | 0.880 | -0.018 | within noise |
| Working calendar features (v2 data) | 0.433 | 0.953 | 0.787 | 0.878 | -0.018 | within noise |
| Longer training (57 epochs, budget lifted) | **0.405** | 0.948 | **0.756** | 0.861 | **-0.049** | **outside noise** |

### 8.1 None of the architectural choices matter

Four of the five variations land inside two standard deviations of the seed
noise. Neither the DAG attention bias nor the calendar features nor the
contrastive objective changes the outcome, and training on a guaranteed-clean
set is no better than training on the contaminated history.

Each of these deserves a specific explanation rather than a shrug.

**The DAG attention bias is redundant because the model finds the graph anyway.**
Attention in the ablated model still places 42.2% of its off-diagonal mass on
true pipeline edges against 22.2% expected by chance, a 1.9x lift, compared with
46.4% and 2.1x when the bias is present. Telling the model the structure nudges
it slightly and changes nothing downstream, because the structure is recoverable
from the data.

**Calendar features are redundant, and this took two attempts to establish.**
The first version of this ablation was not a valid test. A unit bug made every
exported timestamp 1000x too small, so the whole 8000-run timeline spanned nine
distinct hours instead of twenty-four and the calendar features carried almost no
signal. Removing features that encode nothing is guaranteed to change nothing.

After fixing the timestamps and retraining with a genuine 24-hour cycle, the
answer is the same: Hit@1 0.787 against a seed mean of 0.805 +/- 0.011, and
validation reconstruction of 0.4332 against 0.4328. So the conclusion stands and
now has a real experiment behind it. A 32-hour window spans more than a full day,
so the model can already infer the phase of the daily cycle from the history it
sees, and an explicit clock adds nothing.

**The contrastive objective hurts reconstruction and does not help anything else.**
Validation reconstruction loss rises from 0.438 to 0.486 while localization is
unchanged. The contrastive loss itself collapses to 0.055, meaning the task
became trivial: with augmentations deliberately kept weak, so as not to
manufacture something that looks like a real fault, the two views are too easy to
match and the objective stops providing signal. Stronger augmentation is not the
obvious fix, because that is precisely what would teach the model to treat real
faults as normal.

**Contamination tolerance is confirmed.** Training on only the clean windows gives
+0.009 Hit@1, inside noise. The claim that the method tolerates the roughly 3%
incident contamination present in real pipeline history holds, and it holds
without any special handling beyond robust normalisation.

### 8.2 Better reconstruction makes localization worse

The one result outside the noise floor is the one we did not go looking for.
Training for 57 epochs instead of 33 improves validation reconstruction loss
substantially, from 0.433 to 0.405, and makes localization worse, from 0.805 to
0.756 Hit@1.

This is the over-generalisation failure familiar from reconstruction-based
anomaly detection. A model that reconstructs well enough starts reconstructing
anomalies too, and the error signal that detection and localization both depend
on gets smaller. The contrastive run points the same way from the opposite
direction: its reconstruction is clearly worse at 0.486 and its localization is
unchanged.

Reconstruction loss and localization accuracy are therefore only loosely coupled,
and past a point they move in opposite directions.

This has an uncomfortable implication for the protocol used here. Checkpoints
were selected on validation reconstruction loss precisely because it needs no
labels, which keeps the method honestly self-supervised. That criterion is now
shown to be poorly aligned with the task the model is being asked to perform,
and a stopping point chosen for methodological purity is not the one that
maximises the metric anyone cares about. Selecting on validation AUPRC would fix
the alignment and break the label-free claim.

Resolving this properly needs a label-free proxy that correlates with
localization rather than with reconstruction, and finding one is left as future
work. Meanwhile the honest position is that the reported numbers come from a
selection rule that is defensible but demonstrably suboptimal, which if anything
understates what the architecture can do.

---

## 9. Error Analysis

Nine of fifteen fault types localize at or near 1.00. Two fail badly.

| Fault type | Hit@1 | Mean tasks affected |
|---|---|---|
| `row_loss_truncation` | 0.29 | 3.77 |
| `duplicate_records` | 0.43 | 2.84 |
| Nine types at 0.94 to 1.00 | | 1.00 to 1.76 |

The correlation between how far a fault spreads and how well it localizes is
**-0.63**. Faults that stay local are easy and faults that travel are hard,
which is the expected shape of the problem.

The error pattern is sharper than the average suggests. Of the mislocalized
`duplicate_records` incidents, **100%** blamed a task further downstream, and
nearly all blamed the same one, `join_payments`. For `row_loss_truncation` the
figure is 85%.

The mechanism is specific to joins. A join's output row count depends on the key
overlap between its inputs, and key overlap is not among the 38 features, so no
model can predict a join's row count from its parents' profiles. When duplication
arrives from upstream, the fan-out it causes at the join is genuinely
unpredictable from the parent profiles, and the join therefore looks like the
origin.

This is a feature-design limitation rather than a modelling one, and it names its
own fix: add a join-key overlap statistic to the profile. The information the
model would need is currently not measured at all.

Localization also degrades gracefully with severity, from 0.66 Hit@1 in the
lowest quartile to 0.88 in the highest. A method that scored faint incidents as
well as severe ones would be suspicious rather than impressive.

![**Figure 8.** Localization accuracy per fault type and by severity quartile. The two row-count faults at the bottom are the failure mode analysed above, and accuracy rising with severity is the expected and desirable shape.](artifacts/results/fig6_localization_breakdown.png)


---

## 10. Limitations and Threats to Validity

**The benchmark is synthetic.** Fault severities, frequencies and the mix of
types are design choices. The relative ordering of methods is more trustworthy
than any absolute number reported here. Validating the simulator against a real
pipeline's telemetry is the single most valuable piece of future work.

**Detection may be too easy to discriminate.** Five of seven methods land within
0.03 AUPRC of each other. The benchmark separates methods on localization but
barely on detection. Lower severities, more concurrent incidents and shorter
windows would all make it harder.

**Alert-budget columns saturate.** F1 at a 1% budget is capped at 0.129 and at a
5% budget at 0.512 for any detector with perfect precision, because one cannot
recall more incidents than one is permitted to raise alerts. Only the 15% column,
roughly one alert per incident, discriminates between methods.

**Three seeds.** Enough to establish that seed variance is small (Hit@1 standard
deviation 0.011) but not enough for a formal significance test between methods.
Differences under about 0.05 Hit@1 should not be treated as real, and on that
threshold every architectural ablation in Section 8 is a null result.

**Model selection is not entirely label-free, and the label-free part is
misaligned.** Checkpoints are chosen on validation reconstruction loss, which
uses no labels, but Section 8.2 shows that criterion does not track localization
and can actively work against it. The detection pooling rule is chosen on
validation AUPRC, one categorical choice among five options, reported rather
than hidden.

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

**Single pipeline topology.** All results come from one nine-task graph.
Generalisation to wider graphs, deeper graphs, or pipelines with cycles in their
scheduling is untested.

---

## 11. Conclusion and Future Work

This project set out to detect ETL failures without labels and to identify which
task caused them. It delivers a benchmark that makes the second question
answerable, a self-supervised model that answers both competitively, and an
explanation layer whose correctness can be measured rather than assumed.

The honest summary of the modelling result is that the transformer does not beat
simple baselines on this benchmark. Local Outlier Factor detects better and an
LSTM autoencoder localizes better, and the differences among the leading methods
are within noise. What the work does establish is more specific and, arguably,
more useful: **the way per-task scores are made comparable matters more than the
model that produces them.** Correcting for cross-task comparability is worth up
to 18.5 Hit@1 points, whereas the entire spread between the leading detectors is
about 4. Two routes to that correction, one structural and one statistical, prove
equivalent in effect.

Four design components were tested and none of them helped. Reporting that is
part of the result rather than an embarrassment. A graph attention bias adds
nothing because the model recovers the pipeline structure from data on its own,
explicit calendar features add nothing because a 32-hour window already contains
more than a full daily cycle, a contrastive objective adds nothing and degrades
reconstruction, and a guaranteed-clean training set is no better than the
contaminated history, which confirms the contamination-tolerance claim.

The one effect that exceeded the noise floor was unwelcome: training longer
improves reconstruction and makes localization worse. That is the
over-generalisation failure of reconstruction-based detection, and it undermines
the label-free checkpoint criterion used throughout, since validation
reconstruction loss turns out to be poorly aligned with the task the model is
actually being asked to do.

### Future work

**Measure join-key overlap.** Section 9 identifies the exact missing feature
behind the two worst-localizing fault types. This is the highest-value next step
and it is cheap.

**Validate the simulator against real telemetry.** Even a few weeks of real
pipeline metrics with a handful of known incidents would let the fault severity
and frequency distributions be calibrated rather than chosen.

**Harder detection settings.** The benchmark should be extended with lower
severities and more concurrent incidents until it separates detectors as clearly
as it separates localizers.

**A label-free selection criterion that tracks localization.** Section 8.2 shows
validation reconstruction loss is a poor proxy. Finding a proxy that is both
label-free and aligned with the downstream task would improve every number here
and is the most interesting open problem the project surfaced.

**Few-shot fault typing.** The representation already supports a fault-type
probe. How many labelled examples per class are needed to classify the fifteen
types is a natural and practical question.

**Online adaptation.** Real pipelines change. A production system would need to
distinguish a genuine fault from a deliberate schema migration, which is a
concept-drift problem this work does not address.

---

## References

Verify venue and year for each of these before submission.

1. P. Malhotra, L. Vig, G. Shroff, P. Agarwal. Long Short Term Memory Networks
   for Anomaly Detection in Time Series. ESANN, 2015.
2. P. Malhotra et al. LSTM-based Encoder-Decoder for Multi-sensor Anomaly
   Detection. ICML Anomaly Detection Workshop, 2016.
3. L. Ruff et al. Deep One-Class Classification. ICML, 2018.
4. K. Hundman et al. Detecting Spacecraft Anomalies Using LSTMs and
   Nonparametric Dynamic Thresholding. KDD, 2018.
5. Y. Su et al. Robust Anomaly Detection for Multivariate Time Series through
   Stochastic Recurrent Neural Network. KDD, 2019.
6. J. Audibert et al. USAD: UnSupervised Anomaly Detection on Multivariate Time
   Series. KDD, 2020.
7. H. Zhao et al. Multivariate Time-series Anomaly Detection via Graph Attention
   Network. ICDM, 2020.
8. A. Deng, B. Hooi. Graph Neural Network-Based Anomaly Detection in
   Multivariate Time Series. AAAI, 2021.
9. S. Tuli, G. Casale, N. Jennings. TranAD: Deep Transformer Networks for
   Anomaly Detection in Multivariate Time Series Data. VLDB, 2022.
10. J. Xu et al. Anomaly Transformer: Time Series Anomaly Detection with
    Association Discrepancy. ICLR, 2022.
11. G. Zerveas et al. A Transformer-based Framework for Multivariate Time Series
    Representation Learning. KDD, 2021.
12. Z. Yue et al. TS2Vec: Towards Universal Representation of Time Series.
    AAAI, 2022.
13. K. He et al. Masked Autoencoders Are Scalable Vision Learners. CVPR, 2022.
14. M. Du et al. DeepLog: Anomaly Detection and Diagnosis from System Logs
    through Deep Learning. CCS, 2017.
15. W. Meng et al. LogAnomaly: Unsupervised Detection of Sequential and
    Quantitative Anomalies in Unstructured Logs. IJCAI, 2019.
16. H. Guo, S. Yuan, X. Wu. LogBERT: Log Anomaly Detection via BERT. IJCNN, 2021.
17. S. Schelter et al. Automating Large-Scale Data Quality Verification.
    VLDB, 2018.
18. E. Breck et al. Data Validation for Machine Learning. SysML, 2019.
19. J. Song, Y. He. Auto-Validate: Unsupervised Data Validation Using
    Data-Domain Patterns. SIGMOD, 2021.
20. S. Redyuk et al. Automating Data Quality Validation for Dynamic Data
    Ingestion. EDBT, 2021.
21. L. Wu et al. MicroRCA: Root Cause Localization of Performance Issues in
    Microservices. NOMS, 2020.
22. M. Li et al. Causal Inference-Based Root Cause Analysis for Online Service
    Systems with Intervention Recognition. KDD, 2022.
23. A. Ikram et al. Root Cause Analysis of Failures in Microservices through
    Causal Discovery. NeurIPS, 2022.
24. M. Sundararajan, A. Taly, Q. Yan. Axiomatic Attribution for Deep Networks.
    ICML, 2017.
25. S. Lundberg, S. Lee. A Unified Approach to Interpreting Model Predictions.
    NeurIPS, 2017.
26. S. Kim et al. Towards a Rigorous Evaluation of Time-series Anomaly
    Detection. AAAI, 2022.
27. R. Wu, E. Keogh. Current Time Series Anomaly Detection Benchmarks are
    Potentially Flawed. TKDE, 2022.
28. F. Liu, K. Ting, Z. Zhou. Isolation Forest. ICDM, 2008.
29. M. Breunig et al. LOF: Identifying Density-Based Local Outliers.
    SIGMOD, 2000.

---

## Appendix A: Reproducing every number

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Dataset, about 4 minutes
PYTHONPATH=src python -m etlguard.data.build_dataset --n-batches 8000 --out data/etl_v1

# Simulator validation and descriptive figures
PYTHONPATH=src python scripts/validate_simulator.py
PYTHONPATH=src python scripts/dataset_report.py --data data/etl_v1

# Pretraining, about 18 minutes on Apple Silicon
PYTHONPATH=src python -m etlguard.train.pretrain --epochs 40 --out runs/ssl

# Main evaluation with all baselines
PYTHONPATH=src python -m etlguard.eval.run_eval --ckpt runs/ssl/best.pt --save-scores

# Seeds and ablations, about two hours
PYTHONPATH=src python scripts/run_experiments.py --only all

# Attribution, error analysis, tables and figures
PYTHONPATH=src python scripts/attribution_study.py --ckpt runs/ssl/best.pt
PYTHONPATH=src python scripts/error_analysis.py --ckpt runs/ssl/best.pt
PYTHONPATH=src python scripts/results_report.py

# Tests
PYTHONPATH=src python -m pytest tests/ -q
```
