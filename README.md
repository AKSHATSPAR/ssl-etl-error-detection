# Self-Supervised AI for ETL Error Detection and Root-Cause Localization

A deep learning system that watches a data pipeline run, learns what healthy
looks like without being given any labels, and when something breaks answers two
questions instead of one:

1. **Detection.** Is this pipeline run broken?
2. **Root-cause localization.** Which task broke it, and what kind of break is it?

The second question is the hard one, and it is the reason this project exists.
When a source system starts emitting nulls, every task downstream of it looks
wrong. The join loses rows, the aggregate under-reports revenue, the warehouse
load comes out thin. A detector that flags nine unhealthy tasks has told the
on-call engineer nothing useful. The goal here is to point at the one task where
the trouble actually started.

---

## Why self-supervised

Labelled ETL failures barely exist. Incidents are rare, they get recorded
inconsistently when they get recorded at all, and a team that has just spent a
night fixing one is not going to sit down and annotate it. Any method that needs
a thousand labelled examples of schema drift is a method nobody can use.

So the model trains on unlabelled pipeline telemetry using masked reconstruction.
Hide part of a run's data-quality profile, then make the network predict it from
what surrounds it, meaning the other tasks and the earlier timesteps. Doing that
well requires actually learning the pipeline's normal behaviour: its daily
rhythm, how tasks depend on each other, the usual relationship between rows in
and rows out. Anomalies are then simply the things the model cannot predict.

Labels do exist in this repository, but only as an evaluation instrument, and as
training data for the supervised baselines the method gets compared against.
Nothing in the training loop touches them, including the checkpoint selection.

---

## The dataset problem, and what was done about it

There is no public benchmark of labelled ETL failures with root-cause
annotations. That absence is also why the literature keeps evaluating on generic
multivariate time-series datasets that have no pipeline structure at all, which
means root-cause localization cannot even be posed on them.

So this project builds one. The simulator in `src/etlguard/simulator/` implements
a nine-task e-commerce warehouse pipeline:

```
extract_customers ─┐
                   ├─ clean_customers ─┐
extract_orders ────┴─ clean_orders ────┴─ join_orders_customers ─┐
                                                                 ├─ join_payments ─ aggregate_daily ─ load_warehouse
extract_payments ────────────────────────────────────────────────┘
```

Every task runs a genuine transformation over actual pandas rows: type casting,
deduplication, currency normalisation, joins, group-by. Faults are injected into
the rows, not into the features. Downstream tasks then consume those corrupted
rows, so propagation across the DAG emerges from the simulation instead of being
written by hand. This matters more than it might sound. If propagation were
scripted, the model would end up learning the propagation rules rather than
anything about data pipelines.

**Fifteen fault types** across six families:

| Family | Faults |
|---|---|
| Schema | `schema_column_drop`, `schema_column_add`, `type_mismatch` |
| Completeness | `null_spike`, `pk_null`, `duplicate_records`, `row_loss_truncation` |
| Value and distribution | `unit_change`, `value_corruption`, `category_corruption` |
| Temporal | `timezone_shift`, `late_arrival` |
| Integrity | `fk_break`, `join_fanout` |
| Operational | `resource_degradation` |

`resource_degradation` is deliberately the awkward one. The task blows its SLA
while its output data stays perfectly valid, so a detector that only looks at
data quality has to avoid confusing it with corruption, and it must not spread
downstream.

The simulator also produces **benign distractors** that a naive threshold would
fire on constantly: daily and weekly seasonality, a growth trend, slow drift in
the customer country mix, and marketing campaigns that push volume up by around
1.7x for a day or two. All of these are labelled normal, on purpose.

### Propagation labels are measured, not assumed

The obvious way to label which tasks an incident affected is to take the origin
plus every descendant. That turns out to be wrong, and provably so. An unexpected
extra column is ignored by every downstream `SELECT`. A task that runs slowly
still emits correct rows. Labelling those descendants anomalous would penalise a
detector for being right about them.

So the simulator re-runs every faulty batch a second time with the fault removed,
holding everything else identical: the same source rows, the same telemetry
noise, the same scheduling jitter. This works because each node draws its
randomness from its own generator seeded on `(seed, batch, node)`. A node counts
as affected only when its profile actually moved away from what that same batch
would otherwise have produced.

Measured reach comes out at 2.48 nodes out of 9 per incident, against 3.94 under
the topological assumption. `schema_column_add` correctly reaches exactly one.
Both labels are stored, so the difference can be reported as an ablation.

Each task run is then summarised into a 38-dimensional data-quality profile
covering operational telemetry, schema stability, completeness, metric
distribution, categorical distribution, freshness and referential integrity. A
pipeline run at time *t* is a `[9, 38]` matrix, and the model sees a window of
them shaped `[W, 9, 38]`.

---

## The model

A factorised spatio-temporal transformer over the `[W, 9, 38]` window, at about
1.1M parameters.

Each encoder block attends along time first, so every task can look at its own
history, then across the DAG, so every timestep can look at the other tasks.
Doing it this way costs `O(W² + N²)` instead of the `O((WN)²)` you would pay for
joint attention, and the two axes carry different kinds of structure anyway.

Structural attention is biased by the DAG. Every ordered pair of tasks is
classified by relationship (direct parent, direct child, distant ancestor,
distant descendant, unrelated), and the model learns one attention bias per
relationship per head. It decides for itself how much a task should listen to its
immediate upstream rather than being forced by a hard mask.

Hour-of-day and day-of-week go in as cyclic features. Without them the largest
reconstruction error in the whole dataset is just "it is 3am", and the model
spends its capacity rediscovering the daily cycle instead of learning anything
about failures.

### Two masking strategies, and why both

**Feature-group masking** hides one whole symptom family at a task and timestep.
Masking scattered individual numbers would be far too easy, because
`null_rate_mean` is trivially recoverable from `null_rate_max` sitting next to it
in the same vector. The model would learn within-vector correlations and never
look at the pipeline at all. Hiding an entire family forces it to use other tasks
and other timesteps.

**Node masking** hides a task's entire profile at a timestep. To fill that in,
the model has to predict what a task should have produced given what its parents
produced. This is the objective the root-cause scorer is built on.

### How localization actually works

Scoring mirrors training. The model was trained to fill in hidden entries, so at
inference it gets asked the same kind of question. Running a masked autoencoder
unmasked lets it copy its input to its output, and the reconstruction error then
measures nothing at all.

Two masking configurations at the final timestep:

- **`e_hist`** hides every task, so each prediction rests on history alone. This
  is large whenever a run departs from its own past, which makes it the detection
  signal.
- **`e_ctx`** hides only task *i*, leaving every other task at that timestep
  visible, parents included. This is surprise given inputs.

The second one is the whole idea. A task downstream of a real failure produces
anomalous output, but output that follows perfectly from its own anomalous
inputs, so `e_ctx` stays small and it reads as a victim. The task where the
incident started has healthy inputs and anomalous output, so `e_ctx` stays large.
That asymmetry is what separates the origin from its blast radius, and it is
something a deviation-magnitude ranking cannot do, because propagation often
makes the symptom bigger further from the cause.

---

## Repository layout

```
src/etlguard/
  simulator/    schema.py      table and column contracts, semantic column roles
                dag.py         the nine tasks and their real transformations
                faults.py      the fifteen fault injectors and symptom ground truth
                profiler.py    rows to a 38-dim data-quality profile
                generator.py   simulation engine, fault scheduler, counterfactuals
  data/         build_dataset.py  CLI that produces the dataset
                dataset.py        windowing, leak-free normalisation, measured labels
  models/       layers.py         attention blocks and DAG relation bias
                st_transformer.py the encoder
                masking.py        the two masking strategies
                baselines.py      classical and deep baselines
  train/        pretrain.py       self-supervised pretraining
  eval/         scoring.py        detection and root-cause scoring
                metrics.py        detection, localization and attribution metrics
                run_eval.py       the full comparison
scripts/        validate_simulator.py  fault to feature response matrix
                dataset_report.py      descriptive figures
tests/          contracts, healthy baselines, fault propagation, counterfactuals
docs/           PROJECT_PLAN.md, LITERATURE_REVIEW.md
```

---

## Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generate the dataset. This takes about four minutes for 8000 hourly runs, which
is roughly eleven months of pipeline history.

```bash
PYTHONPATH=src python -m etlguard.data.build_dataset --n-batches 8000 --out data/etl_v1
```

Pretrain the model, around 20 minutes on Apple Silicon:

```bash
PYTHONPATH=src python -m etlguard.train.pretrain --epochs 40 --out runs/ssl
```

Evaluate against every baseline:

```bash
PYTHONPATH=src python -m etlguard.eval.run_eval --ckpt runs/ssl/best.pt
```

Descriptive figures, the fault response matrix, and the tests:

```bash
PYTHONPATH=src python scripts/dataset_report.py --data data/etl_v1
PYTHONPATH=src python scripts/validate_simulator.py
PYTHONPATH=src python -m pytest tests/ -q
```

---

## Evaluation protocol

The split is chronological, never random. Train on the past and test on the
future, because a random split over a time series leaks tomorrow into today.

- Train 60%, validation 15%, test 25%.
- The training history carries about 3% incident contamination, the way real
  history does. The self-supervised model has to tolerate that, and being able to
  is part of the claim.
- At run *t* the model sees only runs `t-W+1` through `t`. Nothing from the future.
- Normalisation statistics use median and IQR, robust to the contamination, and
  are fitted on the training split alone.
- Checkpoints are selected on validation reconstruction loss, which needs no
  labels. Selecting on validation AUPRC would quietly make this weakly
  supervised and inflate everything downstream.
- The only thing chosen with labels is the detection pooling rule, one choice out
  of four, made on validation. It is stated in the results rather than buried.
- Point-adjusted F1 is not the headline metric. Under point adjustment, catching
  a single timestep inside an incident credits you with the whole incident, and
  published work has shown random scores beating real models. It is reported
  alongside the honest number and labelled as such.

---

## Headline results

Test split, 2000 runs, 291 with incidents. Full detail and confidence intervals
in [docs/RESULTS.md](docs/RESULTS.md).

**The self-supervised model does not win, on either task, and the write-up says
so.** LOF has the best detection (0.973 AUPRC against 0.953) and an LSTM
autoencoder has the best localization (0.818 Hit@1 against 0.797). With 291
incidents the standard error on a Hit@1 near 0.8 is 0.023, so the top four
localizers are tied rather than ranked.

The finding that survived scrutiny is about scoring rather than architecture.
Ranking nine raw errors asks which task has the largest error, when the question
is which task is furthest from its own normal. There are two ways to fix that,
and they turn out to be worth exactly the same:

| SSL localizer variant | Hit@1 [95% CI] |
|---|---|
| Deviation magnitude, raw | 0.588 [0.533, 0.646] |
| Deviation magnitude, **statistically** calibrated | 0.773 [0.725, 0.821] |
| Ancestor-conditioned, raw (**structural**) | 0.773 [0.725, 0.821] |
| Both | 0.797 [0.749, 0.842] |

Rows two and three are identical to three decimals. Conditioning on a task's
parents and comparing a task to its own history are different mechanisms that
achieve the same comparability, and combining them adds little.

Chunk 2 of this project claimed ancestor conditioning was worth 18.5 points. That
compared it against an *uncalibrated* baseline. Corrected, it is worth 2.4 points
and the intervals overlap.

Other results:

- Attribution accuracy is 0.935 with Integrated Gradients and 0.924 from plain
  reconstruction error, both scored against the fault's known symptom family.
- Structural attention puts 46.4% of its off-diagonal mass on true DAG edges
  against 22.2% by chance, a 2.1x lift, with nothing forcing it to.
- Localization fails on two row-count faults, and the reason is specific: a
  join's output row count depends on key overlap between its inputs, which is not
  in the feature set, so no model can predict it from the parents. 100% of
  mislocalized `duplicate_records` incidents blamed a task further downstream.
  The fix is a feature, not a model.

## Status

- [x] **Chunk 1.** Simulator, fault taxonomy, profiler, dataset builder,
      windowing and normalisation, test suite.
- [x] **Chunk 2.** Model, self-supervised objectives, training loop, baselines,
      evaluation harness, literature review.
- [ ] **Chunk 3.** Ablations, Integrated Gradients attribution, error analysis.
- [ ] **Chunk 4.** Final results, figures, written report.

## Documents

| Document | What it is |
|---|---|
| [Report](artifacts/report/ETL_Error_Detection_Report.docx) | The full write-up, 7400 words with 8 figures |
| [Literature review](artifacts/report/Literature_Review.pdf) | 27 papers, review matrix, themes, gap analysis |
| [Complete explainer](artifacts/report/Complete_Explainer.pdf) | Every concept, decision, result and mistake, in one place |
| [Presentation and demo guide](artifacts/report/Presentation_and_Demo_Guide.pdf) | How to run the demo and explain every part in plain words |
| [Speaking script](artifacts/report/Speaking_Script.pdf) | Word-for-word script for two presenters |
| [Slides](artifacts/slides/ETL_Project_Viva.pptx) | 18 slides with speaker notes |
| [docs/RESULTS.md](docs/RESULTS.md) | Every number, with confidence intervals |
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | The research design |

## Demo

One command, about ten seconds. Takes a real held-out incident and diagnoses it,
then prints the ground truth so the answer can be checked.

```bash
PYTHONPATH=src python scripts/demo.py --n 2
```

Or open [artifacts/web/pipeline_demo.html](artifacts/web/pipeline_demo.html) by
double-clicking it. One file, no server, no internet: pick a run, the pipeline
lights up, and the ground truth stays hidden behind a button.
