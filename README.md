# Self-Supervised AI for ETL Error Detection and Root-Cause Localization

A deep-learning system that watches a data pipeline's runs, learns what "healthy"
looks like **without any labels**, and when something breaks answers two
questions instead of one:

1. **Detection** — is this pipeline run broken?
2. **Root-cause localization** — *which task* broke it, and *which symptom* gives
   it away?

The second question is the hard one. When a source system starts emitting nulls,
every downstream task looks wrong: the join loses rows, the aggregate under-reports
revenue, the warehouse load looks thin. A detector that flags nine unhealthy tasks
has told an on-call engineer nothing. The goal here is to point at the *one* task
where the incident originated.

---

## Why self-supervised?

Labelled ETL failures barely exist. Incidents are rare, inconsistently recorded,
and a team that has just survived one is not going to annotate it. Any method that
needs a labelled corpus of "1000 examples of schema drift" is unusable in practice.

So the model is trained on **unlabelled pipeline telemetry** using masked
reconstruction: hide part of a run's data-quality profile and make the network
predict it from the surrounding context (other tasks, other timesteps). Learning to
do that well requires internalising the pipeline's normal structure — its seasonality,
its inter-task dependencies, the usual relationship between rows in and rows out.
Anomalies are then whatever that model cannot predict.

Labels exist in this repository, but only ever as an **evaluation instrument** and
as training data for the supervised baselines the method is compared against.

---

## The dataset problem, and how it is solved

There is no public benchmark of labelled ETL failures with root-cause annotations,
which is exactly why the literature keeps evaluating on generic multivariate
time-series datasets that have no pipeline structure at all.

This project builds one. `src/etlguard/simulator/` implements a **nine-task
e-commerce warehouse pipeline** and simulates it at the row level:

```
extract_customers ─┐
                   ├─ clean_customers ─┐
extract_orders ────┴─ clean_orders ────┴─ join_orders_customers ─┐
                                                                 ├─ join_payments ─ aggregate_daily ─ load_warehouse
extract_payments ────────────────────────────────────────────────┘
```

Every task runs a real transformation (cast, dedup, currency normalisation, joins,
group-by) over actual `pandas` rows. **Faults are injected into the rows, not into
the features.** Downstream tasks then consume the corrupted rows, so fault
propagation across the DAG is *emergent* rather than hand-coded — which is the
entire reason root-cause localization is a genuine problem here and not a
lookup table.

**15 fault types** across six families:

| Family | Faults |
|---|---|
| Schema | `schema_column_drop`, `schema_column_add`, `type_mismatch` |
| Completeness | `null_spike`, `pk_null`, `duplicate_records`, `row_loss_truncation` |
| Value / distribution | `unit_change`, `value_corruption`, `category_corruption` |
| Temporal | `timezone_shift`, `late_arrival` |
| Integrity | `fk_break`, `join_fanout` |
| Operational | `resource_degradation` |

`resource_degradation` is deliberately the hard case: the task blows its SLA while
its output data stays perfectly valid, so a data-quality-only detector must not
confuse it with corruption — and it must **not** propagate downstream.

The simulator also emits **benign distractors** that a naive threshold would
misfire on: daily and weekly seasonality, a growth trend, slow drift in the
customer-country mix, and marketing campaigns that lift volume ~1.7x for a day or
two. These are labelled *normal* on purpose.

Each task run is summarised into a **38-dimensional data-quality profile**
(`simulator/profiler.py`) covering operational telemetry, schema stability,
completeness, metric distribution, categorical distribution, freshness and
referential integrity. A pipeline run at time *t* is therefore a `[9, 38]` matrix,
and the model sees a window of them: `[W, 9, 38]`.

---

## Repository layout

```
src/etlguard/
  simulator/    schema.py  · table + column contracts, semantic column roles
                dag.py     · the nine tasks and their real transformations
                faults.py  · the 15 fault injectors and the propagation rules
                profiler.py· rows -> 38-dim data-quality profile
                generator.py· the simulation engine and fault scheduler
  data/         build_dataset.py · CLI that produces the dataset
                dataset.py       · windowing + leak-free robust normalisation
  models/       (chunk 2) masked-reconstruction encoder + baselines
  train/        (chunk 2) pretraining and evaluation loops
  eval/         (chunk 3) detection + localization metrics, attribution
scripts/        validate_simulator.py · fault -> feature response matrix
tests/          contract, healthy-baseline and fault-propagation tests
docs/           PROJECT_PLAN.md · full research design and phase plan
```

---

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generate the dataset (~6 minutes for 8000 hourly runs ≈ 11 months of history):

```bash
PYTHONPATH=src python -m etlguard.data.build_dataset --n-batches 8000 --out data/etl_v1
```

Check that every fault type actually moves the features it should:

```bash
PYTHONPATH=src python scripts/validate_simulator.py
```

Run the tests:

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

---

## Evaluation protocol

The split is **chronological**, never random — train on the past, test on the
future, because a random split over a time series leaks tomorrow into today.

* Train 60% / validation 15% / test 25%.
* Training history carries ~3% incident contamination, as real history does. The
  self-supervised model must tolerate it; that is part of the claim.
* At run *t* the model sees only runs `t-W+1 .. t`. Nothing from the future.
* Normalisation statistics (median/IQR, robust to the contamination) are fitted
  on the **training split only**.

---

## Status

- [x] **Chunk 1** — simulator, fault taxonomy, profiler, dataset builder, windowing
      and normalisation, test suite
- [ ] **Chunk 2** — self-supervised model, baselines, training loop, literature review
- [ ] **Chunk 3** — root-cause localization, attribution/explainability, ablations
- [ ] **Chunk 4** — results, figures, final report

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full design.
