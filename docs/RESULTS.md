# Results

All numbers are on the **test split** (2000 runs, the last 25% of the timeline,
291 of them carrying an incident). Nothing here was tuned on test. The model was
trained for 40 epochs with the checkpoint chosen on validation reconstruction
loss, which uses no labels.

Regenerate everything with:

```bash
PYTHONPATH=src python -m etlguard.eval.run_eval --ckpt runs/ssl/best.pt
PYTHONPATH=src python scripts/results_report.py
```

---

## 1. Main comparison

| method | AUROC | AUPRC | best F1 | F1 @1% | F1 @5% | F1 @15% | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| LOF | 0.983 | **0.973** | **0.963** | 0.129 | 0.512 | **0.931** | 0.797 | 0.952 | 0.880 |
| Dense autoencoder | 0.975 | 0.956 | 0.910 | 0.129 | 0.512 | 0.893 | 0.725 | 0.911 | 0.828 |
| **SSL transformer (ours)** | 0.974 | 0.953 | 0.935 | 0.129 | 0.512 | 0.890 | 0.773 | **0.955** | 0.867 |
| LSTM autoencoder | 0.974 | 0.951 | 0.928 | 0.129 | 0.512 | 0.887 | 0.643 | 0.969 | 0.803 |
| PCA residual | 0.968 | 0.942 | 0.928 | 0.129 | 0.512 | 0.876 | **0.801** | 0.966 | **0.886** |
| Rolling z-score | 0.730 | 0.502 | 0.519 | 0.129 | 0.512 | 0.399 | 0.663 | 0.959 | 0.805 |
| Isolation Forest | 0.779 | 0.466 | 0.469 | 0.116 | 0.327 | 0.450 | 0.323 | 0.687 | 0.543 |

The 1% and 5% budget columns are at their analytic ceiling (0.129 and 0.512) for
every method with perfect precision, so they separate nothing. That is a property
of the base rate, not of the detectors: you cannot recall more incidents than you
are permitted to raise alerts. The 15% column, roughly one alert per incident, is
the one that discriminates.

### What this says, plainly

**The self-supervised model does not win outright, and the write-up should say
so.** LOF has the best detection (AUPRC 0.973 against 0.953) and PCA residual has
the best Hit@1 (0.801 against 0.773). Both are simple, both are fast, and on this
benchmark both are strong.

That result is worth understanding rather than tuning away. Each task's profile
is only 38 dimensions and the healthy manifold is densely sampled by 4800
training runs, which is close to the ideal setting for a density or subspace
method. The deep models buy little on detection because there is not much left
to buy: five of the seven methods sit between 0.94 and 0.97 AUPRC.

The honest headline is therefore not "our model is best". It is that **the
ancestor-conditioned scoring rule is what improves localization**, and that is
shown below on the same trained model, holding everything else fixed.

Two baselines deserve comment. Rolling z-score, the stand-in for what a
Great Expectations or Deequ style tool does today, collapses on detection
(AUPRC 0.502) because the benign distractors defeat it: seasonality, growth and
campaigns move features as hard as several fault types do. Isolation Forest is
the weakest overall, which is a known weakness of axis-parallel splits on
correlated features.

---

## 2. The main finding: how you score matters more than what you score with

Same trained model, same test split, only the scoring rule changed.

| scoring rule | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| Ancestor-conditioned (hide the node **and its descendants**) | **0.773** | 0.955 | **0.867** |
| Self-masked only (hide just the node) | 0.632 | 0.945 | 0.787 |
| Deviation magnitude (history-only error) | 0.588 | 0.931 | 0.762 |

Ancestor conditioning adds **18.5 Hit@1 points** over ranking by deviation
magnitude on the very same model. That is a larger gap than anything separating
the seven methods in the main table, which is the point: the scoring rule is
doing more work here than the choice of architecture.

The middle row exists because it was the first implementation, and it was wrong.
Hiding only the node under test does not measure surprise given inputs, because
structural attention is bidirectional and the node's own descendants stay
visible. During an incident those descendants are corrupted too, so they hand the
corrupted value back to the model. Hiding the descendants as well leaves only
ancestors and history, which is the intended quantity.

### The depth breakdown confirms the mechanism

If the story above is right, the leak should hurt most where a node has the most
descendants, meaning upstream, and should vanish entirely at the sink.

| origin depth | ancestor-conditioned | self-masked only | deviation magnitude | n |
|---|---|---|---|---|
| 0 (source tasks) | 0.598 | 0.546 | 0.505 | 97 |
| 1 | **0.806** | 0.583 | 0.556 | 72 |
| 2 | **0.821** | 0.643 | 0.643 | 28 |
| 3 | 0.889 | 0.889 | 0.667 | 18 |
| 4 | **0.938** | 0.625 | 0.562 | 48 |
| 5 (sink, `load_warehouse`) | 0.893 | 0.893 | 0.893 | 28 |

At depth 5 all three rules agree exactly, because `load_warehouse` has no
descendants and the three masks are then identical. That is not a tuned result,
it falls out of the construction, and it is the strongest evidence that the gain
comes from the mechanism claimed and not from somewhere else.

Depth 0 is the hardest row and the gain there is smallest, which also follows:
source tasks have no ancestors either, so conditioning has nothing extra to
offer and the score falls back on history alone.

---

## 3. Where localization fails

| fault type | Hit@1 | n |
|---|---|---|
| type_mismatch, schema_column_add, resource_degradation, late_arrival, join_fanout | 1.00 | 5 to 21 each |
| unit_change | 0.95 | 20 |
| pk_null | 0.94 | 17 |
| schema_column_drop | 0.89 | 44 |
| category_corruption | 0.86 | 21 |
| timezone_shift | 0.76 | 21 |
| null_spike | 0.72 | 18 |
| fk_break | 0.67 | 6 |
| value_corruption | 0.64 | 11 |
| duplicate_records | 0.43 | 35 |
| row_loss_truncation | **0.29** | 28 |

The two failures are both row-count faults, and they fail for the same reason.
A change in row count propagates through every downstream task almost
undiminished: if `clean_orders` loses 40% of its rows, then the join, the
aggregate and the warehouse load all lose about 40% too. Every node on the path
shows an almost identical `row_ratio` anomaly, so there is very little asymmetry
for the ancestor-conditioned score to exploit. The measured propagation reach
backs this up, at 3.77 nodes for truncation and 2.84 for duplication, against
1.00 for the faults that localize perfectly.

This is a genuine limitation rather than a tuning problem, and it points at the
most promising next step: row counts are conserved through the DAG in a way most
other features are not, so a task-specific conservation prior (expected row ratio
per task, learned from healthy history) should recover exactly the asymmetry that
is missing.

Localization also degrades gracefully with severity, from 0.66 Hit@1 in the
lowest quartile to 0.88 in the highest, which is what should happen. Low-severity
incidents are genuinely close to invisible, and a method that scored them as well
as severe ones would be suspicious.

---

## 4. Attribution

Attribution accuracy is **0.924** over 291 incidents, meaning that in 92% of
cases the highest-scoring symptom family at the identified origin matched the
family that fault type is mechanistically expected to move.

This number is only meaningful because the simulator defines the ground truth. In
the published literature attribution is almost always displayed rather than
scored, precisely because there is nothing to check it against.

---

## 5. Detection pooling rule

Chosen on validation AUPRC, before touching test:

| rule | val AUPRC |
|---|---|
| `max_anc` (selected) | 0.946 |
| `max_ctx` | 0.944 |
| `max_hist` | 0.936 |
| `top2_hist` | 0.911 |
| `mean_hist` | 0.885 |

The spread is small, so detection is not sensitive to this choice. Localization,
as Section 2 shows, very much is.

---

## 6. Threats to validity

**The benchmark is synthetic.** Fault severities, frequencies and the mix of
fault types are all design choices. The relative ordering of methods is more
trustworthy than any absolute number here.

**Detection may be too easy.** Five of seven methods land within 0.03 AUPRC of
each other, which suggests the benchmark does not discriminate well on detection
even though it clearly does on localization. Harder settings worth running are
lower severities, more concurrent incidents, and shorter windows.

**One seed.** Every number is from a single training run. Repeats with different
seeds and a variance estimate are needed before any of these gaps is called real,
and the smaller ones (0.953 against 0.956, say) almost certainly are not.

**Model selection.** Checkpoints were chosen on validation reconstruction loss,
which uses no labels, but the detection pooling rule was chosen on validation
AUPRC. That is one categorical choice out of five options and it is reported
rather than hidden, but it is not zero label use.
