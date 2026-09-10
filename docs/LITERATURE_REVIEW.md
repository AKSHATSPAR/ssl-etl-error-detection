# Literature Review

*Self-Supervised AI for ETL Error Detection and Root-Cause Localization*

> **Before you submit this.** Every reference below is a real paper, but please
> confirm the exact venue, year and author list on Google Scholar or DBLP before
> the report goes in. Getting a year wrong in a viva is an easy thing to be
> caught on and an easy thing to prevent. Anything you cannot verify, drop.

This review follows the four steps in the guide provided: build a review matrix,
group the papers into themes, write comparatively rather than as a list of
summaries, and then state the gap the project fills.

---

## 1. Scope and search strategy

The project title sits at the meeting point of three research communities that
mostly do not read each other:

1. **Multivariate time-series anomaly detection**, which is where the deep
   learning is.
2. **Data quality and data validation for pipelines**, which is where the
   problem is.
3. **Root-cause analysis for distributed systems**, which is where the
   localization ideas are.

Papers were gathered from the major venues of each community (KDD, VLDB, SIGMOD,
ICLR, AAAI, ICDM, CCS, ASE) using combinations of the terms *self-supervised
anomaly detection*, *multivariate time series*, *data quality validation*, *data
pipeline monitoring*, *log anomaly detection* and *root cause localization*.
Work that only does univariate detection, or that assumes a fully labelled
training set, was left out, since neither assumption survives contact with a real
ETL pipeline.

---

## 2. Literature review matrix

| No. | Reference | Problem addressed | Dataset | Methodology | Reported result | Limitations | Key contribution |
|---|---|---|---|---|---|---|---|
| 1 | Malhotra et al., ESANN 2015 | Time-series anomaly detection | ECG, power demand | Stacked LSTM predictor, error modelled as Gaussian | Detects anomalies without labels | Univariate focus, no localization | Showed prediction error is a usable anomaly signal |
| 2 | Malhotra et al., ICML AD Workshop 2016 | Multi-sensor anomaly detection | Engine, power | LSTM encoder-decoder reconstruction | Works on unpredictable signals | Needs clean training data | Reconstruction beats prediction when signals are noisy |
| 3 | Ruff et al., ICML 2018 | One-class deep learning | MNIST, CIFAR | Deep SVDD, minimum-volume hypersphere | Strong on image one-class tasks | Prone to representation collapse | Made one-class objectives work end to end |
| 4 | Hundman et al., KDD 2018 | Spacecraft telemetry anomalies | SMAP, MSL | LSTM prediction, nonparametric thresholding | Deployed at NASA JPL | Thresholding is heuristic | Released two widely used benchmarks |
| 5 | Su et al., KDD 2019 | Multivariate server anomalies | SMD | Stochastic RNN with planar normalising flows (OmniAnomaly) | F1 around 0.86 on SMD | Slow, needs clean data | Added a probabilistic account of normality |
| 6 | Audibert et al., KDD 2020 | Fast unsupervised detection | SMD, SWaT, SMAP | Adversarially trained pair of autoencoders (USAD) | Comparable accuracy, far faster | Adversarial training is unstable | Made accuracy affordable at production scale |
| 7 | Zhao et al., ICDM 2020 | Correlations between sensors | SMAP, MSL, SMD | Two graph attention layers, one over features, one over time (MTAD-GAT) | Beats non-graph baselines | Graph is learned, not given | Showed inter-variable structure carries signal |
| 8 | Deng and Hooi, AAAI 2021 | Sensor anomaly detection with structure | SWaT, WADI | Learned sensor graph plus GNN forecasting (GDN) | State of the art on water testbeds | Deviation ranking only, no causal claim | Introduced graph deviation scoring for localization |
| 9 | Tuli et al., VLDB 2022 | Efficient transformer detection | Multiple public sets | Transformer with adversarial two-phase training (TranAD) | Faster training, competitive scores | Evaluated with point adjustment | Brought transformers into this problem properly |
| 10 | Xu et al., ICLR 2022 | Distinguishing anomalies from normal points | SMD, SWaT, PSM | Association discrepancy between prior and series attention | Very high reported F1 | Scores rest on point adjustment | Novel criterion beyond reconstruction error |
| 11 | Zerveas et al., KDD 2021 | Representation learning for time series | UEA archive | Masked-value transformer pretraining | Beats supervised training in low-label settings | Not built for anomaly detection | Showed masked modelling transfers to time series |
| 12 | Yue et al., AAAI 2022 | Universal time-series representations | 128 UCR sets | Hierarchical contrastive learning (TS2Vec) | Strong across many tasks | Contrastive views need care | Contrastive pretraining works for time series |
| 13 | He et al., CVPR 2022 | Scalable self-supervised vision | ImageNet | High-ratio masked autoencoding (MAE) | 87.8% ImageNet top-1 | Vision domain | Aggressive masking makes reconstruction a real task |
| 14 | Du et al., CCS 2017 | Anomaly detection from system logs | HDFS, OpenStack | LSTM over log key sequences (DeepLog) | F1 around 0.96 on HDFS | Needs a stable log template set | First strong deep model for log anomalies |
| 15 | Meng et al., IJCAI 2019 | Unstable log templates | HDFS, BGL | Template2Vec plus LSTM (LogAnomaly) | Handles new templates | Still sequence-only | Made log detection robust to drift |
| 16 | Guo et al., IJCNN 2021 | Self-supervised log detection | HDFS, BGL, Thunderbird | BERT-style masked log-key modelling (LogBERT) | Beats DeepLog | Logs only, no data content | Masked modelling applied to operational data |
| 17 | Schelter et al., VLDB 2018 | Declarative data quality at scale | Amazon production data | Constraint checks with incremental metrics (Deequ) | Runs on production volumes | Constraints are written by hand | Made data quality a first-class system |
| 18 | Breck et al., SysML 2019 | Data errors in ML pipelines | Google production | Schema inference plus skew and drift checks (TFDV) | Caught real production bugs | Rule based, no learned model | Named and organised the ETL error taxonomy |
| 19 | Song and He, SIGMOD 2021 | Removing manual rule writing | 7000 real columns | Mining data-domain patterns (Auto-Validate) | Far fewer false positives than hand rules | Single-column scope | Automated validation-rule discovery |
| 20 | Redyuk et al., EDBT 2021 | Quality checks on changing data | Public and industry data | ML-based thresholds over quality metrics | Adapts to drift | Detection only, no root cause | Learned thresholds beat static ones |
| 21 | Wu et al., NOMS 2020 | Root cause in microservices | Testbed traces | Attributed service graph plus random walk (MicroRCA) | Localizes faulty service | Needs service call traces | Graph walking for localization |
| 22 | Li et al., KDD 2022 | Root cause with interventions | Real service data | Causal graph plus regression hypothesis testing (CIRCA) | Beats correlation ranking | Needs a causal graph | Framed root cause as intervention recognition |
| 23 | Ikram et al., NeurIPS 2022 | Root cause without a given graph | Microservice benchmarks | Hierarchical causal discovery (RCD) | Scales to many metrics | Sensitive to independence tests | Causal discovery for failure localization |
| 24 | Sundararajan et al., ICML 2017 | Attributing model outputs | Various | Integrated Gradients | Satisfies sensitivity and implementation invariance | Baseline choice matters | Axiomatic feature attribution |
| 25 | Lundberg and Lee, NeurIPS 2017 | Unified explanation framework | Various | Shapley-value additive attribution (SHAP) | Consistent, model agnostic | Expensive to compute | Standard tool for feature attribution |
| 26 | Kim et al., AAAI 2022 | Are these benchmarks meaningful? | SMD, SWaT, SMAP, MSL | Re-evaluation under point adjustment | A random score beats published models | Critique, not a method | Showed the field's headline metric is broken |
| 27 | Wu and Keogh, TKDE 2022 | Benchmark quality | Common AD benchmarks | Manual inspection of datasets | Triviality, mislabelling, run-to-failure bias | Critique, not a method | Argued for better benchmark design |

---

## 3. Common themes

Reading across the matrix, the work falls into six groups.

**Group A: Reconstruction and prediction models.** Papers 1, 2, 5, 6.
The shared idea is to learn to reproduce normal behaviour and treat whatever the
model cannot reproduce as anomalous.

**Group B: Structure-aware models.** Papers 7, 8.
These add a graph over variables so the detector can use the fact that sensors,
or services, or tasks, influence each other.

**Group C: Transformer and attention models.** Papers 9, 10, 11.
Attention replaces recurrence, which brings longer context and, in principle,
interpretable attention maps.

**Group D: Self-supervised pretraining.** Papers 11, 12, 13, 16.
Masked modelling and contrastive learning, imported from language and vision, to
learn representations with no labels at all.

**Group E: Data quality and pipeline validation.** Papers 17, 18, 19, 20.
Systems work, focused on the actual ETL setting, but mostly rule-based rather
than learned.

**Group F: Root-cause localization and explainability.** Papers 21, 22, 23, 24, 25.
Going past "something is wrong" to "here is where, and here is why".

Papers 26 and 27 sit outside these groups. They are critiques of how the whole
field measures itself, and they shaped the evaluation protocol used here more
than any single method did.

---

## 4. Comparative synthesis

### Learning what normal looks like

The reconstruction line of work (Group A) rests on a single assumption: train on
normal data, and anything the model reconstructs badly is abnormal. Malhotra et
al. established that prediction error alone carries the signal, then found that
reconstruction holds up better when the series is noisy and genuinely
unpredictable, which is the more common case. OmniAnomaly extended this by making
normality probabilistic rather than a point estimate, so the model expresses how
confident it is. That gains accuracy at a real cost in speed, which is exactly
what USAD then attacked, using two adversarially trained autoencoders to reach
similar accuracy far faster.

What none of them question is the training data. All four assume a clean, or
near-clean, set of normal examples. In an ETL setting that assumption quietly
fails, because nobody knows which historical runs were healthy. The incidents
that were never noticed are still sitting in the history, and they are precisely
the ones a detector would need to learn to distrust.

### Adding structure, and stopping short of causality

Group A treats variables as an unordered bundle. MTAD-GAT and GDN both push back
on that, and both improve on the flat baselines, which is good evidence that
inter-variable structure carries real signal. The difference between them is
where the graph comes from. MTAD-GAT learns attention over features, GDN learns
an explicit sensor graph and then scores each node by how far it deviates from
what its neighbours predict.

GDN comes closest to what this project needs, and the gap between the two is
instructive. GDN's deviation score ranks sensors by anomaly magnitude, and its
authors describe this as localization. In a sensor network that is defensible.
In a pipeline it is not, because magnitude does not survive propagation. A five
percent null rate at extraction becomes a thirty percent revenue drop three tasks
later, so the largest deviation lands on the task furthest from the fault. Any
ranking by magnitude will systematically blame the wrong end of the DAG. Groups B
and F therefore do not connect: the structure-aware detectors have the graph but
rank by magnitude, and the root-cause work reasons about causality but operates
on service traces rather than on data content.

### Self-supervision, borrowed but not yet arrived

Group D is the newest thread and the most promising. Zerveas et al. showed masked
modelling helps on time series, TS2Vec showed contrastive learning does too, and
LogBERT showed masked pretraining transfers to operational data specifically. MAE
supplied the design lesson that matters most here, which is that the masking has
to be aggressive enough that the task cannot be solved by local interpolation.

None of this has reached ETL data quality. LogBERT is the nearest neighbour, and
it models log key sequences, so it sees that a job ran and in what order, but
never what the data looked like. A pipeline can execute flawlessly, emit a
perfect log, and load a table where every amount is off by a factor of a hundred.

### The systems community has the problem but not the model

Group E is where the actual problem lives. Deequ made data quality checks a
production concern, TFDV built the schema and skew machinery that most of the
industry now copies, and both encode a taxonomy of ETL failures that is more
grounded than anything in the ML papers. The limitation they share is that
somebody has to write the rules. Auto-Validate is the most serious attempt to
remove that burden and works well, but stays within a single column, so it cannot
see a fault that only appears as a relationship between two tables. Redyuk et al.
learn thresholds instead of fixing them, which handles drift, but the output is
still a flag on a metric rather than a diagnosis.

### How the field grades itself

Papers 26 and 27 deserve more attention than their citation counts suggest. Kim
et al. re-evaluated the standard benchmarks under point-adjusted F1, the metric
almost every paper in Groups A and C reports, and found that a randomly generated
score can beat published models. The reason is mechanical: point adjustment
credits you with an entire anomalous segment for detecting any single timestep
inside it, so on datasets with long anomalous stretches, alerting at random works.
Wu and Keogh reached a related conclusion from the data side, documenting
mislabelling and triviality in the benchmarks themselves.

Taken together, a large fraction of the reported progress in Groups A and C may
be an artefact of measurement. That is not a reason to discard the methods, but it
is a strong reason not to report point-adjusted F1 as a headline number, and to
be suspicious of any comparison that only exists in that metric.

---

## 5. Research gaps

### 5.1 Dataset gap

There is no public benchmark of ETL pipeline failures with root-cause labels.
Groups A, B and C evaluate almost entirely on SMD, SWaT, SMAP and MSL, which are
server and sensor telemetry. None has a task DAG, none has data-content quality
metrics, and none labels which component originated a failure. Group E works on
real pipeline data but that data is proprietary and unreleased.

The practical result is that root-cause localization in a data pipeline cannot
currently be posed as a benchmark task, let alone measured. Wu and Keogh's
criticism applies with extra force: the field is not just using flawed
benchmarks, it has none at all for this problem.

### 5.2 Methodological gap

Three specific things are missing.

Detection and localization are treated separately. Groups A to D detect, Group F
localizes, and the localization work needs service call traces that a data
pipeline does not produce.

Where localization does appear inside a detector, it is deviation ranking, which
propagation defeats for the reason given in Section 4.

Self-supervision has not been applied to data-quality profiles at all. Group D
has the machinery and Group E has the problem, and nobody has connected them.

### 5.3 Explainability gap

Explanation in Groups A to C means a reconstruction-error bar chart per variable.
That says which number moved, not what kind of failure occurred. Integrated
Gradients and SHAP are mature tools that could give a much better answer, but
they are rarely applied here, and when attribution is shown it is almost never
scored. Nobody asks whether the explanation is correct, because there is no
ground truth to check it against.

### 5.4 Application gap

Nearly all of this work outputs a binary flag. An on-call engineer at 3am needs
three things: whether the run is broken, which task broke it, and what kind of
break it is. No single system in the matrix produces all three.

There is also a blind spot around benign change. Real pipelines have seasonality,
growth and one-off events like marketing campaigns, all of which move volume
sharply and none of which is a fault. The public benchmarks contain little of
this, so published false-positive rates are optimistic.

### 5.5 Evaluation gap

Following papers 26 and 27, the standard protocol has three problems worth fixing
rather than inheriting: point-adjusted F1 inflates results to the point of
meaninglessness, training sets are assumed clean when real history is not, and
random rather than chronological splits let information from the future leak into
the past.

---

## 6. How this project addresses the gaps

| Gap | Response |
|---|---|
| Dataset (5.1) | A nine-task warehouse pipeline simulated at row level, with fifteen fault types injected into the rows so that propagation through the DAG is emergent rather than scripted. Every run is labelled with its origin task, its measured blast radius, and its fault type. |
| Methodological (5.2) | One self-supervised model that detects and localizes together. Localization uses an ancestor-conditioned score instead of deviation ranking: a task whose output follows from its inputs is a victim, and a task whose output is surprising given healthy inputs is the origin. |
| Explainability (5.3) | Attribution over eight named symptom families rather than raw features, and because the simulator knows which family each fault should move, attribution accuracy is scored as a number rather than displayed as a picture. |
| Application (5.4) | Three outputs per run: is it broken, which task, and which symptom. Benign seasonality, growth, drift and campaigns are generated and labelled normal on purpose, so false positives on ordinary business variation are visible in the results. |
| Evaluation (5.5) | AUPRC and F1 at a fixed alert budget as headline metrics, point-adjusted F1 reported only alongside and labelled as such, three percent contamination left in the training history, chronological splits, and normalisation statistics fitted on the training split alone. |

---

## 7. Positioning in one paragraph

The closest neighbours to this work are GDN, which brings graph structure to
anomaly detection but ranks by deviation magnitude, LogBERT, which brings masked
self-supervision to operational data but models log sequences rather than data
content, and TFDV, which has the right taxonomy of ETL failures but detects them
with hand-written rules. This project sits in the space those three leave open:
self-supervised learning over data-quality profiles, structured by the pipeline
DAG, producing a root-cause score that is designed to survive propagation, and
evaluated on a benchmark built specifically because none existed.
