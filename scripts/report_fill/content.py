"""The written content for the VIT project report, chapter by chapter.

Kept apart from the document plumbing so the prose can be read and edited
without wading through python-docx calls. Every heading level is marked, so the
builder can apply the template's formatting without guessing.

("h1", text)   chapter heading
("h2", text)   section heading
("h3", text)   subsection heading
("p",  text)   body paragraph
("b",  text)   bullet
("fig", path, caption)
("tbl", caption, [[row], [row]])
("code", text)
("pb",)        page break
"""

TITLE = "Self-Supervised AI for ETL Error Detection and Root-Cause Localization"
GUIDE = "Dr. Abdul Gaffar H"
GUIDE_DESIG = "Associate Professor Senior"
PROGRAMME = "Computer Science and Engineering with Specialization in Data Science"
# Shorter form, for the cover where the full name wraps badly.
PROGRAMME_SHORT = "Computer Science and Engineering (Data Science)"
COURSE = "Deep Learning Course Project"
STUDENTS = [("23BDS0149", "AKSHAT SPARSH"), ("23BDS0162", "ANUPRAKASH DUBEY")]
DATE = "01 November 2026"
# Scanned signatures for the declaration, keyed by registration number. These
# are personal, so they stay out of version control; prepare_signatures.py
# builds them from photos dropped in artifacts/signatures/raw.
SIGNATURES = {
    "23BDS0149": "artifacts/signatures/akshat.png",
    "23BDS0162": "artifacts/signatures/anuprakash.png",
}
MONTH = "November 2026"

ABSTRACT = """Extract, Transform and Load pipelines move data between the systems a
business runs on, and they fail in a way that is easy to miss. The job completes,
the scheduler reports success, and the data that lands is wrong. A currency
convention changes upstream and every amount becomes a hundred times too large.
Nothing raises an exception, so nothing raises an alarm, and the error is usually
found weeks later by a person looking at a report that does not make sense.

Detecting these failures automatically is difficult for two reasons. Labelled
examples barely exist, because incidents are rare and are recorded
inconsistently when they are recorded at all. And locating the cause is harder
still, because once a fault appears, every task downstream of it also looks
broken. The damage frequently grows as it travels, so the task with the largest
anomaly is often the furthest from the fault.

This project addresses both problems with a self-supervised model trained on
unlabelled pipeline telemetry. Because no public benchmark of ETL failures with
root-cause annotations exists, the work first constructs one: a nine-task
warehouse pipeline simulated at row level, with fifteen fault types injected
into the rows rather than into the summary statistics, so that propagation
through the task graph emerges from the simulation instead of being scripted.
The extent of each incident is measured rather than assumed, by re-running every
faulty batch with the fault removed under identical random seeds and comparing
the two. Measured reach is 2.48 of nine tasks per incident, against 3.94 under
the usual topological assumption.

The model is a factorised spatio-temporal Transformer of 1.1 million parameters,
trained from scratch by masked reconstruction over the data-quality profiles the
pipeline produces. On detection it reaches 0.953 AUPRC. On root-cause
localization it reaches 0.797 Hit@1 and 0.973 Hit@3, which is statistically tied
with the strongest baselines rather than ahead of them.

The finding that survives scrutiny concerns scoring rather than architecture.
Ranking tasks by raw anomaly magnitude asks which task has the largest error,
when the useful question is which task is furthest from its own normal. Two
independent corrections for this, conditioning each task on its parents and
calibrating each task against its own history, prove worth the same, 0.773 Hit@1
each against 0.588 uncalibrated. Combining them adds little. Explanations are
scored rather than displayed, and Integrated Gradients names the correct symptom
family for 93.5 per cent of incidents.

Keywords: self-supervised learning, anomaly detection, root-cause analysis, data
quality, ETL pipelines, Transformers, explainable AI."""

CHAPTERS = [

# ===================== CHAPTER 1 =====================
("h1", "CHAPTER 1\nINTRODUCTION"),

("h2", "1.1 Background"),
("p", "An Extract, Transform and Load pipeline is the plumbing of a modern data "
      "platform. It pulls records out of the systems that create them, reshapes "
      "and cleans those records, joins them together, and loads the result into "
      "a warehouse that reporting and analytics sit on top of. A pipeline of "
      "this kind is a sequence of dependent tasks, usually drawn as a directed "
      "acyclic graph, and it runs on a schedule without anyone watching it."),
("p", "Because these pipelines run unattended, the organisation's trust in its "
      "own numbers rests on them behaving correctly. When a scheduled job "
      "crashes, existing monitoring handles it well: the scheduler notices, an "
      "alert fires, and somebody fixes it the same day. That failure mode is "
      "already solved."),
("p", "The failure mode that is not solved is the silent one. The task finishes, "
      "reports success, and writes data that is wrong. An upstream team changes "
      "a price field from rupees to paise. A source system begins emitting empty "
      "values for a field that was always populated. A timestamp column switches "
      "from UTC to local time. In each case the pipeline runs to completion, the "
      "scheduler sees nothing wrong, and incorrect data accumulates in the "
      "warehouse for as long as it takes a human being to notice that a report "
      "looks implausible."),
("p", "Industry has responded with rule-based data validation. Systems such as "
      "Amazon's Deequ and Google's TensorFlow Data Validation let engineers "
      "declare constraints on their data and check them as the pipeline runs. "
      "These systems work, and the taxonomy of failures they encode is better "
      "grounded than anything in the research literature. Their limitation is "
      "that a person has to write every constraint, for every column, in every "
      "table, and keep them current as the schema evolves. Rules only catch what "
      "somebody already thought to check for, and the unit change catches you "
      "precisely because nobody thought of it."),

("h2", "1.2 Motivations"),
("p", "Three observations motivated this work."),
("p", "The first is that the cost of a silent data failure is carried by every "
      "decision made on the affected data, for the entire period before anyone "
      "notices. A crash costs an hour of engineering time. A silent corruption "
      "costs weeks of wrong reporting, and the cost is invisible until it is "
      "discovered."),
("p", "The second is that labelled failures are not available and are unlikely to "
      "become available. Incidents are rare relative to the number of successful "
      "runs. They are documented inconsistently, if at all, and the incidents "
      "nobody ever noticed are still sitting unlabelled in the run history. Any "
      "method that requires a corpus of labelled failures is a method that cannot "
      "be deployed in practice."),
("p", "The third, and the one that shaped the whole project, is that detection on "
      "its own is not useful. When a fault appears at one task, every task "
      "downstream of it consumes the corrupted output and looks faulty too. The "
      "disturbance frequently grows as it propagates, so a five per cent null "
      "rate at extraction can present as a thirty per cent revenue shortfall "
      "after aggregation. A detector that reports nine unhealthy tasks out of "
      "nine has given the engineer on call no information they did not already "
      "have. The useful output is not a list of symptoms but the identity of the "
      "task where the trouble began."),

("h2", "1.3 Scope of the Project"),
("p", "The project covers the design, implementation and evaluation of a "
      "self-supervised deep learning system that, for each scheduled run of a "
      "data pipeline, answers three questions: whether the run is broken, which "
      "task originated the problem, and what kind of problem it is."),
("p", "Because no public dataset supports the second of those questions, the scope "
      "also includes the construction of a benchmark. A nine-task warehouse "
      "pipeline is simulated at row level over eight thousand hourly runs, with "
      "fifteen types of realistic fault injected into the data itself."),
("p", "The following are deliberately outside the scope of this work. The system "
      "does not repair faults, it reports them. It does not integrate with a "
      "production scheduler or alerting system. It is evaluated on simulated "
      "rather than real telemetry, for the reason given in Chapter 2. And it "
      "assumes the structure of the pipeline is known in advance, which is "
      "reasonable, since the organisation running a pipeline knows its shape."),
("pb",),

# ===================== CHAPTER 2 =====================
("h1", "CHAPTER 2\nPROJECT DESCRIPTION AND GOALS"),

("h2", "2.1 Literature Review"),
("p", "The problem addressed here sits at the meeting point of three research "
      "communities that largely do not read one another: multivariate time-series "
      "anomaly detection, data quality validation for pipelines, and root-cause "
      "analysis for distributed systems. Twenty-seven papers were surveyed across "
      "the major venues of all three, and they group into six themes."),

("h3", "2.1.1 Reconstruction and prediction methods"),
("p", "The dominant approach in anomaly detection learns to reproduce normal "
      "behaviour and treats whatever the model cannot reproduce as anomalous. "
      "Malhotra et al. established that the prediction error of a stacked LSTM "
      "carries a usable anomaly signal, then found that reconstruction holds up "
      "better than prediction when a series is genuinely unpredictable, which is "
      "the more common case. OmniAnomaly extended this by making normality "
      "probabilistic rather than a point estimate, at a considerable cost in "
      "speed, which USAD then addressed using two adversarially trained "
      "autoencoders to reach comparable accuracy far faster."),
("p", "What none of these question is the training data. All of them assume a "
      "clean set of normal examples. In a pipeline setting that assumption fails "
      "quietly, because nobody knows which historical runs were healthy."),

("h3", "2.1.2 Structure-aware and graph-based methods"),
("p", "The methods above treat variables as an unordered collection. MTAD-GAT and "
      "GDN both push back on that by introducing a graph over the variables, and "
      "both improve on the flat baselines, which is reasonable evidence that "
      "inter-variable structure carries signal. They differ in where the graph "
      "comes from: MTAD-GAT learns attention over features, while GDN learns an "
      "explicit sensor graph and scores each node by how far it deviates from "
      "what its neighbours predict."),
("p", "GDN comes closest to what this project needs, and the gap between them is "
      "instructive. GDN ranks sensors by deviation magnitude and describes this "
      "as localization. In a sensor network that is defensible. In a pipeline it "
      "is not, because magnitude does not survive propagation, so a ranking by "
      "deviation systematically favours the end of the graph furthest from the "
      "fault."),

("h3", "2.1.3 Transformer and attention-based methods"),
("p", "TranAD brought transformers to this problem with a two-phase adversarial "
      "training scheme, and Anomaly Transformer proposed a detection criterion "
      "based on association discrepancy rather than reconstruction error. Both "
      "report strong numbers, and both report them under a metric this project "
      "declines to headline, for reasons given in Section 2.1.6."),

("h3", "2.1.4 Self-supervised pretraining"),
("p", "This is the newest thread and the most promising. Zerveas et al. showed "
      "masked-value modelling transfers to multivariate time series, TS2Vec "
      "showed hierarchical contrastive learning does the same, and LogBERT "
      "applied masked modelling to operational data specifically. Masked "
      "autoencoding supplied the design lesson that matters most here, which is "
      "that the masking has to be aggressive enough that the task cannot be "
      "solved by local interpolation."),
("p", "None of this has reached ETL data quality. LogBERT is the nearest "
      "neighbour, and it models sequences of log keys, so it observes that a job "
      "ran and in what order but never what the data looked like. A pipeline can "
      "execute flawlessly, emit a perfect log, and load a table in which every "
      "amount is wrong by a factor of a hundred."),

("h3", "2.1.5 Data quality and pipeline validation systems"),
("p", "This is where the problem actually lives. Deequ made declarative data "
      "quality checks practical at production scale, and TensorFlow Data "
      "Validation built the schema and skew machinery much of the industry now "
      "copies. Both encode a taxonomy of ETL failures more grounded than anything "
      "in the machine learning papers, and the fifteen fault types used in this "
      "project are drawn from it. Their shared limitation is that the constraints "
      "are written by hand. Auto-Validate is the most serious attempt to remove "
      "that burden and works well, but operates within a single column, so it "
      "cannot see a fault that only appears as a relationship between two tables. "
      "Redyuk et al. learn thresholds rather than fixing them, which handles "
      "drift, but the output remains a flag on a metric rather than a diagnosis."),

("h3", "2.1.6 Evaluation practice"),
("p", "Two papers deserve more attention than their citation counts suggest. Kim "
      "et al. re-evaluated the standard benchmarks under point-adjusted F1, the "
      "metric most of this literature reports, and found that a randomly "
      "generated anomaly score can beat published models. The mechanism is "
      "simple: point adjustment credits a method with an entire anomalous segment "
      "for detecting any single timestep inside it, so on datasets with long "
      "anomalous stretches, alerting at random performs well. Wu and Keogh "
      "reached a related conclusion from the data side, documenting triviality, "
      "mislabelling and run-to-failure bias in the benchmark datasets themselves."),
("p", "Taken together, a meaningful fraction of the reported progress in this area "
      "may be an artefact of measurement. That is not grounds for discarding the "
      "methods, but it is strong grounds for not reporting point-adjusted F1 as a "
      "headline figure, which this project does not."),

("h2", "2.2 Research Gap"),
("p", "The survey exposes five gaps, which the objectives in Section 2.3 are "
      "written against."),
("h3", "2.2.1 Dataset gap"),
("p", "There is no public benchmark of ETL pipeline failures with root-cause "
      "annotations. The detection literature evaluates almost entirely on SMD, "
      "SWaT, SMAP and MSL, which are server and sensor telemetry. None of them "
      "contains a task graph, none contains data-content quality metrics, and "
      "none labels which component originated a failure. The systems research "
      "that does work on real pipeline data uses proprietary data that is never "
      "released. The practical consequence is that root-cause localization in a "
      "data pipeline cannot currently be posed as a benchmark task, let alone "
      "measured."),
("h3", "2.2.2 Methodological gap"),
("p", "Detection and localization are treated separately. The detection community "
      "detects, the root-cause community localizes, and the localization work "
      "depends on service call traces that a data pipeline does not produce. "
      "Where localization does appear inside a detector, it is deviation ranking, "
      "which propagation defeats. And self-supervised learning has not been "
      "applied to data-quality profiles at all: one community has the machinery, "
      "another has the problem, and nobody has connected them."),
("h3", "2.2.3 Explainability gap"),
("p", "Explanation in the detection literature means a bar chart of reconstruction "
      "error per variable, which says which number moved rather than what kind of "
      "failure occurred. Integrated Gradients and SHAP are mature tools that are "
      "rarely applied here. More significantly, when attribution is shown it is "
      "almost never scored, because there is no ground truth against which to "
      "check it."),
("h3", "2.2.4 Application gap"),
("p", "Nearly all of this work outputs a binary flag. An engineer responding to an "
      "incident needs three things: whether the run is broken, which task broke "
      "it, and what kind of break it is. No single system in the survey produces "
      "all three. There is also a blind spot around benign change. Real pipelines "
      "exhibit seasonality, growth and one-off events such as marketing "
      "campaigns, all of which move volume sharply and none of which is a fault. "
      "The public benchmarks contain little of this, so published false-positive "
      "rates are optimistic."),
("h3", "2.2.5 Evaluation gap"),
("p", "Following the critiques above, three aspects of the standard protocol are "
      "worth fixing rather than inheriting: point-adjusted F1 inflates results to "
      "the point of meaninglessness, training sets are assumed clean when real "
      "history is not, and random rather than chronological splits allow "
      "information from the future to reach the past."),

("h2", "2.3 Objectives"),
("p", "Four objectives were set, each with the criterion used to judge whether it "
      "had been met. The criteria were fixed before the experiments were run, so "
      "that success could not be redefined after seeing the results."),
("tbl", "Table 2.1  Objectives and their success criteria",
  [["No.", "Objective", "Criterion fixed in advance", "Outcome"],
   ["1", "Detect broken pipeline runs using no labels during training",
    "AUPRC above 0.90 on held-out runs, with checkpoints selected on a "
    "label-free criterion", "Met, 0.953"],
   ["2", "Identify the task that originated an incident rather than the tasks "
    "merely affected by it",
    "Originating task ranked first in more than half of incidents and within "
    "the top three in more than 90 per cent", "Met, 0.797 and 0.973"],
   ["3", "Explain each finding as a named symptom rather than a feature index",
    "Named symptom family matches the fault's expected family in more than 80 "
    "per cent of incidents", "Met, 0.935"],
   ["4", "Evaluate honestly against baselines on a benchmark that makes the "
    "questions answerable",
    "Chronological splits, contaminated training history, confidence intervals, "
    "no point-adjusted F1 as a headline, at least four comparison methods",
    "Met, six baselines"]]),
("p", "Objective 2 is the one that defines the project. Objectives 1 and 3 are "
      "achievable by existing methods; locating the origin rather than the "
      "symptoms is not."),

("h2", "2.4 Problem Statement"),
("p", "An ETL pipeline is a directed acyclic graph of N tasks. Each scheduled run "
      "t produces, for every task, a data-quality profile of F numbers, so a run "
      "is a matrix of shape N by F and the observable history is a tensor of "
      "shape T by N by F. In this work N is nine, F is thirty-eight and T is "
      "eight thousand."),
("p", "Given a window of W consecutive runs ending at time t, the system must "
      "produce a run-level anomaly score, a per-task root-cause score, and a "
      "per-feature attribution identifying the symptom. The constraint that "
      "defines the problem is that none of the training signal may come from "
      "labels; all three outputs must fall out of a self-supervised objective."),
("p", "Formally, during an incident originating at task v, the set of anomalous "
      "tasks is v together with those downstream of it that the fault actually "
      "reaches. Detection asks whether any task is anomalous. Localization asks "
      "which one is v. Because downstream symptoms are frequently larger than the "
      "originating one, ranking tasks by anomaly magnitude is a poor localizer, "
      "and the model must therefore reason over the graph rather than over each "
      "task independently."),

("h2", "2.5 Project Plan"),
("tbl", "Table 2.2  Project plan and phases",
  [["Phase", "Work carried out", "Deliverable"],
   ["1", "Literature survey, problem formulation, simulator design",
    "Review matrix, gap analysis, task graph specification"],
   ["2", "Simulator implementation, fault taxonomy, profiler, counterfactual "
    "labelling", "Benchmark of 8000 runs with measured labels"],
   ["3", "Model architecture, masking strategies, training loop",
    "Trained model and training logs"],
   ["4", "Baselines, scoring rules, evaluation harness, metrics",
    "Comparative results with confidence intervals"],
   ["5", "Ablations, seed repeats, error analysis, explainability study",
    "Six ablation runs and an attribution score"],
   ["6", "Demonstration, documentation, report",
    "Terminal and web demonstrations, final report"]]),
("pb",),
]

CHAPTERS += [

# ===================== CHAPTER 3 =====================
("h1", "CHAPTER 3\nTECHNICAL SPECIFICATION"),

("h2", "3.1 Requirements"),
("h3", "3.1.1 Functional Requirements"),
("tbl", "Table 3.1  Functional requirements",
  [["ID", "Requirement"],
   ["FR1", "Generate a reproducible benchmark of pipeline runs with injected "
    "faults and measured labels"],
   ["FR2", "Compute a fixed-width data-quality profile for every task in every run"],
   ["FR3", "Train a model on unlabelled run history, using no labels for "
    "training or checkpoint selection"],
   ["FR4", "Produce a run-level score indicating whether a run is broken"],
   ["FR5", "Rank the nine tasks by the likelihood that each originated the incident"],
   ["FR6", "Name the symptom family and the single measurement most responsible"],
   ["FR7", "Compare against classical and deep baselines on identical data and splits"],
   ["FR8", "Report detection, localization and attribution metrics with "
    "confidence intervals"],
   ["FR9", "Demonstrate a diagnosis interactively without requiring a server"]]),

("h3", "3.1.2 Non-Functional Requirements"),
("tbl", "Table 3.2  Non-functional requirements",
  [["ID", "Requirement"],
   ["NFR1", "Reproducibility. Every reported number must be regenerable from a "
    "script with a fixed seed"],
   ["NFR2", "Training must complete in minutes on consumer hardware, so that "
    "ablations remain affordable"],
   ["NFR3", "No training signal may derive from labels, including the choice of "
    "saved checkpoint"],
   ["NFR4", "Splits must be chronological, so no information from the future "
    "reaches the past"],
   ["NFR5", "The demonstration must run with no network access and no running "
    "server process"],
   ["NFR6", "The codebase must be covered by automated tests that verify its "
    "own invariants"],
   ["NFR7", "Interpretability. Each diagnosis must be expressible in terms a "
    "data engineer can act on"]]),

("h2", "3.2 Feasibility Study"),
("h3", "3.2.1 Technical Feasibility"),
("p", "The project is technically feasible on ordinary hardware, and this was a "
      "design constraint rather than an accident. The model holds 1.1 million "
      "parameters and the dataset is eight thousand runs of nine tasks by "
      "thirty-eight features, which is small by contemporary standards. A single "
      "training run completes in roughly eighteen minutes on the integrated GPU "
      "of an Apple M4 laptop using PyTorch's Metal backend."),
("p", "That budget was chosen deliberately. The evaluation requires nine separate "
      "training runs for the seed repeats and ablations, and a project needing "
      "many cheap experiments is better served by a small model trained locally "
      "than by one large model trained on rented hardware. No cloud compute, no "
      "pre-trained weights and no external services are required at any stage."),
("h3", "3.2.2 Economic Feasibility"),
("p", "The direct cost of the project is zero. Every tool in the stack is open "
      "source: Python, PyTorch, pandas, scikit-learn and matplotlib. No dataset "
      "was purchased, no compute was rented, and no commercial licence is needed "
      "to reproduce any result in this report. The indirect cost is the "
      "approximately three hours of laptop GPU time consumed by the nine "
      "training runs."),
("p", "Considered as a prospective deployment rather than a study, the economics "
      "remain favourable. Inference over a single pipeline run takes a fraction "
      "of a second, so a production system would consume negligible compute "
      "relative to the pipeline it monitors, and the cost it offsets is the "
      "engineering time currently spent writing and maintaining validation rules "
      "by hand."),
("h3", "3.2.3 Social Feasibility"),
("p", "The system is intended to assist a data engineer rather than replace one. "
      "Its output is a ranked shortlist of candidate tasks and a named symptom, "
      "which an engineer then investigates. It takes no automated remedial "
      "action, so a wrong answer costs a few minutes of misdirected attention "
      "rather than damaged data."),
("p", "Two considerations shaped the design in this respect. The first is "
      "interpretability: a diagnosis expressed as a feature index would be "
      "useless to the person receiving it, so attribution is reported in terms of "
      "eight named symptom families instead. The second is alert fatigue, which "
      "is the usual reason monitoring tools get ignored. Evaluation therefore "
      "includes performance at a fixed alert budget, and the benchmark "
      "deliberately includes benign variation such as seasonality and marketing "
      "campaigns, so that false alarms on ordinary business change are visible in "
      "the results rather than hidden."),
("p", "The data used is entirely synthetic, so no personal or commercial "
      "information is involved and no privacy or consent issues arise."),

("h2", "3.3 System Specification"),
("h3", "3.3.1 Hardware Specification"),
("tbl", "Table 3.3  Hardware used",
  [["Component", "Specification"],
   ["Machine", "Apple MacBook, M4 system on a chip"],
   ["Memory", "16 GB unified memory"],
   ["GPU", "Integrated Apple GPU, accessed through the Metal Performance "
    "Shaders backend"],
   ["Storage required", "Approximately 100 MB for the dataset, trained models "
    "and all results"],
   ["Minimum to reproduce", "Any machine with 8 GB of memory; a GPU shortens "
    "training but is not required"]]),
("h3", "3.3.2 Software Specification"),
("tbl", "Table 3.4  Software stack",
  [["Component", "Version", "Purpose"],
   ["Python", "3.12", "Implementation language"],
   ["PyTorch", "2.15", "Model definition, training, deep baselines"],
   ["NumPy", "2.5", "Numerical computation and dataset storage"],
   ["pandas", "3.0", "Row-level pipeline simulation and transformations"],
   ["scikit-learn", "1.9", "Classical baselines and evaluation metrics"],
   ["SciPy", "1.18", "Statistical functions"],
   ["matplotlib", "3.11", "All figures"],
   ["pytest", "9.1", "Automated test suite"],
   ["HTML, CSS, JavaScript", "No framework", "Self-contained web demonstration"],
   ["Git and GitHub", "Latest", "Version control"]]),
("pb",),

# ===================== CHAPTER 4 =====================
("h1", "CHAPTER 4\nDESIGN APPROACH AND DETAILS"),

("h2", "4.1 System Architecture"),
("p", "The system is organised into five layers. Data flows in one direction, and "
      "every layer writes its output to disk before the next reads it. This "
      "matters more than it may appear: the simulator does not know the model "
      "exists, the model does not know the demonstration exists, and any layer "
      "can therefore be re-run on its own. It is also the reason every figure and "
      "table in this report is regenerable from a script rather than assembled by "
      "hand."),
("fig", "artifacts/report_diagrams/fig_4_1_architecture.png",
        "Figure 4.1  System architecture, five layers with one-directional flow"),
("p", "The simulation layer defines the pipeline and produces the benchmark. The "
      "data layer slices the resulting history into overlapping windows, "
      "normalises it and splits it chronologically. The model layer holds the "
      "Transformer and the masking logic. The scoring layer converts a trained "
      "model into the three outputs. The presentation layer produces metrics, "
      "figures and the two demonstrations."),

("h2", "4.2 Design"),
("h3", "4.2.1 Data Flow Diagram"),
("p", "Figure 4.2 shows the level-1 data flow. Four processes transform data "
      "between four stores. The raw rows produced by the simulation are never "
      "seen by the model; only the profiles derived from them are."),
("fig", "artifacts/report_diagrams/fig_4_2_dfd.png",
        "Figure 4.2  Level-1 data flow diagram"),
("h3", "4.2.2 Use Case Diagram"),
("p", "Two actors interact with the system. A researcher generates the benchmark "
      "and trains models. A data engineer, who is the intended end user, submits "
      "runs and reads diagnoses."),
("fig", "artifacts/report_diagrams/fig_4_3_usecase.png",
        "Figure 4.3  Use case diagram"),
("h3", "4.2.3 Class Diagram"),
("p", "Figure 4.4 shows the principal classes. The Pipeline holds the task graph "
      "and answers questions about ancestry, which the scoring layer depends on. "
      "FaultEvent describes one injected incident. Profiler converts a table of "
      "rows into the thirty-eight-number profile. ETLData prepares windows and "
      "normalisation. STTransformer is the model, and ScoreBundle carries the "
      "scores it produces."),
("fig", "artifacts/report_diagrams/fig_4_4_class.png",
        "Figure 4.4  Class diagram of the principal components"),
("h3", "4.2.4 Sequence Diagram"),
("p", "Figure 4.5 traces a single diagnosis. The important detail is that the "
      "model is queried more than once per run. It is asked to reconstruct the "
      "final run with every task hidden, which yields the detection signal, and "
      "then once per task with that task and all of its descendants hidden, which "
      "yields the localization signal. Section 5.1.4 explains why the descendants "
      "must also be hidden."),
("fig", "artifacts/report_diagrams/fig_4_5_sequence.png",
        "Figure 4.5  Sequence diagram for one diagnosis"),
("pb",),
]

CHAPTERS += [

# ===================== CHAPTER 5 =====================
("h1", "CHAPTER 5\nMETHODOLOGY AND TESTING"),

("h2", "5.1 Module Description"),

("h3", "5.1.1 The simulation module"),
("p", "The simulator implements a nine-task e-commerce warehouse pipeline. Three "
      "tasks extract customers, orders and payments. Two clean them. Two join "
      "them together. One aggregates to a daily grain, and one publishes to the "
      "warehouse. Every task performs a genuine transformation over real pandas "
      "rows: type casting, deduplication, currency normalisation, left joins and "
      "group-by aggregation. Transforms are written to degrade rather than crash, "
      "as production jobs do, so a task whose join key was removed upstream "
      "produces empty joins instead of an exception."),
("p", "Source data is drawn from a persistent pool of twenty thousand customers, "
      "so foreign keys point at records that exist and referential integrity is a "
      "meaningful signal. Volume follows a daily cycle, a weekly cycle, a growth "
      "trend, slow drift in the customer country mix, and occasional marketing "
      "campaigns that raise volume by around seventy per cent for a day or two. "
      "All of this variation is labelled normal on purpose. Without it, a simple "
      "threshold on row count would score well and no deep model could be "
      "justified."),
("p", "Fifteen fault types are implemented across six families, listed in Table "
      "5.1. The single most important design decision in the project is that "
      "faults corrupt the rows, never the summary statistics. Had the statistics "
      "been corrupted directly, somebody would have had to decide by hand how the "
      "damage spreads to the next task, and the model would then have been "
      "learning those hand-written rules rather than anything about data "
      "pipelines. By breaking the actual data, the next task genuinely reads "
      "corrupted rows and the damage spreads as a consequence of the real "
      "transformations."),
("tbl", "Table 5.1  Fault taxonomy, fifteen types across six families",
  [["Family", "Fault", "What happens"],
   ["Schema", "schema_column_drop", "A column silently disappears"],
   ["", "schema_column_add", "An unannounced column appears"],
   ["", "type_mismatch", "Numbers arrive as text and casts fail"],
   ["Completeness", "null_spike", "A source begins emitting empty fields"],
   ["", "pk_null", "Record identifiers go missing, breaking deduplication"],
   ["", "duplicate_records", "A redelivered batch duplicates rows"],
   ["", "row_loss_truncation", "A partial load cuts the batch short"],
   ["Value", "unit_change", "A currency or unit convention changes upstream"],
   ["", "value_corruption", "Extreme or sentinel values appear"],
   ["", "category_corruption", "Encoding drift mangles category labels"],
   ["Temporal", "timezone_shift", "A producer switches from UTC to local time"],
   ["", "late_arrival", "The source lags and carries stale events"],
   ["Integrity", "fk_break", "Key formatting drifts so joins orphan rows"],
   ["", "join_fanout", "Duplicate keys turn a one-to-one join into one-to-many"],
   ["Operational", "resource_degradation", "The task misses its deadline while "
    "its output stays valid"]]),
("p", "The last of these is deliberately awkward. The task blows its service "
      "level objective while the data it produces is perfectly correct, so a "
      "detector that looks only at data quality must not confuse it with "
      "corruption, and it must not propagate downstream."),

("h3", "5.1.2 Counterfactual labelling"),
("p", "The obvious way to label which tasks an incident affected is to take the "
      "origin plus every descendant. That turns out to be wrong, and provably so. "
      "An unexpected extra column is ignored by every downstream select. A task "
      "that runs slowly still emits correct rows. Labelling those descendants "
      "anomalous would penalise a detector for being right about them."),
("p", "The simulator therefore re-runs every faulty batch a second time with the "
      "fault removed, holding the source rows, the telemetry noise and the "
      "scheduling jitter identical. This works because each task draws its "
      "randomness from its own generator seeded on the triple of seed, batch and "
      "task, which guarantees the two runs are bit-identical wherever the fault "
      "does not reach. A test asserts exactly that. A task counts as affected "
      "only when its profile actually moved."),
("p", "Measured reach is 2.48 of nine tasks per incident, against 3.94 under the "
      "topological assumption. The fault schema_column_add correctly reaches "
      "exactly one task where the assumption claims four and a half."),

("h3", "5.1.3 Feature extraction and preparation"),
("p", "Each task run is summarised into thirty-eight numbers, aggregated by the "
      "semantic role of each column rather than by column name. That is the "
      "mechanism that keeps the profile a fixed width across tasks whose tables "
      "share no columns at all. The thirty-eight group into eight families, "
      "listed in Table 5.2, and those families are the vocabulary the "
      "explanation layer speaks in."),
("tbl", "Table 5.2  The eight symptom families",
  [["Family", "Measurements", "What it captures"],
   ["Operational", "8", "Runtime, memory, retries, queue delay, rows in and out"],
   ["Metric distribution", "8", "Mean, spread, extremes of the primary measure"],
   ["Completeness", "5", "Null rates, duplicate keys, missing identifiers"],
   ["Temporal", "5", "Freshness, staleness, timestamps outside the window"],
   ["Schema", "4", "Column count, schema changes, cast failures"],
   ["Categorical distribution", "4", "Cardinality, unseen values, entropy"],
   ["Numeric distribution", "3", "Nulls, outliers and negatives in secondary columns"],
   ["Integrity", "1", "Orphaned foreign keys"]]),
("p", "Preparation reflects what the data actually looks like. Twenty-two of the "
      "thirty-eight features have an absolute skew above two and the worst "
      "reaches one hundred and forty-five, because row counts and runtimes are "
      "long tailed by nature, so heavy-tailed quantities are stored as log values "
      "and normalisation uses median and interquartile range rather than mean and "
      "standard deviation. Seventeen of the thirty-eight never vary at all while "
      "the pipeline is healthy, so their interquartile range is zero; these fall "
      "back to standard deviation, because treating them as constants would make "
      "a schema flag flipping register as a one-sigma event, weaker than ordinary "
      "volume noise."),

("h3", "5.1.4 The model and its training"),
("p", "The model is a factorised spatio-temporal Transformer of 1.1 million "
      "parameters. Its input is a window of thirty-two consecutive runs, so a "
      "tensor of shape thirty-two by nine by thirty-eight. Each encoder layer "
      "attends twice: once along time, so every task sees its own history, and "
      "once across the task graph, so every timestep sees the other tasks. "
      "Attending over all two hundred and eighty-eight positions jointly would "
      "cost an order of magnitude more and would mix two kinds of structure that "
      "behave differently, since the time axis is long and the task axis is "
      "short. Four such layers are stacked."),
("p", "Structural attention carries a learned bias per graph relationship. Every "
      "ordered pair of tasks is classified as self, direct parent, direct child, "
      "distant ancestor, distant descendant or unrelated, and the model learns "
      "one bias per relationship per attention head, so it decides for itself how "
      "much a task should attend to its immediate upstream rather than having a "
      "hard mask imposed."),
("p", "Training is masked reconstruction. A subset of entries is hidden and the "
      "model predicts them from context, with the loss measured only on the "
      "hidden entries. The mask is supplied to the model as an input rather than "
      "used only as a loss weight, so filling the gap is a well-posed question "
      "instead of a guess about whether a zero means healthy or missing. Two "
      "masking strategies are used. Feature-group masking hides one whole symptom "
      "family at a task and timestep, which matters because null_rate_mean sits "
      "beside null_rate_max in the same vector and is trivially recoverable from "
      "it; hiding a whole family closes that shortcut. Task masking hides a task's "
      "entire profile, which is the objective the root-cause scorer rests on."),
("p", "Checkpoints are selected on validation reconstruction loss, which requires "
      "no labels. Selecting on detection accuracy would quietly make the method "
      "weakly supervised while still calling it self-supervised."),

("h3", "5.1.5 The scoring module"),
("p", "Scoring mirrors training. The model was trained to fill in hidden entries, "
      "so at inference it is asked the same kind of question. Running a masked "
      "autoencoder unmasked allows it to copy its input to its output, and the "
      "reconstruction error then measures nothing at all."),
("p", "Two masking configurations are applied at the final timestep. In the first, "
      "every task is hidden, so each prediction rests on history alone; this is "
      "large whenever a run departs from its own past and serves as the detection "
      "signal. In the second, task i and all of its descendants are hidden, so the "
      "prediction rests only on that task's ancestors and its own history. This is "
      "surprise given inputs. A task downstream of a failure emits anomalous "
      "output that follows perfectly from its own anomalous inputs, so its score "
      "stays low and it reads correctly as a victim. The task where the incident "
      "began has healthy inputs and anomalous output, so its score stays high."),
("p", "Hiding the descendants is essential and was absent from the first "
      "implementation. Structural attention is bidirectional, so a hidden task "
      "could be reconstructed from its own descendants, which during an incident "
      "are corrupted too and leak the value straight back. Correcting this raised "
      "localization accuracy from 0.632 to 0.773, and Section 7.4 gives the "
      "evidence that the gain comes from the stated mechanism."),
("p", "Finally, scores are calibrated per task. Tasks are not equally predictable, "
      "so their raw errors are not comparable and whichever task is intrinsically "
      "hardest to predict wins any ranking by default. Each task's score is "
      "therefore converted to a percentile against its own distribution over "
      "unlabelled training runs. Percentile rank is used rather than a z-score "
      "because the error distributions are heavy tailed; z-scoring was tried and "
      "measurably performed worse."),

("h2", "5.2 Testing"),
("p", "The codebase carries fifty-nine automated tests. They are not incidental: "
      "two of them exist because they caught real defects that would otherwise "
      "have invalidated results, and several assert invariants that nothing else "
      "in the system would notice breaking."),
("tbl", "Table 5.3  Test coverage by category",
  [["Test file", "Tests", "What it verifies"],
   ["test_simulator.py", "24", "Table contracts, healthy baselines, per-fault "
    "response, propagation, counterfactual isolation"],
   ["test_models.py", "11", "Graph relation matrix, masking behaviour, encoder "
    "shapes, mask leakage"],
   ["test_eval.py", "17", "Scoring indexing, descendant masking, detection and "
    "localization metrics"],
   ["test_dataset.py", "7", "Robust scaling, contamination resistance, "
    "constant features"]]),
("p", "Four tests are worth describing individually."),
("p", "The healthy-baseline test asserts that with no faults injected, integrity "
      "and completeness features sit at zero. It failed during development and "
      "exposed a genuine simulator defect: payments could settle after their "
      "batch window closed, so healthy runs reported around ten per cent of "
      "timestamps as being in the future, which made the freshness features "
      "meaningless."),
("p", "The counterfactual isolation test asserts that tasks outside the blast "
      "radius are bit-identical between the factual and counterfactual runs. The "
      "entire measured-propagation labelling rests on that property, and nothing "
      "else would reveal its loss."),
("p", "The mask leakage test rewrites a masked value to an absurd number and "
      "asserts the model's output does not change. If it did, the model could "
      "read the answer it is being asked to predict, and every reconstruction "
      "error afterwards would be meaningless."),
("p", "The scoring index test compares the batched per-task scoring path against "
      "a plain per-task loop. The batched implementation replicates each window "
      "nine times and gathers one row back out, and an off-by-one there would "
      "silently read another task's error, making every localization number "
      "wrong without any visible failure."),
("pb",),

# ===================== CHAPTER 6 =====================
("h1", "CHAPTER 6\nPROJECT DEMONSTRATION"),
("p", "Two demonstrations were built, for different purposes."),
("h2", "6.1 Terminal demonstration"),
("p", "The first loads the trained model and runs inference live. It selects real "
      "incidents from the held-out test split, diagnoses each one, and prints the "
      "ground truth beneath each answer so the result can be checked rather than "
      "taken on trust. It completes in about two seconds and exists to prove that "
      "the model genuinely runs."),
("h2", "6.2 Web demonstration"),
("p", "The second presents the same diagnoses visually. It is a single "
      "self-contained HTML file of forty-three kilobytes with the results already "
      "inside it, so it opens by double-clicking with no server, no Python "
      "runtime and no network access. The predictions in it are the model's own, "
      "computed on test data it had never seen, but they are precomputed rather "
      "than generated in the browser. The reasoning is simple: a demonstration "
      "that depends on a running process is a demonstration that can fail at the "
      "moment it is needed."),
("fig", "artifacts/slides/shots/demo_screenshot.png",
        "Figure 6.1  The web demonstration diagnosing a single pipeline run"),
("p", "The interface answers the three questions in order. A bar at the top gives "
      "the anomaly score against the alert threshold. The task graph below it "
      "highlights the tasks the damage reached and marks the one the model "
      "identifies as the origin. Two panels underneath rank the candidate tasks "
      "and name the symptom family, with the single most responsible measurement "
      "quoted in plain text. The ground truth is concealed behind a control, so "
      "an observer can form a judgement before the answer is revealed."),
("p", "The list on the left holds nineteen runs: one example of each of the "
      "fifteen fault types, chosen as the most severe instance of each so the "
      "effect is legible, together with four healthy runs. One of the nineteen is "
      "a case the model gets wrong, and it is retained deliberately, since a "
      "failure that was predicted and explained is more informative than a "
      "sequence of successes."),
("pb",),
]

CHAPTERS += [

# ===================== CHAPTER 7 =====================
("h1", "CHAPTER 7\nRESULT AND DISCUSSION"),
("p", "All results are on the test split: the final twenty-five per cent of the "
      "timeline, two thousand runs, of which two hundred and ninety-one carry an "
      "incident. Nothing was tuned on the test split."),

("h2", "7.1 Main comparison"),
("p", "Every method's per-task scores are percentile-calibrated against its own "
      "unlabelled training runs, so the comparison is like for like."),
("tbl", "Table 7.1  Comparison against six baselines, test split",
  [["Method", "AUPRC [95% CI]", "Best F1", "Hit@1 [95% CI]", "Hit@3", "MRR"],
   ["LSTM autoencoder", "0.951 [0.932, 0.968]", "0.928", "0.818 [0.773, 0.859]",
    "0.986", "0.900"],
   ["PCA residual", "0.942 [0.920, 0.963]", "0.928", "0.801 [0.756, 0.849]",
    "0.955", "0.880"],
   ["Proposed model", "0.953 [0.934, 0.971]", "0.935", "0.797 [0.749, 0.842]",
    "0.973", "0.883"],
   ["Local Outlier Factor", "0.973 [0.958, 0.987]", "0.963",
    "0.773 [0.725, 0.821]", "0.976", "0.872"],
   ["Dense autoencoder", "0.956 [0.937, 0.973]", "0.910", "0.749 [0.701, 0.797]",
    "0.924", "0.843"],
   ["Rolling z-score", "0.502 [0.455, 0.554]", "0.519", "0.570 [0.515, 0.629]",
    "0.942", "0.748"],
   ["Isolation Forest", "0.466 [0.413, 0.527]", "0.469", "0.340 [0.285, 0.395]",
    "0.701", "0.550"]]),
("fig", "artifacts/slides/bw/bw_methods.png",
        "Figure 7.1  Detection and localization across all seven methods"),
("p", "The proposed model does not win on either task, and this report states "
      "that rather than working around it. Local Outlier Factor detects better "
      "and the LSTM autoencoder localizes better. With two hundred and "
      "ninety-one incidents the standard error on a Hit@1 near 0.8 is 0.023, and "
      "the seed-to-seed standard deviation of the proposed model is 0.011, so "
      "the top four localizers should be read as tied rather than ranked; their "
      "confidence intervals overlap throughout."),
("p", "The result deserves an explanation rather than an excuse. Each task's "
      "profile is only thirty-eight dimensions and the healthy manifold is "
      "densely covered by nearly five thousand training runs, which is close to "
      "the ideal operating regime for a density or subspace method. There was "
      "little headroom to claim: five of the seven methods sit between 0.94 and "
      "0.97 AUPRC."),
("p", "Two baselines do fail, and both failures are informative. The rolling "
      "z-score, which stands in for the rule-based tooling teams actually run "
      "today, collapses to 0.502 AUPRC because the benign distractors defeat it; "
      "seasonality, growth and campaigns move features as hard as several genuine "
      "fault types do, which is precisely why those distractors are in the "
      "benchmark. Isolation Forest is weakest overall, a known consequence of "
      "axis-parallel splits on correlated features."),

("h2", "7.2 The principal finding: cross-task comparability"),
("p", "Ranking nine raw errors asks which task has the largest error. The question "
      "that matters is which task is furthest from its own normal, and because "
      "tasks are not equally predictable those are different questions with "
      "different answers. Two independent corrections exist, and they prove worth "
      "the same."),
("tbl", "Table 7.2  Four scoring rules on the same trained model",
  [["Scoring rule", "Hit@1 [95% CI]", "MRR"],
   ["Deviation magnitude, raw", "0.588 [0.533, 0.646]", "0.762"],
   ["Deviation magnitude, statistically calibrated", "0.773 [0.725, 0.821]",
    "0.872"],
   ["Ancestor-conditioned, structural", "0.773 [0.725, 0.821]", "0.867"],
   ["Ancestor-conditioned and calibrated", "0.797 [0.749, 0.842]", "0.883"]]),
("fig", "artifacts/slides/bw/bw_scoring.png",
        "Figure 7.2  The same model under four scoring rules"),
("p", "Rows two and three are identical to three decimal places, with the same "
      "confidence interval. Structural conditioning achieves comparability by "
      "predicting each task from its parents, so whatever its inputs already "
      "explain is subtracted away. Statistical calibration achieves it by "
      "converting each task's error into a percentile against its own healthy "
      "history. Different mechanisms, the same effect, the same number. Applying "
      "both adds 2.4 points, which the intervals cannot separate from noise."),
("p", "That eighteen-point gain is larger than the entire spread between the seven "
      "methods in Table 7.1, which is roughly four points. On this problem, how "
      "the tasks are scored matters more than which model produces the score."),
("p", "An earlier draft of this work reported the structural correction alone as "
      "worth 18.5 points. That figure compared it against an uncalibrated "
      "baseline, which was not a fair comparison. The corrected version is "
      "reported here, and it identifies what the apparent gain was actually made "
      "of."),

("h2", "7.3 Calibration is not universally beneficial"),
("tbl", "Table 7.3  Effect of per-task calibration on each method",
  [["Method", "Uncalibrated Hit@1", "Calibrated", "Change"],
   ["LSTM autoencoder", "0.643", "0.818", "+0.175"],
   ["Dense autoencoder", "0.725", "0.749", "+0.024"],
   ["Proposed model", "0.773", "0.797", "+0.024"],
   ["PCA residual", "0.801", "0.801", "0.000"],
   ["Local Outlier Factor", "0.797", "0.773", "-0.024"],
   ["Rolling z-score", "0.663", "0.570", "-0.093"]]),
("p", "Calibration transforms the LSTM autoencoder and damages the rolling "
      "z-score. The pattern follows how far apart a method's per-task error "
      "scales already are. The LSTM reconstructs all nine tasks through a single "
      "shared hidden state, so its per-task scales differ widely and there is a "
      "great deal to correct. PCA and Local Outlier Factor already produce "
      "roughly comparable per-task scores, so the rank transform only discards "
      "magnitude information. The practical lesson is that per-task calibration "
      "should be checked rather than assumed, and checking costs nothing because "
      "it requires no labels."),

("h2", "7.4 Localization by graph depth"),
("p", "Section 5.1.5 described a defect in which hiding only the task under test "
      "allowed the model to read the answer from that task's own descendants. If "
      "that account is correct, the leak should hurt most where a task has the "
      "most descendants, which is upstream, and should vanish entirely at the "
      "final task, which has none."),
("tbl", "Table 7.4  Localization accuracy by depth of the originating task",
  [["Origin depth", "Ancestor-conditioned", "Self-masked only", "Deviation "
    "magnitude", "n"],
   ["0 (source tasks)", "0.598", "0.546", "0.505", "97"],
   ["1", "0.806", "0.583", "0.556", "72"],
   ["2", "0.821", "0.643", "0.643", "28"],
   ["3", "0.889", "0.889", "0.667", "18"],
   ["4", "0.938", "0.625", "0.562", "48"],
   ["5 (sink task)", "0.893", "0.893", "0.893", "28"]]),
("p", "At depth five all three rules agree exactly, because the final task has no "
      "descendants and the three masks are then the same tensor. That agreement "
      "is not a tuned result; it falls out of the construction, and it is the "
      "clearest available evidence that the improvement comes from the mechanism "
      "claimed rather than from somewhere else. Depth zero is the hardest row and "
      "gains least, which also follows, since source tasks have no ancestors "
      "either and conditioning has nothing extra to offer."),

("h2", "7.5 Ablation study"),
("p", "Every configuration received an identical budget of forty epochs with the "
      "same early-stopping patience, so that a component appearing to help could "
      "not be confused with a run having trained longer. Three seeds establish "
      "the noise floor at Hit@1 0.805 plus or minus 0.011."),
("tbl", "Table 7.5  Ablation results against a noise floor of 0.011",
  [["Configuration", "Val. recon.", "AUPRC", "Hit@1", "Change", "Verdict"],
   ["Full model, three seeds", "0.438", "0.951", "0.805", "reference", ""],
   ["Clean training set", "0.436", "0.952", "0.814", "+0.009", "within noise"],
   ["No graph attention bias", "0.435", "0.953", "0.801", "-0.005", "within noise"],
   ["No calendar features", "0.433", "0.953", "0.790", "-0.015", "within noise"],
   ["Working calendar features", "0.433", "0.953", "0.787", "-0.018", "within noise"],
   ["Contrastive objective added", "0.486", "0.953", "0.787", "-0.018", "within noise"],
   ["Longer training, 57 epochs", "0.405", "0.948", "0.756", "-0.049",
    "outside noise"]]),
("p", "Four of the design components were removed and none of their removals "
      "changed the outcome. Each null result has a specific explanation."),
("p", "The graph attention bias is redundant because the model recovers the "
      "pipeline structure without it. With the bias removed, attention still "
      "places 42.2 per cent of its off-diagonal weight on true pipeline "
      "connections against 22.2 per cent expected by chance, compared with 46.4 "
      "per cent when the bias is present. Supplying the structure nudges the "
      "model slightly and changes nothing downstream."),
("p", "Calendar features are redundant because a thirty-two hour window already "
      "spans more than a full day, so the phase of the daily cycle is inferable "
      "from the visible history. This ablation had to be run twice. The first "
      "attempt was invalid, because a unit error made the exported timestamps a "
      "thousand times too small, so the features being removed encoded almost "
      "nothing and removing them could not have tested anything. Retrained on "
      "corrected data with a genuine twenty-four hour cycle, the answer was "
      "unchanged."),
("p", "The contrastive objective costs four times the compute per epoch, degrades "
      "validation reconstruction from 0.438 to 0.486, and changes nothing else. "
      "Its own loss collapses to 0.055, meaning the task became trivial. The "
      "augmentations are deliberately weak, because strong amplitude scaling is "
      "indistinguishable from a unit change and heavy dropout from a null spike, "
      "so strengthening them would teach the model that real faults are normal."),
("p", "Contamination tolerance is confirmed. Training on only the clean windows "
      "gives plus 0.009, inside noise, so the method genuinely tolerates the "
      "three per cent contamination that real pipeline history carries."),

("h2", "7.6 An unexpected result: longer training is worse"),
("p", "The single effect outside the noise floor was not one the study set out to "
      "find. Training for fifty-seven epochs rather than thirty-three improves "
      "validation reconstruction from 0.433 to 0.405 and drops localization from "
      "0.805 to 0.756 Hit@1."),
("p", "This is the over-generalisation failure familiar from reconstruction-based "
      "anomaly detection. The method depends on being surprised by corrupted "
      "data. Train the model well enough and it becomes capable of predicting "
      "corrupted data too, so the surprise that detection and localization both "
      "rest on diminishes. The contrastive run supports the same reading from the "
      "opposite direction: clearly worse reconstruction, unchanged localization."),
("p", "The implication for this project's own protocol is uncomfortable and is "
      "stated rather than omitted. Checkpoints were selected on validation "
      "reconstruction loss precisely because that criterion requires no labels. "
      "It is now shown not to track the task the model is being asked to perform, "
      "so the numbers reported here come from a selection rule that is defensible "
      "but demonstrably suboptimal, which if anything understates what the "
      "architecture can do."),

("h2", "7.7 Explainability"),
("tbl", "Table 7.6  Attribution accuracy against the known symptom family",
  [["Method", "Accuracy over 291 incidents"],
   ["Integrated Gradients", "0.935"],
   ["Reconstruction error", "0.924"]]),
("p", "Both are high and the gap is small. Integrated Gradients is clearly better "
      "on timezone_shift, 0.95 against 0.76, which fits its additional "
      "capability: it can credit evidence in other tasks and earlier runs, while "
      "reconstruction error can only inspect the flagged task at the flagged "
      "timestep, and a timestamp shift presents as a relationship across the "
      "window rather than as one large residual. Nine of the fifteen fault types "
      "reach 1.00 under both methods. Since Integrated Gradients costs "
      "thirty-two additional forward and backward passes per explanation for an "
      "average gain of one point, reconstruction error is the sensible default."),
("fig", "artifacts/slides/bw/bw_attention.png",
        "Figure 7.3  Structural attention averaged over incidents"),
("p", "Averaged over incidents, structural attention places 46.4 per cent of its "
      "off-diagonal weight on true pipeline connections against 22.2 per cent "
      "expected by chance, a lift of 2.1 times. Nothing forces this: the graph "
      "bias is a learned scalar per relationship rather than a mask, so the model "
      "was free to ignore the structure and did not."),

("h2", "7.8 Error analysis"),
("p", "Nine of the fifteen fault types localize at or near 1.00. Two fail badly, "
      "and the correlation between how far a fault spreads and how well it "
      "localizes is minus 0.63."),
("tbl", "Table 7.7  The two fault types the method handles poorly",
  [["Fault type", "Hit@1", "Mean tasks reached"],
   ["row_loss_truncation", "0.29", "3.77"],
   ["duplicate_records", "0.43", "2.84"],
   ["Nine other types", "0.94 to 1.00", "1.00 to 1.76"]]),
("p", "The error pattern is sharper than the averages suggest. Of the "
      "mislocalized duplicate_records incidents, every single one blamed a task "
      "further downstream, and nearly all blamed the same join. For "
      "row_loss_truncation the figure is eighty-five per cent."),
("p", "The mechanism is specific to joins. A join's output row count depends on "
      "the key overlap between its inputs, and key overlap is not among the "
      "thirty-eight measurements, so no model can predict a join's row count from "
      "its parents' profiles. When a duplication arrives from upstream, the "
      "fan-out it causes at the join is genuinely unpredictable from the parent "
      "profiles, and the join therefore looks like the origin. This is a "
      "feature-design limitation rather than a modelling one, and it names its "
      "own remedy."),
("fig", "artifacts/results/fig6_localization_breakdown.png",
        "Figure 7.4  Localization accuracy by fault type and by severity quartile"),
("p", "Localization also degrades gracefully with severity, from 0.66 Hit@1 in the "
      "lowest quartile to 0.88 in the highest, which is the expected shape. A "
      "method that scored faint incidents as well as severe ones would warrant "
      "suspicion rather than confidence."),

("h2", "7.9 Cost analysis"),
("tbl", "Table 7.8  Computational cost",
  [["Stage", "Time", "Hardware"],
   ["Benchmark generation, 8000 runs", "4 minutes", "CPU"],
   ["One training run, 40 epochs", "18 minutes", "Apple M4 GPU"],
   ["Nine training runs for seeds and ablations", "about 3 hours", "Apple M4 GPU"],
   ["Full evaluation with six baselines", "6 minutes", "Apple M4 GPU and CPU"],
   ["Inference for one pipeline run", "under one second", "Apple M4 GPU"]]),
("p", "The monetary cost is zero. Every component of the stack is open source, no "
      "dataset was purchased and no compute was rented. Considered as a "
      "prospective deployment, inference over a single hourly run costs a "
      "fraction of a second, which is negligible beside the pipeline being "
      "monitored."),
("pb",),

# ===================== CHAPTER 8 =====================
("h1", "CHAPTER 8\nCONCLUSION"),
("p", "This project set out to detect failures in ETL pipelines without labelled "
      "examples and to identify which task caused them. It delivers a benchmark "
      "that makes the second question answerable, a self-supervised model that "
      "answers both competitively, and an explanation layer whose correctness can "
      "be measured rather than assumed."),
("p", "All four objectives set in Section 2.3 were met against criteria fixed "
      "before the experiments were run. Detection reaches 0.953 AUPRC with no "
      "labels used in training or in checkpoint selection. Localization reaches "
      "0.797 Hit@1 and 0.973 Hit@3. Attribution names the correct symptom family "
      "for 93.5 per cent of incidents."),
("p", "The honest summary of the modelling result is that the Transformer does "
      "not beat simple baselines on this benchmark. Local Outlier Factor detects "
      "better, an LSTM autoencoder localizes better, and the differences among "
      "the leading methods lie within their confidence intervals. What the work "
      "does establish is more specific and arguably more useful: the way in which "
      "per-task scores are made comparable matters more than the model that "
      "produces them. Correcting for cross-task comparability is worth up to "
      "eighteen Hit@1 points, whereas the entire spread between the leading "
      "detectors is about four. Two routes to that correction, one structural and "
      "one statistical, prove equivalent in effect."),
("p", "Four design components were tested and none helped. Reporting that is part "
      "of the result rather than an embarrassment. A graph attention bias adds "
      "nothing because the model recovers the pipeline structure from data on its "
      "own; explicit calendar features add nothing because the observation window "
      "already contains a full daily cycle; a contrastive objective adds nothing "
      "while degrading reconstruction; and a guaranteed-clean training set is no "
      "better than contaminated history, which confirms the contamination "
      "tolerance the method claims."),
("p", "One effect exceeded the noise floor and it was unwelcome. Training longer "
      "improves reconstruction and worsens localization, which undermines the "
      "label-free checkpoint criterion used throughout, since validation "
      "reconstruction loss turns out to be poorly aligned with the task."),

("h2", "8.1 Future work"),
("p", "Four directions follow, in the order they should be attempted."),
("b", "Measure join-key overlap. Section 7.8 identifies the exact missing "
      "measurement behind the two fault types the method handles poorly, and the "
      "error analysis predicts that supplying it should repair them. It is a "
      "contained change and the highest-value next step."),
("b", "Find a label-free checkpoint criterion that tracks localization. Section "
      "7.6 shows the present one does not. This is the most interesting open "
      "problem the work surfaced."),
("b", "Validate the simulator against real telemetry. Even a few weeks of real "
      "pipeline metrics containing a handful of known incidents would allow the "
      "fault severity and frequency distributions to be calibrated rather than "
      "chosen."),
("b", "Harden the detection benchmark. Five of seven methods land within 0.03 "
      "AUPRC, so the benchmark separates methods on localization but barely on "
      "detection. Lower severities, more concurrent incidents and shorter "
      "windows would all help."),
("pb",),
]

REFERENCES = [
 "P. Malhotra, L. Vig, G. Shroff and P. Agarwal, \"Long short term memory "
 "networks for anomaly detection in time series,\" in Proc. European Symp. "
 "Artificial Neural Networks (ESANN), Bruges, Belgium, 2015, pp. 89-94.",

 "P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal and G. Shroff, "
 "\"LSTM-based encoder-decoder for multi-sensor anomaly detection,\" in Proc. "
 "ICML Anomaly Detection Workshop, New York, NY, USA, 2016.",

 "L. Ruff et al., \"Deep one-class classification,\" in Proc. 35th Int. Conf. "
 "Machine Learning (ICML), Stockholm, Sweden, 2018, pp. 4393-4402.",

 "K. Hundman, V. Constantinou, C. Laporte, I. Colwell and T. Soderstrom, "
 "\"Detecting spacecraft anomalies using LSTMs and nonparametric dynamic "
 "thresholding,\" in Proc. 24th ACM SIGKDD Int. Conf. Knowledge Discovery and "
 "Data Mining (KDD), London, UK, 2018, pp. 387-395.",

 "Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun and D. Pei, \"Robust anomaly "
 "detection for multivariate time series through stochastic recurrent neural "
 "network,\" in Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data "
 "Mining (KDD), Anchorage, AK, USA, 2019, pp. 2828-2837.",

 "J. Audibert, P. Michiardi, F. Guyard, S. Marti and M. A. Zuluaga, \"USAD: "
 "unsupervised anomaly detection on multivariate time series,\" in Proc. 26th "
 "ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), 2020, pp. "
 "3395-3404.",

 "H. Zhao et al., \"Multivariate time-series anomaly detection via graph "
 "attention network,\" in Proc. IEEE Int. Conf. Data Mining (ICDM), Sorrento, "
 "Italy, 2020, pp. 841-850.",

 "A. Deng and B. Hooi, \"Graph neural network-based anomaly detection in "
 "multivariate time series,\" in Proc. AAAI Conf. Artificial Intelligence, "
 "vol. 35, no. 5, 2021, pp. 4027-4035.",

 "S. Tuli, G. Casale and N. R. Jennings, \"TranAD: deep transformer networks "
 "for anomaly detection in multivariate time series data,\" Proc. VLDB "
 "Endowment, vol. 15, no. 6, pp. 1201-1214, 2022.",

 "J. Xu, H. Wu, J. Wang and M. Long, \"Anomaly Transformer: time series "
 "anomaly detection with association discrepancy,\" in Proc. Int. Conf. "
 "Learning Representations (ICLR), 2022.",

 "G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty and C. Eickhoff, \"A "
 "transformer-based framework for multivariate time series representation "
 "learning,\" in Proc. 27th ACM SIGKDD Conf. Knowledge Discovery and Data "
 "Mining (KDD), 2021, pp. 2114-2124.",

 "Z. Yue et al., \"TS2Vec: towards universal representation of time series,\" "
 "in Proc. AAAI Conf. Artificial Intelligence, vol. 36, no. 8, 2022, pp. "
 "8980-8987.",

 "K. He, X. Chen, S. Xie, Y. Li, P. Dollar and R. Girshick, \"Masked "
 "autoencoders are scalable vision learners,\" in Proc. IEEE/CVF Conf. "
 "Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16000-16009.",

 "M. Du, F. Li, G. Zheng and V. Srikumar, \"DeepLog: anomaly detection and "
 "diagnosis from system logs through deep learning,\" in Proc. ACM SIGSAC "
 "Conf. Computer and Communications Security (CCS), Dallas, TX, USA, 2017, "
 "pp. 1285-1298.",

 "W. Meng et al., \"LogAnomaly: unsupervised detection of sequential and "
 "quantitative anomalies in unstructured logs,\" in Proc. 28th Int. Joint "
 "Conf. Artificial Intelligence (IJCAI), Macao, China, 2019, pp. 4739-4745.",

 "H. Guo, S. Yuan and X. Wu, \"LogBERT: log anomaly detection via BERT,\" in "
 "Proc. Int. Joint Conf. Neural Networks (IJCNN), Shenzhen, China, 2021, pp. "
 "1-8.",

 "S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann and A. "
 "Grafberger, \"Automating large-scale data quality verification,\" Proc. "
 "VLDB Endowment, vol. 11, no. 12, pp. 1781-1794, 2018.",

 "E. Breck, N. Polyzotis, S. Roy, S. E. Whang and M. Zinkevich, \"Data "
 "validation for machine learning,\" in Proc. Conf. Systems and Machine "
 "Learning (SysML), Stanford, CA, USA, 2019.",

 "J. Song and Y. He, \"Auto-Validate: unsupervised data validation using "
 "data-domain patterns inferred from data lakes,\" in Proc. ACM SIGMOD Int. "
 "Conf. Management of Data, 2021, pp. 1678-1691.",

 "S. Redyuk, Z. Kaoudi, V. Markl and S. Schelter, \"Automating data quality "
 "validation for dynamic data ingestion,\" in Proc. 24th Int. Conf. Extending "
 "Database Technology (EDBT), 2021, pp. 61-72.",

 "L. Wu, J. Tordsson, E. Elmroth and O. Kao, \"MicroRCA: root cause "
 "localization of performance issues in microservices,\" in Proc. IEEE/IFIP "
 "Network Operations and Management Symp. (NOMS), Budapest, Hungary, 2020, "
 "pp. 1-9.",

 "M. Li et al., \"Causal inference-based root cause analysis for online "
 "service systems with intervention recognition,\" in Proc. 28th ACM SIGKDD "
 "Conf. Knowledge Discovery and Data Mining (KDD), Washington, DC, USA, 2022, "
 "pp. 3230-3240.",

 "A. Ikram, S. Chakraborty, S. Mitra, S. Saini, S. Bagchi and M. Kocaoglu, "
 "\"Root cause analysis of failures in microservices through causal "
 "discovery,\" in Advances in Neural Information Processing Systems "
 "(NeurIPS), vol. 35, 2022, pp. 31158-31170.",

 "M. Sundararajan, A. Taly and Q. Yan, \"Axiomatic attribution for deep "
 "networks,\" in Proc. 34th Int. Conf. Machine Learning (ICML), Sydney, "
 "Australia, 2017, pp. 3319-3328.",

 "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model "
 "predictions,\" in Advances in Neural Information Processing Systems "
 "(NeurIPS), vol. 30, 2017, pp. 4765-4774.",

 "S. Kim, K. Choi, H.-S. Choi, B. Lee and S. Yoon, \"Towards a rigorous "
 "evaluation of time-series anomaly detection,\" in Proc. AAAI Conf. "
 "Artificial Intelligence, vol. 36, no. 7, 2022, pp. 7194-7201.",

 "R. Wu and E. J. Keogh, \"Current time series anomaly detection benchmarks "
 "are flawed and are creating the illusion of progress,\" IEEE Trans. "
 "Knowledge and Data Engineering, vol. 35, no. 3, pp. 2421-2429, 2023.",

 "F. T. Liu, K. M. Ting and Z.-H. Zhou, \"Isolation forest,\" in Proc. 8th "
 "IEEE Int. Conf. Data Mining (ICDM), Pisa, Italy, 2008, pp. 413-422.",

 "M. M. Breunig, H.-P. Kriegel, R. T. Ng and J. Sander, \"LOF: identifying "
 "density-based local outliers,\" in Proc. ACM SIGMOD Int. Conf. Management "
 "of Data, Dallas, TX, USA, 2000, pp. 93-104.",

 "A. Vaswani et al., \"Attention is all you need,\" in Advances in Neural "
 "Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 5998-6008.",
]

ABBREVIATIONS = [
 ("AUPRC", "Area Under the Precision-Recall Curve"),
 ("AUROC", "Area Under the Receiver Operating Characteristic Curve"),
 ("CI", "Confidence Interval"),
 ("CSV", "Comma Separated Values"),
 ("DAG", "Directed Acyclic Graph"),
 ("DFD", "Data Flow Diagram"),
 ("ETL", "Extract, Transform and Load"),
 ("FK", "Foreign Key"),
 ("GAT", "Graph Attention Network"),
 ("GNN", "Graph Neural Network"),
 ("GPU", "Graphics Processing Unit"),
 ("IQR", "Interquartile Range"),
 ("LOF", "Local Outlier Factor"),
 ("LSTM", "Long Short-Term Memory"),
 ("MAE", "Masked Autoencoder"),
 ("MPS", "Metal Performance Shaders"),
 ("MRR", "Mean Reciprocal Rank"),
 ("PCA", "Principal Component Analysis"),
 ("PK", "Primary Key"),
 ("RCA", "Root Cause Analysis"),
 ("SLA", "Service Level Agreement"),
 ("SSL", "Self-Supervised Learning"),
 ("TFDV", "TensorFlow Data Validation"),
 ("UTC", "Coordinated Universal Time"),
]

SYMBOLS = [
 ("N", "Number of tasks in the pipeline graph, nine in this work"),
 ("F", "Number of features in a task profile, thirty-eight in this work"),
 ("T", "Number of scheduled runs in the history, eight thousand in this work"),
 ("W", "Length of the observation window in runs, thirty-two in this work"),
 ("X", "The observed history tensor, of shape T by N by F"),
 ("v", "The task at which an incident originates"),
 ("e_hist", "Reconstruction error with every task hidden, the detection signal"),
 ("e_anc", "Reconstruction error with a task and its descendants hidden"),
 ("tau", "Deviation threshold in robust units for the measured propagation label"),
]

FIGURES = [
 ('4.1', 'System architecture, five layers with one-directional flow', '11'),
 ('4.2', 'Level-1 data flow diagram', '11'),
 ('4.3', 'Use case diagram', '12'),
 ('4.4', 'Class diagram of the principal components', '12'),
 ('4.5', 'Sequence diagram for one diagnosis', '13'),
 ('6.1', 'The web demonstration diagnosing a single pipeline run', '19'),
 ('7.1', 'Detection and localization across all seven methods', '21'),
 ('7.2', 'The same model under four scoring rules', '22'),
 ('7.3', 'Structural attention averaged over incidents', '26'),
 ('7.4', 'Localization accuracy by fault type and by severity quartile', '27'),
]

TABLES = [
 ('2.1', 'Objectives and their success criteria', '5'),
 ('2.2', 'Project plan and phases', '6'),
 ('3.1', 'Functional requirements', '8'),
 ('3.2', 'Non-functional requirements', '8'),
 ('3.3', 'Hardware used', '10'),
 ('3.4', 'Software stack', '10'),
 ('5.1', 'Fault taxonomy, fifteen types across six families', '14'),
 ('5.2', 'The eight symptom families', '15'),
 ('5.3', 'Test coverage by category', '17'),
 ('7.1', 'Comparison against six baselines, test split', '21'),
 ('7.2', 'Four scoring rules on the same trained model', '22'),
 ('7.3', 'Effect of per-task calibration on each method', '23'),
 ('7.4', 'Localization accuracy by depth of the originating task', '23'),
 ('7.5', 'Ablation results against a noise floor', '24'),
 ('7.6', 'Attribution accuracy against the known symptom family', '25'),
 ('7.7', 'The two fault types the method handles poorly', '26'),
 ('7.8', 'Computational cost', '27'),
]

TOC = [
 ('', 'List of Figures', 'i'),
 ('', 'List of Tables', 'ii'),
 ('', 'List of Abbreviations', 'iii'),
 ('', 'Symbols and Notations', 'iv'),
 ('', 'Abstract', 'v'),
 ('1.', 'INTRODUCTION', '1'),
 ('', '1.1 Background', '1'),
 ('', '1.2 Motivations', '1'),
 ('', '1.3 Scope of the Project', '2'),
 ('2.', 'PROJECT DESCRIPTION AND GOALS', '3'),
 ('', '2.1 Literature Review', '3'),
 ('', '2.2 Research Gap', '4'),
 ('', '2.3 Objectives', '5'),
 ('', '2.4 Problem Statement', '6'),
 ('', '2.5 Project Plan', '6'),
 ('3.', 'TECHNICAL SPECIFICATION', '8'),
 ('', '3.1 Requirements', '8'),
 ('', '3.2 Feasibility Study', '9'),
 ('', '3.3 System Specification', '9'),
 ('4.', 'DESIGN APPROACH AND DETAILS', '11'),
 ('', '4.1 System Architecture', '11'),
 ('', '4.2 Design', '11'),
 ('5.', 'METHODOLOGY AND TESTING', '14'),
 ('', '5.1 Module Description', '14'),
 ('', '5.2 Testing', '17'),
 ('6.', 'PROJECT DEMONSTRATION', '19'),
 ('7.', 'RESULT AND DISCUSSION', '21'),
 ('', '7.1 Main comparison', '21'),
 ('', '7.2 The principal finding', '22'),
 ('', '7.3 Calibration is not universally beneficial', '23'),
 ('', '7.4 Localization by graph depth', '23'),
 ('', '7.5 Ablation study', '24'),
 ('', '7.6 An unexpected result', '25'),
 ('', '7.7 Explainability', '25'),
 ('', '7.8 Error analysis', '26'),
 ('', '7.9 Cost analysis', '27'),
 ('8.', 'CONCLUSION', '28'),
 ('9.', 'REFERENCES', '30'),
 ('', 'APPENDIX A - SAMPLE CODE', '33'),
]
