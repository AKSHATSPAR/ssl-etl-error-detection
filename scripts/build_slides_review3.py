"""Build the Review 3 deck, black and white only.

    PYTHONPATH=src python scripts/build_slides_review3.py

Monochrome is a real constraint, not a filter applied at the end. With no hue to
carry meaning, every distinction has to be made with value, weight, size or
rule. So emphasis here is a solid black panel with reversed text, de-emphasis is
a light grey fill, and the figures are redrawn rather than desaturated, because
desaturating a colour chart collapses distinct hues into near-identical greys.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path("artifacts/slides/ETL_Review3_BW.pptx")
BW = Path("artifacts/slides/bw")

BLACK = RGBColor(0x00, 0x00, 0x00)
INK   = RGBColor(0x1A, 0x1A, 0x1A)
DARK  = RGBColor(0x4D, 0x4D, 0x4D)
MID   = RGBColor(0x75, 0x75, 0x75)
LIGHT = RGBColor(0xD9, 0xD9, 0xD9)
PALE  = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

HEAD = "Cambria"
BODY = "Calibri"
W, H = 13.333, 7.5


def deck():
    p = Presentation()
    p.slide_width, p.slide_height = Inches(W), Inches(H)
    return p


def blank(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = BLACK if dark else WHITE
    return s


def text(sl, x, y, w, h, runs, size=15, color=INK, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, space=6, line=None, anchor=MSO_ANCHOR.TOP):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    items = [runs] if isinstance(runs, str) else runs
    for i, item in enumerate(items):
        body, over = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = over.get("align", align)
        p.space_after = Pt(over.get("space", space))
        if line or over.get("line"):
            p.line_spacing = over.get("line", line)
        r = p.add_run()
        r.text = body
        fo = r.font
        fo.size = Pt(over.get("size", size))
        fo.bold = over.get("bold", bold)
        fo.italic = over.get("italic", False)
        fo.name = over.get("font", font)
        fo.color.rgb = over.get("color", color)
    return tb


def box(sl, x, y, w, h, fill, outline=None, shape=MSO_SHAPE.RECTANGLE):
    sh = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if outline is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = outline
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def title(sl, t, sub=None, dark=False):
    text(sl, 0.85, 0.6, 11.6, 1.0, t, size=33, bold=True, font=HEAD,
         color=WHITE if dark else BLACK)
    if sub:
        text(sl, 0.85, 1.45, 11.6, 0.6, sub, size=14.5,
             color=LIGHT if dark else DARK, font=BODY)
    # A single hairline rule instead of a colour accent, since there is no colour.
    box(sl, 0.85, 2.02, 11.6, 0.012, LIGHT if dark else RGBColor(0xC0, 0xC0, 0xC0))


def pic(sl, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return sl.shapes.add_picture(str(BW / name), Inches(x), Inches(y), **kw)


def notes(sl, s):
    sl.notes_slide.notes_text_frame.text = s


def numbered(sl, x, y, n, dark=False):
    c = box(sl, x, y, 0.42, 0.42, WHITE if dark else BLACK,
            shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run()
    r.text = str(n)
    r.font.size = Pt(15); r.font.bold = True; r.font.name = HEAD
    r.font.color.rgb = BLACK if dark else WHITE


def build():
    prs = deck()

    # 1 --------------------------------------------------------------- title
    s = blank(prs, dark=True)
    text(s, 1.0, 2.3, 11.3, 1.7,
         "Self-Supervised AI for ETL Error Detection\nand Root-Cause Localization",
         size=38, bold=True, font=HEAD, color=WHITE, line=1.15)
    box(s, 1.0, 4.25, 1.8, 0.03, WHITE)
    text(s, 1.0, 4.65, 11.3, 1.4,
         [("Review 3", {"size": 17, "bold": True, "color": WHITE, "font": HEAD}),
          ("Finding the one task that broke a data pipeline, "
           "without any labelled failures", {"size": 15, "color": LIGHT})],
         line=1.3)
    text(s, 1.0, 6.3, 11.3, 0.5, "B.Tech Deep Learning Project",
         size=13, color=MID)
    notes(s, "Open with the one-line version: pipelines break quietly, every step "
             "downstream looks broken, and this finds the one that actually caused it.")

    # 2 ------------------------------------------------------------- problem
    s = blank(prs)
    title(s, "The problem",
          "The job finishes successfully and the data is wrong")
    for i, (h, b, dark) in enumerate([
        ("What people expect",
         "A job crashes. Someone is alerted within minutes. It is fixed the "
         "same day.", False),
        ("What actually happens",
         "The job succeeds. A price field switched from rupees to paise "
         "upstream. Every number is now a hundred times too big.", False),
        ("The cost",
         "Nobody notices for weeks. Every report built on that table is wrong "
         "for the whole period.", True)]):
        x = 0.85 + i * 3.95
        box(s, x, 2.5, 3.6, 3.3, BLACK if dark else PALE)
        text(s, x + 0.3, 2.8, 3.0, 0.45, h, size=16, bold=True, font=HEAD,
             color=WHITE if dark else BLACK)
        text(s, x + 0.3, 3.4, 3.0, 2.2, b, size=13.5,
             color=LIGHT if dark else DARK, line=1.25)
    text(s, 0.85, 6.2, 11.6, 0.6,
         "Nothing failed, so nothing alerted. That is the class of failure this "
         "project targets.", size=14.5, bold=True, color=BLACK)
    notes(s, "Give the concrete example. A currency convention changes and amounts "
             "become 100x too large. No exception is raised anywhere.")

    # 3 ---------------------------------------------------------- objectives
    s = blank(prs)
    title(s, "Objectives, and whether they were met",
          "Success criteria were fixed before the experiments ran")
    rows = [("1", "Detect broken runs using no labels in training",
             "AUPRC above 0.90", "0.953"),
            ("2", "Identify the task that originated the incident",
             "Top-1 above 0.50, top-3 above 0.90", "0.797 / 0.973"),
            ("3", "Explain each finding as a named symptom",
             "Correct symptom family above 0.80", "0.935"),
            ("4", "Evaluate honestly against real baselines",
             "Chronological splits, intervals, six baselines", "Met")]
    box(s, 0.85, 2.35, 11.6, 0.5, BLACK)
    for lbl, x, w in [("Objective", 1.55, 4.6), ("Criterion set in advance", 6.3, 3.6),
                      ("Result", 10.2, 2.0)]:
        text(s, x, 2.48, w, 0.3, lbl, size=11.5, bold=True, color=WHITE,
             font=HEAD)
    for i, (n, obj, crit, res) in enumerate(rows):
        y = 2.95 + i * 0.95
        if i % 2 == 0:
            box(s, 0.85, y, 11.6, 0.88, PALE)
        numbered(s, 1.0, y + 0.22, n)
        text(s, 1.55, y + 0.26, 4.6, 0.5, obj, size=13, color=INK)
        text(s, 6.3, y + 0.26, 3.6, 0.5, crit, size=12, color=DARK)
        text(s, 10.2, y + 0.2, 2.0, 0.5, res, size=15, bold=True, font=HEAD,
             color=BLACK)
    text(s, 0.85, 6.9, 11.6, 0.5,
         "Objective 2 is the one that defines the project. The other three are "
         "achievable by existing methods.", size=13, color=DARK)
    notes(s, "This is the slide a review panel most wants. Criteria were set before "
             "running anything, so success could not be redefined afterwards.")

    # 4 --------------------------------------------------------- why it is hard
    s = blank(prs)
    title(s, "Why locating the cause is hard",
          "One fault, and almost every task looks broken")
    pic(s, "bw_propagation.png", 0.85, 2.35, w=8.3)
    box(s, 9.45, 2.35, 3.0, 3.0, BLACK)
    text(s, 9.75, 2.62, 2.4, 2.5,
         ["The fault was injected into one task only.", "",
          "Eight of nine tasks move.", "",
          "Several move harder than the one that caused it."],
         size=12.5, color=LIGHT, line=1.25)
    text(s, 0.85, 6.25, 11.6, 0.6,
         "The damage grows as it travels, so the loudest signal is usually "
         "furthest from the fault.", size=14.5, bold=True, color=BLACK)
    notes(s, "Every one of those tasks would trip an ordinary alarm. Only one is "
             "the cause. Ranking by severity picks the wrong one.")

    # 5 --------------------------------------------------------- the dataset
    s = blank(prs)
    title(s, "Dataset: built, because none exists",
          "No public benchmark labels which pipeline task caused a failure")
    text(s, 0.85, 2.4, 6.3, 2.4,
         ["Existing datasets are server and sensor readings. They contain no "
          "pipeline, so the question cannot be asked of them.", "",
          "Research using real pipeline data uses company data that is never "
          "released."], size=14, color=DARK, line=1.3)
    box(s, 0.85, 5.0, 6.3, 1.5, PALE)
    text(s, 1.15, 5.3, 5.7, 1.0,
         "Constructing the benchmark is part of the contribution, not a "
         "shortcut around a missing one.", size=14, bold=True, color=BLACK,
         line=1.25)
    for i, (v, l) in enumerate([("8,000", "pipeline runs"), ("9", "tasks each"),
                                ("15", "fault types"), ("273", "labelled incidents")]):
        x = 7.6 + (i % 2) * 2.5
        y = 2.4 + (i // 2) * 2.0
        text(s, x, y, 2.2, 1.0, v, size=44, bold=True, font=HEAD, color=BLACK)
        text(s, x, y + 1.05, 2.2, 0.5, l, size=12.5, color=MID)
    notes(s, "Expect 'why not real data'. Answer: no public dataset labels the "
             "causing step, and company pipeline data is not released.")

    # 6 ------------------------------------------------- understanding the data
    s = blank(prs)
    title(s, "What the data looks like",
          "Four properties that drove decisions later in the method")
    facts = [
        ("92.6% healthy, 7.4% broken",
         "Answering 'healthy' every time scores 92.6% accuracy and is useless, "
         "so accuracy is never reported."),
        ("Training history is 3% contaminated",
         "Real history contains failures nobody noticed. The method must "
         "tolerate that rather than assume a clean training set."),
        ("22 of 38 features are heavily skewed",
         "Row counts and runtimes are long tailed, which is why normalisation "
         "uses median and IQR rather than mean and standard deviation."),
        ("Volume swings 9.6x, and none of it is a fault",
         "311 rows at the quietest, 2984 at the busiest. Any threshold on row "
         "count fires constantly."),
    ]
    for i, (h, b) in enumerate(facts):
        x = 0.85 + (i % 2) * 5.95
        y = 2.45 + (i // 2) * 2.2
        box(s, x, y, 5.6, 1.95, PALE)
        text(s, x + 0.3, y + 0.28, 5.0, 0.45, h, size=14.5, bold=True,
             font=HEAD, color=BLACK)
        text(s, x + 0.3, y + 0.82, 5.0, 1.0, b, size=12.5, color=DARK, line=1.22)
    notes(s, "These are not decorative statistics. Each one forced a specific "
             "design choice, and the slide says which.")

    # 7 ----------------------------------------------------------- method 1
    s = blank(prs)
    title(s, "Method: learning without labels",
          "Hide part of a run, then predict what was hidden")
    for i, (n, h, b) in enumerate([
        ("1", "Watch normal runs",
         "Thousands of ordinary runs. Nobody says which are good or bad."),
        ("2", "Hide something",
         "Cover one task's numbers, or one family of measurements."),
        ("3", "Predict it back",
         "Fill the gap using the other tasks and the recent history."),
        ("4", "Measure surprise",
         "Once it predicts well, whatever it cannot predict is unusual.")]):
        x = 0.85 + i * 3.03
        box(s, x, 2.45, 2.75, 3.2, PALE)
        numbered(s, x + 0.3, 2.75, n)
        text(s, x + 0.3, 3.4, 2.2, 0.5, h, size=14, bold=True, font=HEAD,
             color=BLACK)
        text(s, x + 0.3, 4.0, 2.2, 1.4, b, size=12, color=DARK, line=1.2)
    text(s, 0.85, 6.05, 11.6, 0.6,
         "The same idea as covering a word in a sentence and predicting it. "
         "Doing that well requires learning how the thing behaves.",
         size=14, color=DARK)
    notes(s, "The sentence analogy is what lands with non-specialists.")

    # 8 ----------------------------------------------------------- method 2
    s = blank(prs)
    title(s, "Method: separating the cause from the victims",
          "Ask each task whether it is surprising given what it was fed")
    box(s, 0.85, 2.45, 5.6, 1.95, PALE)
    text(s, 1.15, 2.75, 5.0, 0.4, "A victim", size=17, bold=True, font=HEAD,
         color=BLACK)
    text(s, 1.15, 3.25, 5.0, 1.0,
         "Bad output, but it follows exactly from the bad input it was given. "
         "Nothing surprising.", size=13.5, color=DARK, line=1.22)
    box(s, 0.85, 4.65, 5.6, 1.95, BLACK)
    text(s, 1.15, 4.95, 5.0, 0.4, "The cause", size=17, bold=True, font=HEAD,
         color=WHITE)
    text(s, 1.15, 5.45, 5.0, 1.0,
         "Healthy input, and bad output anyway. That is where it began.",
         size=13.5, color=LIGHT, line=1.22)
    box(s, 7.0, 2.45, 5.45, 4.15, PALE)
    text(s, 7.3, 2.75, 4.85, 0.45, "A mistake worth reporting", size=16,
         bold=True, font=HEAD, color=BLACK)
    text(s, 7.3, 3.35, 4.85, 3.0,
         "The first version hid only the task being tested. The model could "
         "still see that task's own downstream steps, which were also broken, "
         "and read the answer off them.\n\n"
         "Hiding the task and everything downstream fixed it.\n\n"
         "Accuracy went from 63% to 77%.",
         size=13, color=DARK, line=1.25)
    notes(s, "Be honest about the leak. It shows the idea was tested rather than "
             "trusted, and the 63 to 77 jump is the evidence.")

    # 9 ------------------------------------------------------------- setup
    s = blank(prs)
    title(s, "Experimental setup", "Chosen so the numbers mean something")
    items = [
        ("Chronological splits", "Train on the past, test on the future. A random "
         "split over a time series leaks tomorrow into today."),
        ("Contaminated training set", "3% of training runs carry real incidents, "
         "as real history does."),
        ("Label-free model selection", "Checkpoints chosen on reconstruction loss. "
         "Choosing on accuracy would make the method weakly supervised."),
        ("Six baselines", "Rolling z-score, PCA, Isolation Forest, LOF, dense "
         "autoencoder, LSTM autoencoder, all given the same calibration."),
        ("No point-adjusted F1 as a headline", "Published work shows random scores "
         "beat real models under that metric."),
    ]
    for i, (h, b) in enumerate(items):
        y = 2.4 + i * 0.92
        if i % 2 == 0:
            box(s, 0.85, y, 11.6, 0.85, PALE)
        text(s, 1.15, y + 0.14, 3.6, 0.5, h, size=13.5, bold=True, color=BLACK)
        text(s, 5.0, y + 0.14, 7.2, 0.6, b, size=12.5, color=DARK, line=1.2)
    notes(s, "A review panel checks protocol as much as results. Each line here "
             "closes a specific way of accidentally cheating.")

    # 10 ----------------------------------------------------------- results
    s = blank(prs)
    title(s, "Results", "Test split: 2000 runs, 291 of them broken, never seen in training")
    pic(s, "bw_methods.png", 0.85, 2.35, w=8.4)
    box(s, 9.55, 2.35, 2.9, 3.0, BLACK)
    text(s, 9.85, 2.62, 2.3, 2.6,
         ["Our model does not win.", "",
          "One method detects better, another localizes better.", "",
          "The top four are within measurement error."],
         size=12.5, color=LIGHT, line=1.25)
    text(s, 0.85, 6.3, 11.6, 0.7,
         "Saying so is the honest result. The useful finding turned out to be "
         "about how tasks are scored, not which model produces the score.",
         size=14.5, bold=True, color=BLACK, line=1.2)
    notes(s, "Do not soften this. The reason is that the data is small and simple "
             "enough that classical methods do well. The next slide is the finding.")

    # 11 ------------------------------------------------------- main finding
    s = blank(prs)
    title(s, "The main finding",
          "Two different fixes for the same problem, worth exactly the same")
    pic(s, "bw_scoring.png", 0.85, 2.3, w=7.4)
    box(s, 8.6, 2.3, 3.85, 3.5, BLACK)
    text(s, 8.9, 2.6, 3.25, 3.0,
         ["Rows two and three are identical.", "",
          "One compares a task to its own history. The other asks whether it is "
          "surprising given its inputs.", "",
          "Doing both adds almost nothing."],
         size=12.5, color=LIGHT, line=1.25)
    text(s, 0.85, 6.1, 11.6, 0.9,
         "That 18 point gain is larger than the whole spread between the seven "
         "methods, which is about 4 points. How you score matters more than "
         "which model you use.", size=14.5, bold=True, color=BLACK, line=1.2)
    notes(s, "Different tasks are naturally harder to predict, so raw scores are "
             "not comparable. Both fixes correct for that, in different ways.")

    # 12 --------------------------------------------------------- ablations
    s = blank(prs)
    title(s, "What did not work", "Each design idea removed, then measured")
    for i, (h, v, b) in enumerate([
        ("Telling the model the pipeline shape", "no change",
         "It works the shape out from the data by itself."),
        ("Giving it a clock", "no change",
         "A 32-hour window already contains a full day."),
        ("Adding contrastive learning", "no change",
         "Four times the compute, no benefit."),
        ("Training on perfectly clean data", "no change",
         "It tolerates messy history, which was the claim."),
        ("Training for longer", "WORSE", "80% down to 76%, and the reason matters.")]):
        y = 2.35 + i * 0.93
        last = i == 4
        box(s, 0.85, y, 11.6, 0.86, BLACK if last else PALE)
        text(s, 1.15, y + 0.13, 5.0, 0.4, h, size=13.5, bold=True,
             color=WHITE if last else BLACK)
        text(s, 1.15, y + 0.5, 5.0, 0.32, b, size=11.5,
             color=LIGHT if last else MID)
        text(s, 6.6, y + 0.24, 2.2, 0.4, v, size=14, bold=True, font=HEAD,
             color=WHITE if last else DARK)
    text(s, 0.85, 7.0, 11.6, 0.4,
         "Four null results, reported rather than buried.", size=13.5,
         bold=True, color=BLACK)
    notes(s, "Null results with reasons, not shrugs. The DAG hint is redundant "
             "because the model recovers the structure anyway, which we checked.")

    # 13 ---------------------------------------------------------- surprise
    s = blank(prs)
    title(s, "Training longer made it worse",
          "And it undermines our own stopping rule")
    box(s, 0.85, 2.45, 5.6, 1.5, PALE)
    text(s, 1.15, 2.78, 5.0, 0.9,
         "Trained for longer, the model got better at its training task.",
         size=14.5, color=BLACK, line=1.22)
    box(s, 0.85, 4.15, 5.6, 1.5, BLACK)
    text(s, 1.15, 4.48, 5.0, 0.9,
         "And worse at finding the cause. 80% down to 76%.",
         size=14.5, bold=True, color=WHITE, line=1.22)
    box(s, 7.0, 2.45, 5.45, 3.2, PALE)
    text(s, 7.3, 2.75, 4.85, 0.45, "Why", size=17, bold=True, font=HEAD,
         color=BLACK)
    text(s, 7.3, 3.35, 4.85, 2.1,
         "The model works by being surprised by broken data. Train it long "
         "enough and it learns to predict the broken data too, so the surprise "
         "disappears.", size=13.5, color=DARK, line=1.25)
    text(s, 0.85, 6.0, 11.6, 1.0,
         "We chose when to stop training using a measure that turns out not to "
         "track what we care about. It is written up as an open problem rather "
         "than hidden.", size=14, color=DARK, line=1.22)
    notes(s, "If asked what you would do differently, this is the answer. We need "
             "a stopping rule that is label-free but tracks localization.")

    # 14 ---------------------------------------------------- explainability
    s = blank(prs)
    title(s, "Explaining the answer", "Not just which task, but what kind of problem")
    pic(s, "bw_attention.png", 0.85, 2.3, h=4.3)
    box(s, 6.5, 2.3, 5.95, 2.0, PALE)
    text(s, 6.8, 2.6, 5.35, 1.5,
         [("It found the pipeline on its own",
           {"size": 16, "bold": True, "font": HEAD, "color": BLACK}),
          ("The dark cells are the real connections between tasks. Nobody told "
           "it which task feeds which.", {"size": 12.5, "color": DARK})])
    box(s, 6.5, 4.6, 5.95, 2.0, BLACK)
    text(s, 6.8, 4.9, 5.35, 1.5,
         [("It names the right symptom 93% of the time",
           {"size": 16, "bold": True, "font": HEAD, "color": WHITE}),
          ("Scored against what each fault should look like, not just "
           "displayed as a picture.", {"size": 12.5, "color": LIGHT})])
    notes(s, "Most papers show an explanation and never check it. Our simulator "
             "knows what each fault should look like, so we can score it.")

    # 15 -------------------------------------------------------- limitations
    s = blank(prs)
    title(s, "Limitations", "Stated before being asked")
    for i, (h, b) in enumerate([
        ("The data is simulated",
         "We chose how often and how severely faults occur. The ranking of "
         "methods is more trustworthy than any single number."),
        ("Two fault types localize badly",
         "Both involve row counts. A join's row count depends on key overlap, "
         "which we never measure, so the model lacks the information."),
        ("Detection is too easy here",
         "Five of seven methods land within 0.03, so the benchmark barely "
         "separates them on that half of the problem."),
        ("One pipeline shape",
         "Everything is measured on a single nine-task pipeline.")]):
        x = 0.85 + (i % 2) * 5.95
        y = 2.45 + (i // 2) * 2.2
        box(s, x, y, 5.6, 1.95, PALE)
        text(s, x + 0.3, y + 0.28, 5.0, 0.45, h, size=14.5, bold=True,
             font=HEAD, color=BLACK)
        text(s, x + 0.3, y + 0.82, 5.0, 1.0, b, size=12.5, color=DARK, line=1.22)
    notes(s, "Volunteering limitations is stronger than conceding them. Each has a "
             "specific reason and, where relevant, a named fix.")

    # 16 -------------------------------------------------------- conclusion
    s = blank(prs, dark=True)
    text(s, 1.0, 1.5, 11.3, 0.8, "What was delivered", size=32, bold=True,
         font=HEAD, color=WHITE)
    box(s, 1.0, 2.45, 1.8, 0.03, WHITE)
    for i, (h, b) in enumerate([
        ("A benchmark", "8,000 runs, 15 fault types, cause labels measured by "
                        "re-running each faulty batch without the fault"),
        ("A method", "Learns normal behaviour with no labels, finds the broken "
                     "run and the task that caused it, 0.953 and 0.797"),
        ("An explanation layer", "Names the symptom family correctly 93.5% of "
                                 "the time, scored rather than displayed"),
        ("An honest evaluation", "Six baselines, four null results, one bug "
                                 "found in our own scoring, one claim corrected")]):
        y = 2.95 + i * 1.0
        text(s, 1.0, y, 3.3, 0.5, h, size=17, bold=True, font=HEAD, color=WHITE)
        text(s, 4.5, y + 0.05, 7.8, 0.85, b, size=13, color=LIGHT, line=1.2)
    text(s, 1.0, 6.75, 11.3, 0.5, "Demonstration follows", size=15,
         bold=True, color=WHITE)
    notes(s, "Close on the honesty point, then run the demo. About two minutes.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT} ({len(prs.slides._sldIdLst)} slides, black and white)")


if __name__ == "__main__":
    build()
