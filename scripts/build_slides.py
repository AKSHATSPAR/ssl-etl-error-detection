"""Build the viva presentation.

    PYTHONPATH=src python scripts/build_slides.py

python-pptx rather than a template, because every element's position is then
explicit and predictable. LibreOffice is not available in this environment, so
slides cannot be rendered to images for visual checking; the layout is therefore
kept deliberately conservative, with wide margins and short lines.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path("artifacts/slides/ETL_Project_Viva.pptx")
FIGS = Path("artifacts")

# Palette chosen for the subject: dark infrastructure slate, an amber that reads
# as "alert", and a teal that reads as "healthy". Amber is the accent because
# the whole project is about noticing when something has gone wrong.
INK    = RGBColor(0x1B, 0x24, 0x30)
SLATE  = RGBColor(0x2C, 0x3E, 0x50)
ALERT  = RGBColor(0xE8, 0x83, 0x3A)
HEAL   = RGBColor(0x3F, 0xB8, 0xA5)
PAPER  = RGBColor(0xF4, 0xF6, 0xF8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
MUTED  = RGBColor(0x6B, 0x7A, 0x8A)
CARD   = RGBColor(0xE9, 0xEE, 0xF2)

HEAD = "Cambria"
BODY = "Calibri"

W, H = 13.333, 7.5


def deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = Inches(W), Inches(H)
    return p


def blank(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = INK if dark else WHITE
    return s


def text(slide, x, y, w, h, runs, size=16, color=SLATE, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, space_after=6, line=None, anchor=MSO_ANCHOR.TOP):
    """runs: a string, or a list of (text, {overrides}) tuples, one per line."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    items = [runs] if isinstance(runs, str) else runs
    for i, item in enumerate(items):
        body, over = (item, {}) if isinstance(item, str) else item
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = over.get("align", align)
        para.space_after = Pt(over.get("space_after", space_after))
        if line or over.get("line"):
            para.line_spacing = over.get("line", line)
        r = para.add_run()
        r.text = body
        f = r.font
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.italic = over.get("italic", False)
        f.name = over.get("font", font)
        f.color.rgb = over.get("color", color)
    return tb


def box(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    if sh.shape_type == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sh.adjustments[0] = 0.12
        except Exception:
            pass
    return sh


def title(slide, t, sub=None, dark=False, y=0.55):
    text(slide, 0.85, y, 11.6, 1.0, t, size=34, bold=True, font=HEAD,
         color=WHITE if dark else INK)
    if sub:
        text(slide, 0.85, y + 0.85, 11.6, 0.55, sub, size=15,
             color=MUTED if not dark else RGBColor(0xA9, 0xB6, 0xC2))


def bullets(slide, x, y, w, items, size=15, gap=0.42, color=SLATE, dot=ALERT):
    for i, it in enumerate(items):
        cy = y + i * gap
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(cy + 0.075),
                                   Inches(0.11), Inches(0.11))
        d.fill.solid(); d.fill.fore_color.rgb = dot
        d.line.fill.background(); d.shadow.inherit = False
        text(slide, x + 0.28, cy, w - 0.28, gap, it, size=size, color=color)


def stat(slide, x, y, w, value, label, color=ALERT, vsize=52):
    # Height and offset are derived from the font size rather than hard-coded,
    # so changing vsize cannot silently push the numeral into its own label.
    num_h = vsize * 1.25 / 72.0
    text(slide, x, y, w, num_h, value, size=vsize, bold=True, font=HEAD,
         color=color)
    text(slide, x, y + num_h + 0.06, w, 0.7, label, size=13, color=MUTED)


def card(slide, x, y, w, h, heading, body_lines, accent=ALERT, hsize=15):
    box(slide, x, y, w, h, CARD)
    text(slide, x + 0.3, y + 0.26, w - 0.6, 0.4, heading, size=hsize, bold=True,
         font=HEAD, color=INK)
    text(slide, x + 0.3, y + 0.78, w - 0.6, h - 1.05, body_lines, size=13,
         color=SLATE, line=1.18)


def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


# --------------------------------------------------------------------------- #

def build() -> None:
    prs = deck()

    # 1 ---------------------------------------------------------------- title
    s = blank(prs, dark=True)
    text(s, 1.0, 2.15, 11.3, 1.6,
         "Catching Silent Failures in Data Pipelines",
         size=42, bold=True, font=HEAD, color=WHITE)
    text(s, 1.0, 3.75, 11.3, 0.9,
         "Self-supervised deep learning that finds a broken pipeline run, "
         "and points at the task that broke it",
         size=17, color=RGBColor(0xA9, 0xB6, 0xC2), line=1.25)
    box(s, 1.0, 4.95, 1.5, 0.045, ALERT, shape=MSO_SHAPE.RECTANGLE)
    text(s, 1.0, 5.35, 11.3, 0.9,
         [("B.Tech Deep Learning Project", {"size": 15, "color": WHITE}),
          ("Akshat Sparsh (23BDS0149)  \u00b7  Anuprakash Dubey (23BDS0162)",
           {"size": 13.5, "color": RGBColor(0xA9, 0xB6, 0xC2)}),
          ("October 2026", {"size": 12.5, "color": RGBColor(0x8A, 0x99, 0xA8)})])
    notes(s, "Open with the one-line version: data pipelines break quietly, and "
             "when they break every downstream step looks broken too. This project "
             "finds the one that actually caused it, without needing labelled examples.")

    # 2 -------------------------------------------------------------- problem
    s = blank(prs)
    title(s, "Data pipelines fail without telling you",
          "The job finishes, the report loads, and the numbers are wrong")
    card(s, 0.85, 2.3, 3.7, 3.5, "What people expect",
         ["A job crashes.", "", "Somebody gets an alert within minutes.",
          "", "It gets fixed the same day."], hsize=16)
    card(s, 4.83, 2.3, 3.7, 3.5, "What actually happens",
         ["The job finishes successfully.", "",
          "A price field switched from rupees to paise upstream.",
          "", "Every number is 100x too big."], hsize=16)
    box(s, 8.81, 2.3, 3.67, 3.5, INK)
    text(s, 9.11, 2.56, 3.07, 0.4, "The cost", size=16, bold=True,
         font=HEAD, color=WHITE)
    text(s, 9.11, 3.08, 3.07, 2.5,
         ["Nobody notices for weeks.", "",
          "Every report built on that table is wrong for the whole period.",
          "", "It is found by a human squinting at a dashboard."],
         size=13, color=RGBColor(0xC9, 0xD3, 0xDC), line=1.18)
    notes(s, "Give a concrete example. A currency convention changes upstream and "
             "amounts silently become 100x too large. Nothing crashes, so nothing "
             "alerts. This is the class of failure the project targets.")

    # 3 ----------------------------------------------------------- why hard
    s = blank(prs)
    title(s, "Two reasons this is hard")
    box(s, 0.85, 2.15, 5.5, 4.0, CARD)
    text(s, 1.2, 2.5, 1.0, 0.8, "1", size=44, bold=True, font=HEAD, color=ALERT)
    text(s, 1.2, 3.4, 4.8, 0.5, "There are no labels", size=20, bold=True,
         font=HEAD, color=INK)
    text(s, 1.2, 4.0, 4.8, 2.0,
         "Failures are rare and recorded badly. A team that has just spent a "
         "night fixing one is not going to sit down and annotate it.\n\n"
         "So the model has to learn what normal looks like on its own.",
         size=14, color=SLATE, line=1.22)
    box(s, 6.98, 2.15, 5.5, 4.0, CARD)
    text(s, 7.33, 2.5, 1.0, 0.8, "2", size=44, bold=True, font=HEAD, color=ALERT)
    text(s, 7.33, 3.4, 4.8, 0.5, "Symptoms are not causes", size=20, bold=True,
         font=HEAD, color=INK)
    text(s, 7.33, 4.0, 4.8, 2.0,
         "When one step breaks, every step after it looks broken too.\n\n"
         "Often the damage grows as it travels, so the biggest warning sign is "
         "furthest from the actual problem.",
         size=14, color=SLATE, line=1.22)
    notes(s, "Reason two is the interesting one and the whole reason the project "
             "exists. Telling someone that nine of nine tasks look unhealthy is "
             "not help. Land this before moving on.")

    # 4 ------------------------------------------------------- propagation fig
    s = blank(prs)
    title(s, "One fault, nine suspects",
          "The problem starts at one task, but almost everything lights up")
    pic(s, FIGS / "dataset_report/fig2_propagation.png", 0.85, 2.45, w=8.4)
    box(s, 9.6, 2.45, 2.88, 3.1, INK)
    text(s, 9.9, 2.72, 2.28, 2.6,
         ["The fault was injected in one task only.",
          "",
          "8 of 9 tasks move.",
          "",
          "Several move harder than the one that caused it."],
         size=13, color=RGBColor(0xC9, 0xD3, 0xDC), line=1.2)
    text(s, 0.85, 6.35, 11.6, 0.5,
         "This is why you cannot simply blame whichever task looks worst.",
         size=15, bold=True, color=INK)
    notes(s, "Point at the bright band. Every one of those tasks would trip an "
             "ordinary alarm. Only one of them is the cause. Ranking by 'how bad "
             "does it look' picks the wrong one.")

    # 5 ------------------------------------------------------------- dataset
    s = blank(prs)
    title(s, "There was no dataset, so I built one",
          "No public benchmark of pipeline failures with cause labels exists")
    bullets(s, 0.85, 2.4, 6.0, [
        "Existing work is tested on server and sensor data.",
        "Those datasets have no pipeline, so there is no 'which step' to ask about.",
        "So the question cannot even be posed on them, let alone measured.",
    ], size=15, gap=0.62)
    box(s, 0.85, 4.5, 6.0, 1.6, CARD)
    text(s, 1.15, 4.78, 5.4, 1.1,
         "Building the benchmark is part of the contribution, not a shortcut "
         "around a missing one.", size=15, bold=True, color=INK, line=1.2)
    stat(s, 7.4, 2.4, 2.4, "8,000", "pipeline runs simulated")
    stat(s, 10.0, 2.4, 2.4, "9", "tasks per run", color=HEAL)
    stat(s, 7.4, 4.15, 2.4, "15", "types of failure")
    stat(s, 10.0, 4.15, 2.4, "273", "incidents labelled", color=HEAL)
    notes(s, "Be ready for 'why not real data'. Answer: no public dataset labels "
             "which step caused a failure, and company pipeline data is not "
             "released. Building one is the only way to measure localization.")

    # 6 ---------------------------------------------------------------- DAG
    s = blank(prs)
    title(s, "The pipeline I simulated",
          "Nine tasks, each doing real work on real rows")
    nodes = [
        ("extract\ncustomers", 0.85, 2.55), ("extract\norders", 0.85, 3.85),
        ("extract\npayments", 0.85, 5.15),
        ("clean\ncustomers", 3.15, 2.55), ("clean\norders", 3.15, 3.85),
        ("join orders +\ncustomers", 5.45, 3.2),
        ("join\npayments", 7.75, 3.85),
        ("aggregate\ndaily", 10.05, 3.85), ("load\nwarehouse", 12.0, 3.85),
    ]
    for i, (name, x, y) in enumerate(nodes):
        w = 1.35 if i < 8 else 1.2
        sh = box(s, x, y, w, 0.95, SLATE if i else ALERT)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = name
        r.font.size = Pt(10.5); r.font.color.rgb = WHITE; r.font.name = BODY
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    text(s, 0.85, 6.45, 11.6, 0.8,
         "Faults are injected into the actual rows, not into the statistics. "
         "The next task then reads those broken rows, so the damage spreads on "
         "its own instead of being scripted by me.",
         size=14, color=SLATE, line=1.2)
    notes(s, "The point of the orange box: a fault injected at extract_customers "
             "flows downstream because later tasks genuinely read the corrupted "
             "rows. I did not write rules for how damage spreads.")

    # 7 -------------------------------------------------------------- faults
    s = blank(prs)
    title(s, "Fifteen ways a pipeline breaks",
          "Grouped into six families, all taken from real failure modes")
    fam = [
        ("Schema", "a column vanishes, a column appears,\nnumbers arrive as text", ALERT),
        ("Completeness", "nulls spike, keys go missing,\nrows duplicated or truncated", ALERT),
        ("Values", "units change, values corrupt,\ncategories get mangled", ALERT),
        ("Time", "timezone shifts, data arrives late", HEAL),
        ("Integrity", "keys stop matching,\njoins explode", HEAL),
        ("Operational", "the task runs slow but the\ndata is perfectly fine", HEAL),
    ]
    for i, (h, b, c) in enumerate(fam):
        x = 0.85 + (i % 3) * 3.95
        y = 2.4 + (i // 3) * 2.05
        box(s, x, y, 3.6, 1.75, CARD)
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.3), Inches(y + 0.3),
                               Inches(0.3), Inches(0.3))
        d.fill.solid(); d.fill.fore_color.rgb = c
        d.line.fill.background(); d.shadow.inherit = False
        text(s, x + 0.72, y + 0.28, 2.6, 0.35, h, size=15, bold=True,
             font=HEAD, color=INK)
        text(s, x + 0.3, y + 0.78, 3.0, 0.85, b, size=12, color=SLATE, line=1.15)
    text(s, 0.85, 6.7, 11.6, 0.5,
         "The last one is deliberately awkward: the task misses its deadline but "
         "its output is correct, so it must not be treated as data corruption.",
         size=13, color=MUTED)
    notes(s, "If asked why 15: they cover the taxonomy that Google's and Amazon's "
             "data validation tools describe in their papers. Operational is the "
             "trap case, included on purpose.")

    # 8 --------------------------------------------------- measured propagation
    s = blank(prs)
    title(s, "I measured the damage instead of guessing it",
          "The obvious labelling turns out to be wrong")
    box(s, 0.85, 2.35, 5.6, 2.0, CARD)
    text(s, 1.15, 2.62, 5.0, 1.5,
         [("The easy assumption", {"size": 15, "bold": True, "font": HEAD,
                                   "color": INK}),
          ("If a task breaks, call every task after it broken too.",
           {"size": 13, "color": SLATE})])
    box(s, 0.85, 4.6, 5.6, 2.0, INK)
    text(s, 1.15, 4.87, 5.0, 1.6,
         [("Why that is wrong", {"size": 15, "bold": True, "font": HEAD,
                                 "color": WHITE}),
          ("An extra unused column changes nothing downstream. A slow task still "
           "produces correct data.",
           {"size": 13, "color": RGBColor(0xC9, 0xD3, 0xDC)})])
    text(s, 7.0, 2.35, 5.5, 1.4,
         "So I run every broken batch a second time with the fault removed, "
         "everything else kept identical, and compare.",
         size=15, color=SLATE, line=1.22)
    stat(s, 7.0, 3.95, 2.5, "2.48", "tasks actually affected", color=HEAL)
    stat(s, 9.9, 3.95, 2.5, "3.94", "tasks the assumption claims")
    text(s, 7.0, 5.6, 5.5, 1.0,
         "The labels are a measurement, not a guess. That matters, because the "
         "model is graded against them.",
         size=13, color=MUTED, line=1.2)
    notes(s, "This is a methods point examiners like. I did not assume how far "
             "damage spreads, I measured it by re-running each broken batch "
             "clean with the same random seeds.")

    # 9 ------------------------------------------------------- how it learns
    s = blank(prs)
    title(s, "How the model learns with no labels",
          "Hide part of the picture, then make it guess what was hidden")
    steps = [
        ("1", "Watch normal runs", "Look at thousands of ordinary pipeline runs. "
                                    "No one says which are good or bad."),
        ("2", "Hide something", "Cover up part of a run: one task's numbers, or "
                                 "one kind of measurement."),
        ("3", "Guess it back", "Predict what was hidden using the other tasks and "
                                "the recent history."),
        ("4", "Watch for surprise", "Once it is good at guessing, whatever it "
                                     "cannot guess is something unusual."),
    ]
    for i, (n, h, b) in enumerate(steps):
        x = 0.85 + i * 3.03
        box(s, x, 2.45, 2.75, 3.3, CARD)
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.28), Inches(2.75),
                               Inches(0.6), Inches(0.6))
        c.fill.solid(); c.fill.fore_color.rgb = ALERT
        c.line.fill.background(); c.shadow.inherit = False
        tf = c.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = n
        r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = WHITE
        r.font.name = HEAD
        text(s, x + 0.28, 3.55, 2.2, 0.5, h, size=15, bold=True, font=HEAD,
             color=INK)
        text(s, x + 0.28, 4.15, 2.2, 1.4, b, size=12, color=SLATE, line=1.18)
    text(s, 0.85, 6.15, 11.6, 0.6,
         "This is the same idea as covering words in a sentence and predicting "
         "them, which is how language models are trained.",
         size=14, color=MUTED)
    notes(s, "The sentence analogy is the one that lands with non-specialists. "
             "Hide a word, predict it, and to do that well you have to learn how "
             "language works. Same here with pipeline behaviour.")

    # 10 ------------------------------------------------------ localization
    s = blank(prs)
    title(s, "How I find the culprit, not just the victims",
          "Ask each task: are you surprising, given what you were fed?")
    box(s, 0.85, 2.4, 5.6, 2.05, CARD)
    text(s, 1.15, 2.68, 5.0, 0.4, "A victim", size=17, bold=True, font=HEAD,
         color=INK)
    text(s, 1.15, 3.2, 5.0, 1.1,
         "Its output looks wrong, but it follows perfectly from the bad input it "
         "was given. Nothing surprising here.", size=14, color=SLATE, line=1.2)
    box(s, 0.85, 4.65, 5.6, 2.05, INK)
    text(s, 1.15, 4.93, 5.0, 0.4, "The culprit", size=17, bold=True, font=HEAD,
         color=ALERT)
    text(s, 1.15, 5.45, 5.0, 1.1,
         "Its input was perfectly healthy and its output is wrong anyway. That is "
         "where the trouble started.", size=14,
         color=RGBColor(0xC9, 0xD3, 0xDC), line=1.2)
    box(s, 7.0, 2.4, 5.5, 4.3, CARD)
    text(s, 7.3, 2.7, 4.9, 0.45, "One catch I got wrong at first",
         size=16, bold=True, font=HEAD, color=INK)
    text(s, 7.3, 3.3, 4.9, 3.1,
         "My first version hid only the task being tested. But the model could "
         "still see that task's own downstream steps, which were also broken, "
         "and quietly read the answer off them.\n\n"
         "Hiding the task and everything downstream of it fixed this.\n\n"
         "Accuracy went from 63% to 77%.",
         size=13.5, color=SLATE, line=1.22)
    notes(s, "Be honest that the first version leaked. It shows you tested your "
             "own idea rather than trusting it. The jump from 63 to 77 percent is "
             "the evidence the fix was real.")

    # 11 ------------------------------------------------------------ results
    s = blank(prs)
    title(s, "Results", "Tested on the last quarter of the timeline, never seen "
                        "during training")
    pic(s, FIGS / "results/fig5_method_comparison.png", 0.85, 2.35, w=8.3)
    box(s, 9.5, 2.35, 2.98, 3.4, INK)
    text(s, 9.8, 2.62, 2.38, 3.0,
         ["My model does not win.", "",
          "A simpler method detects slightly better, another localizes "
          "slightly better.", "",
          "All four leaders are within measurement error of each other."],
         size=13, color=RGBColor(0xC9, 0xD3, 0xDC), line=1.2)
    text(s, 0.85, 6.3, 11.6, 0.7,
         "Saying so is the honest result. The interesting finding turned out to "
         "be about how you score the tasks, not which model you use.",
         size=15, bold=True, color=INK, line=1.2)
    notes(s, "Do not hide this. If an examiner asks 'so did your model win', the "
             "answer is no, and the reason is that this data is small and simple "
             "enough that classic methods do well. What I found instead is on the "
             "next slide.")

    # 12 ------------------------------------------------------- main finding
    s = blank(prs)
    title(s, "The main finding",
          "Two completely different fixes turned out to be worth exactly the same")
    rows = [
        ("Rank tasks by how bad they look", "58.8%", MUTED),
        ("Compare each task to its own normal", "77.3%", HEAL),
        ("Ask if it is surprising given its input", "77.3%", HEAL),
        ("Do both", "79.7%", ALERT),
    ]
    for i, (label, val, c) in enumerate(rows):
        y = 2.55 + i * 0.92
        box(s, 0.85, y, 8.1, 0.75, CARD if i else RGBColor(0xDD, 0xE3, 0xE8))
        text(s, 1.2, y + 0.2, 6.4, 0.45, label, size=15, color=INK,
             bold=(i == 3))
        text(s, 7.5, y + 0.13, 1.2, 0.5, val, size=20, bold=True, font=HEAD,
             color=c, align=PP_ALIGN.RIGHT)
    box(s, 9.3, 2.55, 3.18, 3.66, INK)
    text(s, 9.6, 2.85, 2.58, 3.2,
         ["Rows 2 and 3 are identical.", "",
          "One is a statistics trick. One is a structural idea about cause and "
          "effect.", "",
          "They are worth the same, and doing both barely helps."],
         size=13, color=RGBColor(0xC9, 0xD3, 0xDC), line=1.2)
    text(s, 0.85, 6.5, 11.6, 0.6,
         "An earlier version of my write-up claimed the structural idea was worth "
         "far more. It was being compared against an unfair baseline.",
         size=13, color=MUTED, line=1.2)
    notes(s, "This is the single best slide to be asked about. Different tasks are "
             "naturally harder to predict than others, so raw scores are not "
             "comparable. Both fixes correct for that in different ways, and it "
             "turns out that is where nearly all the gain comes from.")

    # 13 ---------------------------------------------------------- ablations
    s = blank(prs)
    title(s, "What did not work", "I removed each design idea and measured the damage")
    items = [
        ("Telling the model the pipeline shape", "no change",
         "It works out the shape from the data by itself."),
        ("Giving it a clock", "no change",
         "A 32-hour window already contains a full day."),
        ("Adding contrastive learning", "no change",
         "Cost four times as much and helped nothing."),
        ("Training on perfectly clean data", "no change",
         "It tolerates messy history, which was the claim."),
    ]
    for i, (h, v, b) in enumerate(items):
        y = 2.35 + i * 1.08
        box(s, 0.85, y, 11.6, 0.92, CARD)
        text(s, 1.2, y + 0.14, 5.3, 0.4, h, size=15, bold=True, color=INK)
        text(s, 1.2, y + 0.52, 5.3, 0.35, b, size=12, color=MUTED)
        text(s, 6.9, y + 0.26, 2.1, 0.4, v, size=14, bold=True, color=HEAL)
    text(s, 0.85, 6.85, 11.6, 0.5,
         "Four ideas, four null results. Reporting them is part of the work.",
         size=14, bold=True, color=INK)
    notes(s, "Examiners respect null results that are reported. Each one has a "
             "reason, not a shrug. The DAG hint is redundant because the model "
             "recovers the structure on its own, which I checked directly.")

    # 14 ----------------------------------------------------------- surprise
    s = blank(prs)
    title(s, "The one thing that did change something",
          "And it was the opposite of what I expected")
    box(s, 0.85, 2.4, 5.5, 1.5, CARD)
    text(s, 1.15, 2.7, 4.9, 0.9,
         "I trained the model for longer. It got better at its training task.",
         size=15, color=INK, line=1.2)
    box(s, 0.85, 4.1, 5.5, 1.5, INK)
    text(s, 1.15, 4.4, 4.9, 0.9,
         "And it got worse at finding the culprit. 80% down to 76%.",
         size=15, color=ALERT, line=1.2, bold=True)
    box(s, 6.9, 2.4, 5.58, 3.2, CARD)
    text(s, 7.2, 2.7, 4.98, 0.45, "Why", size=17, bold=True, font=HEAD, color=INK)
    text(s, 7.2, 3.3, 4.98, 2.1,
         "The model relies on being surprised by broken data. Train it long "
         "enough and it gets good at predicting the broken data too, so the "
         "surprise disappears.",
         size=14, color=SLATE, line=1.22)
    text(s, 0.85, 5.95, 11.6, 1.0,
         "This is awkward for me: I chose when to stop training using a measure "
         "that turns out not to track what I actually care about. It is written "
         "up as an open problem rather than hidden.",
         size=14, color=MUTED, line=1.2)
    notes(s, "If they ask 'what would you do differently', this is the answer. "
             "I need a stopping rule that tracks localization and still needs no "
             "labels. I do not have one yet.")

    # 15 ------------------------------------------------------ explainability
    s = blank(prs)
    title(s, "Explaining the answer",
          "Not just which task, but what kind of problem")
    pic(s, FIGS / "results/fig7_attention.png", 0.85, 2.3, h=4.3)
    box(s, 6.6, 2.3, 5.88, 2.0, CARD)
    text(s, 6.9, 2.58, 5.28, 1.5,
         [("It learned the pipeline on its own",
           {"size": 16, "bold": True, "font": HEAD, "color": INK}),
          ("The bright squares are the real connections between tasks. Nobody "
           "told it which tasks feed which.",
           {"size": 13, "color": SLATE})])
    box(s, 6.6, 4.6, 5.88, 2.0, INK)
    text(s, 6.9, 4.88, 5.28, 1.5,
         [("It names the right symptom 93% of the time",
           {"size": 16, "bold": True, "font": HEAD, "color": WHITE}),
          ("A unit change is reported as a value problem, a timezone shift as a "
           "timing problem. This is scored, not just displayed.",
           {"size": 13, "color": RGBColor(0xC9, 0xD3, 0xDC)})])
    notes(s, "Most papers show an explanation picture and never check it. Because "
             "my simulator knows what each fault should look like, I can score "
             "whether the explanation is correct. 93 percent.")

    # 16 -------------------------------------------------------- limitations
    s = blank(prs)
    title(s, "What this cannot do yet", "The honest list")
    lims = [
        ("The data is simulated",
         "I chose how often faults happen and how severe they are. The ranking of "
         "methods is more trustworthy than any single number."),
        ("Two failure types localize badly",
         "Both involve row counts changing. I know exactly why, and the fix is a "
         "missing measurement rather than a better model."),
        ("Detection is too easy here",
         "Five of seven methods land within a hair of each other, so the benchmark "
         "does not separate them well on that half of the problem."),
        ("Only one pipeline shape",
         "Everything is measured on one nine-task pipeline."),
    ]
    for i, (h, b) in enumerate(lims):
        x = 0.85 + (i % 2) * 5.98
        y = 2.35 + (i // 2) * 2.2
        box(s, x, y, 5.6, 1.95, CARD)
        text(s, x + 0.3, y + 0.26, 5.0, 0.4, h, size=15, bold=True, font=HEAD,
             color=INK)
        text(s, x + 0.3, y + 0.78, 5.0, 1.0, b, size=13, color=SLATE, line=1.18)
    notes(s, "Volunteering limitations before being asked is the strongest move "
             "available. Each one here has a specific reason and, where relevant, "
             "a named fix.")

    # 17 ------------------------------------------------------------- next
    s = blank(prs)
    title(s, "What I would do next")
    nxt = [
        ("Measure how well join keys match",
         "This is the exact missing piece behind the two failure types that "
         "localize badly. My error analysis predicts it should fix them."),
        ("Find a better stopping rule",
         "One that needs no labels but still tracks the thing I care about. This "
         "is the most interesting open problem the project turned up."),
        ("Test against a real pipeline",
         "Even a few weeks of real telemetry would let me check the simulator's "
         "assumptions instead of choosing them."),
    ]
    for i, (h, b) in enumerate(nxt):
        y = 2.4 + i * 1.5
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.85), Inches(y + 0.2),
                               Inches(0.55), Inches(0.55))
        c.fill.solid(); c.fill.fore_color.rgb = ALERT
        c.line.fill.background(); c.shadow.inherit = False
        tf = c.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = str(i + 1)
        r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = WHITE
        r.font.name = HEAD
        text(s, 1.7, y + 0.18, 10.6, 0.4, h, size=18, bold=True, font=HEAD,
             color=INK)
        text(s, 1.7, y + 0.68, 10.6, 0.7, b, size=14, color=SLATE, line=1.2)
    notes(s, "Item one is the strongest, because it is a prediction made by my "
             "own error analysis. If they ask what you would test first, say that.")

    # 18 ------------------------------------------------------------- close
    s = blank(prs, dark=True)
    text(s, 1.0, 1.75, 11.3, 0.8, "What I built", size=32, bold=True, font=HEAD,
         color=WHITE)
    closing = [
        ("A benchmark", "8,000 pipeline runs, 15 failure types, cause labels "
                        "measured rather than assumed"),
        ("A model", "learns normal behaviour with no labels, finds the run that "
                    "broke and the task that broke it"),
        ("An honest evaluation", "including four null results, one bug I found in "
                                 "my own work, and a claim I had to correct"),
    ]
    for i, (h, b) in enumerate(closing):
        y = 2.9 + i * 1.15
        text(s, 1.0, y, 3.2, 0.5, h, size=19, bold=True, font=HEAD, color=ALERT)
        text(s, 4.4, y + 0.04, 7.9, 0.9, b, size=14,
             color=RGBColor(0xC9, 0xD3, 0xDC), line=1.2)
    text(s, 1.0, 6.5, 11.3, 0.5, "Live demo next", size=16, bold=True, color=WHITE)
    notes(s, "Close on the honesty point, then run the demo. It takes about "
             "thirty seconds and shows a real broken run being diagnosed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
