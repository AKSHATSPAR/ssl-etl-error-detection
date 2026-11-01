"""Fill the VIT BCSE497J report template from content.py.

    PYTHONPATH=src python scripts/report_fill/build.py

Edits the supplied template in place rather than generating a fresh document,
so the institute's title page, logo, declaration and certificate keep their own
formatting. Paragraphs are located by their placeholder text rather than by
index, because any edit shifts the indices underneath.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content as C

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "artifacts/report_vit/_template.docx"
OUT = ROOT / "artifacts/report_vit/Project_Report_Akshat_Anuprakash.docx"
FONT = "Times New Roman"


# ----------------------------------------------------------------- helpers --
def style_run(run, size=12, bold=False, italic=False, caps=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if caps:
        run.font.all_caps = True
    return run


def set_text(par, text, **kw):
    """Replace a paragraph's text, keeping its paragraph-level formatting."""
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    style_run(par.add_run(text), **kw)
    return par


def drop(par):
    par._element.getparent().remove(par._element)


def find(doc, needle, start=0):
    """Locate a paragraph by its text, ignoring how it is spaced.

    The template pads several paragraphs with runs of consecutive spaces, so a
    plain substring search misses them and the placeholder survives into the
    finished report. Collapsing whitespace on both sides avoids that.
    """
    want = " ".join(needle.lower().split())
    for i, p in enumerate(doc.paragraphs):
        if i >= start and want in " ".join(p.text.lower().split()):
            return i
    return -1


def add_par(doc, text="", size=12, bold=False, italic=False, align=None,
            space_before=0, space_after=6, line=1.15, indent=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    if indent is not None:
        pf.left_indent = Inches(indent)
    if text:
        style_run(p.add_run(text), size=size, bold=bold, italic=italic)
    return p


def add_table(doc, caption, rows, widths=None):
    """Caption above the table, per the template's own instruction."""
    add_par(doc, caption, size=10, bold=True, space_before=10, space_after=4)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            style_run(p.add_run(str(val)), size=10, bold=(ri == 0))
            if widths:
                cell.width = Inches(widths[ci])
    add_par(doc, "", space_after=6)
    return t


def add_figure(doc, path, caption, width=5.9):
    f = ROOT / path
    if not f.exists():
        add_par(doc, f"[figure missing: {path}]", size=10, italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(f), width=Inches(width))
    add_par(doc, caption, size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
            space_after=10)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------ front matter --
def fill_front(doc):
    # The cover names the course, not the project-work code: this is submitted
    # for the Deep Learning course rather than as Project-I.
    i = find(doc, "BCSE497J")
    if i >= 0:
        set_text(doc.paragraphs[i], C.COURSE, size=14, bold=True)
        doc.paragraphs[i].alignment = WD_ALIGN_PARAGRAPH.CENTER
    i = find(doc, "B.Tech.")
    if i >= 0 and "B.Tech." == doc.paragraphs[i].text.strip():
        set_text(doc.paragraphs[i], "B.Tech.", size=14, bold=True)
        doc.paragraphs[i].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Title on the cover.
    i = find(doc, "<Title of the Project>")
    set_text(doc.paragraphs[i], C.TITLE.upper(), size=18, bold=True)
    doc.paragraphs[i].alignment = WD_ALIGN_PARAGRAPH.CENTER
    drop(doc.paragraphs[find(doc, "Times New Roman 18, Bold, Upper Case")])

    j = find(doc, "Programme")
    if j >= 0:
        set_text(doc.paragraphs[j], C.PROGRAMME_SHORT, size=14, bold=True)
        doc.paragraphs[j].alignment = WD_ALIGN_PARAGRAPH.CENTER

    k = find(doc, "mention names of team members")
    if k >= 0:
        drop(doc.paragraphs[k])

    # Student table: two authors, so the third template row goes.
    t = doc.tables[0]
    for ri, (reg, name) in enumerate(C.STUDENTS):
        for ci, val in enumerate((reg, name)):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_run(p.add_run(val), size=13, bold=True)
    if len(t.rows) > len(C.STUDENTS):
        for row in list(t.rows)[len(C.STUDENTS):]:
            row._element.getparent().remove(row._element)

    # Supervisor table.
    g = doc.tables[1]
    for ri, val in enumerate([C.GUIDE, C.GUIDE_DESIG,
                              "School of Computer Science and Engineering (SCOPE)"]):
        cell = g.cell(ri, 0)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(p.add_run(val), size=13, bold=(ri == 0))

    d = find(doc, "October 2026")
    if d >= 0:
        set_text(doc.paragraphs[d], C.MONTH, size=13, bold=True)
        doc.paragraphs[d].alignment = WD_ALIGN_PARAGRAPH.CENTER

    names_and = " and ".join(f"{n} ({r})" for r, n in C.STUDENTS)
    names_plain = " and ".join(n.title() for _, n in C.STUDENTS)

    # Declaration.
    i = find(doc, "hereby declare that the project entitled")
    set_text(doc.paragraphs[i],
             f"We hereby declare that the project entitled {C.TITLE} submitted "
             f"by us, for the award of the degree of Bachelor of Technology in "
             f"{C.PROGRAMME} to VIT is a record of bonafide work carried out by "
             f"us under the supervision of {C.GUIDE}.", size=12)
    i = find(doc, "I further declare that the work reported")
    set_text(doc.paragraphs[i],
             "We further declare that the work reported in this project has not "
             "been submitted and will not be submitted, either in part or in "
             "full, for the award of any other degree or diploma in this "
             "institute or any other institute or university.", size=12)
    i = find(doc, "Signature of the Candidate")
    set_text(doc.paragraphs[i], "Signature of the Candidates", size=12, bold=True)
    i = find(doc, "Note: Include your project title and guide name")
    if i >= 0:
        drop(doc.paragraphs[i])

    # Certificate.
    i = find(doc, "This is to certify that the project entitled")
    set_text(doc.paragraphs[i],
             f"This is to certify that the project entitled {C.TITLE} submitted "
             f"by {names_and}, School of Computer Science and Engineering, VIT, "
             f"for the award of the degree of Bachelor of Technology in "
             f"{C.PROGRAMME}, is a record of bonafide work carried out by them "
             f"under my supervision during Fall Semester 2026-2027, as per the "
             f"VIT code of academic and research ethics.", size=12)
    i = find(doc, "Note: Include your project title and student names")
    if i >= 0:
        drop(doc.paragraphs[i])
    i = find(doc, "<< Head of the Department >>")
    if i >= 0:
        set_text(doc.paragraphs[i], "Head of the Department", size=12, bold=True)
    i = find(doc, "<< Programme >>")
    if i >= 0:
        set_text(doc.paragraphs[i], C.PROGRAMME, size=12, bold=True)

    return names_plain


def remove_acknowledgements(doc):
    """The student asked for this page to be removed."""
    start = find(doc, "ACKNOWLEDGEMENTS")
    end = find(doc, "TABLE OF CONTENTS")
    if start < 0 or end < 0:
        return 0
    removed = 0
    for p in list(doc.paragraphs)[start:end]:
        drop(p)
        removed += 1
    return removed


def remove_certificate(doc):
    """The student asked for this page to be removed."""
    start = find(doc, "CERTIFICATE")
    end = find(doc, "TABLE OF CONTENTS")
    if start < 0 or end < 0:
        return 0
    removed = 0
    for p in list(doc.paragraphs)[start:end]:
        drop(p)
        removed += 1
    return removed


def fill_toc(doc):
    t = doc.tables[2]
    while len(t.rows) > 1:
        t.rows[-1]._element.getparent().remove(t.rows[-1]._element)
    for sl, title, page in C.TOC:
        row = t.add_row()
        chapter = bool(sl)
        for ci, val in enumerate((sl, title, page)):
            cell = row.cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            if ci != 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_run(p.add_run(val), size=11, bold=chapter)


def fill_figures_tables(doc):
    t = doc.tables[3]
    while len(t.rows) > 1:
        t.rows[-1]._element.getparent().remove(t.rows[-1]._element)
    for no, cap, page in C.FIGURES:
        row = t.add_row()
        for ci, val in enumerate((no, cap, page)):
            cell = row.cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            if ci != 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_run(p.add_run(val), size=10)

    # A note the template leaves for whoever fills it in, not report content.
    for note in ("caption should come below the figure",
                 "Contents, Times New Roman 12"):
        i = find(doc, note)
        if i >= 0:
            drop(doc.paragraphs[i])

    # The template's list of tables is loose paragraphs, not a table.
    i = find(doc, "Table No.\tTitle")
    if i >= 0:
        set_text(doc.paragraphs[i], "Table No.\tTitle\tPage No.", size=11, bold=True)
    i = find(doc, "2.1\tTable caption")
    if i >= 0:
        par = doc.paragraphs[i]
        set_text(par, "\n".join(f"{no}\t{cap}\t{page}"
                                for no, cap, page in C.TABLES), size=10)


def two_column_list(doc, par, rows, left_width, right_width, size=11,
                    spacer=False):
    """Lay a two-column list out as a real table, replacing *par*.

    A tab stop cannot wrap. Any description wider than its column runs back to
    the left margin and collides with the next entry, which is what the symbol
    list did. Table cells wrap inside their own column.
    """
    tbl = doc.add_table(rows=0, cols=2)
    tbl.autofit = False
    for left, right in rows:
        cells = tbl.add_row().cells
        for cell, val, width in ((cells[0], left, left_width),
                                 (cells[1], right, right_width)):
            cell.width = width
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            style_run(p.add_run(val), size=size)
    par._element.addnext(tbl._element)
    if spacer:
        # The abbreviations list inherits a blank line from the template and
        # this one does not, so give it the same breathing room.
        set_text(par, "", size=size)
    else:
        drop(par)
    return tbl


def fill_abbreviations(doc):
    first = find(doc, "2G\tSecond Generation")
    if first < 0:
        return
    two_column_list(doc, doc.paragraphs[first], C.ABBREVIATIONS,
                    Inches(1.3), Inches(4.7))
    for stale in ["3GPP\t", "3G\tThird", "4G\tFourth", "AWGN\t", "BBE\t",
                  "CSMA\t", "DAS\t"]:
        i = find(doc, stale)
        if i >= 0:
            drop(doc.paragraphs[i])
    i = find(doc, "All the abbreviations should follow")
    if i >= 0:
        drop(doc.paragraphs[i])
    for stale in ["f\tCFO", "\tNCFO"]:
        i = find(doc, stale)
        if i >= 0:
            if stale == "f\tCFO":
                two_column_list(doc, doc.paragraphs[i], C.SYMBOLS,
                                Inches(1.0), Inches(5.0), spacer=True)
            else:
                drop(doc.paragraphs[i])


def _page_field(par):
    """Drop a PAGE field into *par*, centred."""
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run()
    style_run(run, size=12)
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, end):
        run._element.append(el)


def _numbering(section, fmt, start):
    """Set the printed page-number format for one section."""
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(qn("w:fmt"), fmt)
    el.set(qn("w:start"), str(start))
    sectPr.append(el)


def _split_section_before(doc, par):
    """Start a new section at *par*, carrying the current settings forward."""
    last = doc.sections[-1]
    carried = copy.deepcopy(last._sectPr)
    holder = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pPr.append(last._sectPr)
    holder.append(pPr)
    par._element.addprevious(holder)
    doc._body._element.append(carried)


def add_page_numbers(doc):
    """Number the pages, roman through the front matter and arabic in the body.

    The template prints no page numbers at all, which leaves the contents page
    pointing at numbers that appear nowhere. The declaration and the contents
    page itself stay unnumbered, as they usually are, so the roman count opens
    on the first page the contents actually lists.
    """
    for needle in ("List of Figures", "CHAPTER 1"):
        i = find(doc, needle)
        if i < 0:
            return 0
        _split_section_before(doc, doc.paragraphs[i])

    secs = doc.sections
    _numbering(secs[-2], "lowerRoman", 1)
    _numbering(secs[-1], "decimal", 1)
    for s in secs[:-2]:
        s.footer.is_linked_to_previous = False
        for p in list(s.footer.paragraphs)[1:]:
            drop(p)
        set_text(s.footer.paragraphs[0], "", size=12)
    for s in secs[-2:]:
        s.footer.is_linked_to_previous = False
        for p in list(s.footer.paragraphs)[1:]:
            drop(p)
        par = s.footer.paragraphs[0]
        set_text(par, "", size=12)
        _page_field(par)
    return len(secs)


def fill_date(doc):
    """The declaration carries a date line the template leaves blank."""
    for p in doc.paragraphs:
        flat = " ".join(p.text.split())
        if flat.startswith("Place") and "Date" in flat:
            for run in reversed(p.runs):
                if run.text.strip() == ":":
                    run.text = f"\t: {C.DATE}"
                    # The template indents this line nearly five inches from
                    # the right, which leaves about an inch to write in. The
                    # date wraps mid-way without this.
                    p.paragraph_format.right_indent = Inches(1.5)
                    # Place and date belong on separate lines. The template
                    # only ever separated them by overflowing the indent
                    # above, so put a real break between them.
                    for idx, r in enumerate(p.runs):
                        if r.text.strip() == "Date" and idx:
                            before = p.runs[idx - 1]
                            before.text = before.text.rstrip()
                            before._element.append(OxmlElement("w:br"))
                            break
                    return 1
    return 0


def add_signatures(doc):
    """Put the candidates' signatures above the line that names them."""
    i = find(doc, "Signature of the Candidates")
    if i < 0:
        return 0
    caption = doc.paragraphs[i]
    present = [(reg, name) for reg, name in C.STUDENTS
               if (ROOT / C.SIGNATURES.get(reg, "")).is_file()]
    if not present:
        return 0

    tbl = doc.add_table(rows=0, cols=len(present))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    rows = [tbl.add_row() for _ in range(3)]
    for ci, (reg, name) in enumerate(present):
        for row in rows:
            row.cells[ci].width = Inches(2.3)
            row.cells[ci].text = ""

        p = rows[0].cells[ci].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(ROOT / C.SIGNATURES[reg]), height=Inches(0.62))

        p = rows[1].cells[ci].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(p.add_run(name), size=11, bold=True)

        p = rows[2].cells[ci].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(p.add_run(reg), size=11)

    spacer = caption.insert_paragraph_before()
    style_run(spacer.add_run(""), size=12)
    caption._element.addprevious(tbl._element)
    # Centre the caption under the pair rather than leaving it off to one side.
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.left_indent = Inches(0)
    caption.paragraph_format.right_indent = Inches(0)
    caption.paragraph_format.space_before = Pt(10)
    return len(present)


def tidy_before_abstract(doc):
    """Clear the tail the template's own symbol list leaves behind.

    Replacing that list with ours empties the paragraphs it used to occupy but
    does not remove them, and the run of blanks pushes the abstract onto a
    second page. Dropping them and breaking the page keeps it on one.
    """
    end = find(doc, "ABSTRACT")
    if end < 0:
        return 0
    ps = doc.paragraphs
    start = end
    while start > 0 and not ps[start - 1].text.strip():
        start -= 1
    removed = 0
    for p in list(ps)[start:end]:
        drop(p)
        removed += 1
    doc.paragraphs[find(doc, "ABSTRACT")].paragraph_format.page_break_before = True
    return removed


# ------------------------------------------------------------------- body ---
def clear_body(doc):
    """Remove everything from the abstract note to the end; it is all placeholder."""
    start = find(doc, "Times New Roman 16, Bold, Upper Case")
    if start < 0:
        start = find(doc, "ABSTRACT") + 1
    for p in list(doc.paragraphs)[start:]:
        drop(p)
    # The template's trailing reference examples live in a different style.
    for p in list(doc.paragraphs):
        if "Apruzzese" in p.text or "Dasgupta" in p.text:
            drop(p)


def write_body(doc):
    for para in C.ABSTRACT.strip().split("\n\n"):
        add_par(doc, " ".join(para.split()), size=12, line=1.15,
                align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    page_break(doc)

    for blk in C.CHAPTERS:
        kind = blk[0]
        if kind == "h1":
            for n, line in enumerate(blk[1].split("\n")):
                add_par(doc, line, size=16, bold=True,
                        align=WD_ALIGN_PARAGRAPH.CENTER,
                        space_before=0 if n else 6, space_after=6 if n else 2,
                        line=1.5)
        elif kind == "h2":
            add_par(doc, blk[1], size=14, bold=True, space_before=12,
                    space_after=6, line=1.5)
        elif kind == "h3":
            add_par(doc, blk[1], size=12.5, bold=True, space_before=10,
                    space_after=4, line=1.5)
        elif kind == "p":
            add_par(doc, " ".join(blk[1].split()), size=12, line=1.15,
                    align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=8)
        elif kind == "b":
            add_par(doc, "•  " + " ".join(blk[1].split()), size=12,
                    line=1.15, indent=0.3, space_after=6,
                    align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        elif kind == "fig":
            add_figure(doc, blk[1], blk[2])
        elif kind == "tbl":
            add_table(doc, blk[1], blk[2])
        elif kind == "pb":
            page_break(doc)

    # Chapter 9, references.
    add_par(doc, "CHAPTER 9", size=16, bold=True,
            align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2, line=1.5)
    add_par(doc, "REFERENCES", size=16, bold=True,
            align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, line=1.5)
    for n, ref in enumerate(C.REFERENCES, 1):
        add_par(doc, f"[{n}]  {ref}", size=11, line=1.15, indent=0.35,
                space_after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    page_break(doc)

    # Appendix.
    add_par(doc, "APPENDIX A", size=16, bold=True,
            align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2, line=1.5)
    add_par(doc, "SAMPLE CODE", size=16, bold=True,
            align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, line=1.5)
    for title, path, note in APPENDIX_FILES:
        add_par(doc, title, size=12.5, bold=True, space_before=10, space_after=3)
        add_par(doc, note, size=11, italic=True, space_after=5)
        src = (ROOT / path).read_text().split("\n")
        body = "\n".join(src[:APPENDIX_LINES])
        p = add_par(doc, "", space_after=10)
        r = p.add_run(body)
        r.font.name = "Courier New"
        r.font.size = Pt(8)
        p.paragraph_format.line_spacing = 1.0


APPENDIX_LINES = 70
APPENDIX_FILES = [
    ("A.1  The model (src/etlguard/models/st_transformer.py)",
     "src/etlguard/models/st_transformer.py",
     "The spatio-temporal Transformer. The mask is given to the model as an "
     "input, not used only as a loss weight."),
    ("A.2  Scoring (src/etlguard/eval/scoring.py)",
     "src/etlguard/eval/scoring.py",
     "The two masking configurations, and the reason a task's descendants must "
     "be hidden along with it."),
    ("A.3  Fault injection (src/etlguard/simulator/faults.py)",
     "src/etlguard/simulator/faults.py",
     "The fifteen fault injectors. Each corrupts rows rather than statistics."),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(str(TEMPLATE))
    fill_front(doc)
    n = remove_acknowledgements(doc)
    nc = remove_certificate(doc)
    fill_toc(doc)
    fill_figures_tables(doc)
    fill_abbreviations(doc)
    nd = fill_date(doc)
    nsig = add_signatures(doc)
    nb = tidy_before_abstract(doc)
    clear_body(doc)
    write_body(doc)
    ns = add_page_numbers(doc)
    doc.save(str(OUT))
    print(f"  removed {n} acknowledgement paragraphs")
    print(f"  removed {nc} certificate paragraphs")
    print(f"  removed {nb} blank paragraphs before the abstract")
    print(f"  date line filled: {bool(nd)}")
    print(f"  signatures placed: {nsig}")
    print(f"  page numbering across {ns} sections")
    print(f"  wrote {OUT.relative_to(ROOT)}  "
          f"({OUT.stat().st_size // 1024} KB, "
          f"{len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables)")


if __name__ == "__main__":
    main()
