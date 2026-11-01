"""Read the rendered report and write its real page numbers back into content.py.

The page numbers in the contents, the list of figures and the list of tables
were estimates. This renders the document, finds which sheet each heading and
caption actually landed on, reads the printed number off that sheet, and
rewrites the three lists. Nothing here changes the layout, so one pass settles
it, but running it twice is harmless and confirms the numbers held.

Usage: python scripts/report_fill/measure_pages.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

import content as C  # noqa: E402

DOCX = ROOT / "artifacts/report_vit/Project_Report_Akshat_Anuprakash.docx"
PDF = DOCX.with_suffix(".pdf")
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

ROMAN = re.compile(r"^[ivxl]+$")
ARABIC = re.compile(r"^\d+$")


def render():
    PDF.unlink(missing_ok=True)
    subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf",
                    "--outdir", str(PDF.parent), str(DOCX)],
                   check=True, capture_output=True)


def sheets():
    """Return one (printed_label, text) pair per sheet of the PDF."""
    out = []
    for page in pypdf.PdfReader(str(PDF)).pages:
        lines = [l.strip() for l in (page.extract_text() or "").splitlines()
                 if l.strip()]
        label = ""
        # The footer field extracts as a line of its own, roman or arabic.
        for cand in (lines[:1] + lines[-1:]):
            if ROMAN.match(cand.lower()) or ARABIC.match(cand):
                label = cand
                break
        out.append((label, " ".join(" ".join(lines).split())))
    return out


def _flat(s):
    """Lower case, one space between words, punctuation dropped.

    A heading that wraps loses its punctuation when the page is read back, so
    "APPENDIX A - SAMPLE CODE" comes out as "APPENDIX A SAMPLE CODE".
    """
    return " ".join(re.sub(r"[^\w\s]", " ", s).lower().split())


def locate(pages, needle, first=0):
    """The printed number of the earliest sheet at or after *first* holding it."""
    want = " ".join(needle.split()).lower()
    loose = _flat(needle)
    for i in range(first, len(pages)):
        label, text = pages[i]
        if want in text.lower() or loose in _flat(text):
            return label, i
    return "", -1


def body_start(pages):
    for i, (label, _) in enumerate(pages):
        if label == "1":
            return i
    return 0


def main():
    render()
    pages = sheets()
    body = body_start(pages)
    print(f"  {len(pages)} sheets, body opens on sheet {body + 1}")

    toc = []
    misses = []
    for sl, title, old in C.TOC:
        # Front matter is roman and sits before the body; everything else is
        # searched from the body onward so the contents page cannot match itself.
        # Front matter is roman and sits before the body. Body headings are
        # searched from the body onward so the contents cannot match itself.
        if ROMAN.match(old.lower()):
            label, _ = locate(pages, title, 3)
        elif sl:
            # Both halves occur in prose on their own: the body says "in
            # Chapter 2" on an earlier page, and "conclusion" earlier still.
            # Together they only match where the chapter actually opens.
            label, _ = locate(pages, f"CHAPTER {sl.rstrip('.')} {title}", body)
            if not label:
                label, _ = locate(pages, title, body)
        else:
            label, _ = locate(pages, title, body)
        if not label:
            misses.append(("toc", title, old))
            label = old
        toc.append((sl, title, label))

    figs = []
    for no, cap, old in C.FIGURES:
        label, _ = locate(pages, f"Figure {no}", body)
        if not label:
            misses.append(("fig", no, old))
            label = old
        figs.append((no, cap, label))

    tbls = []
    for no, cap, old in C.TABLES:
        label, _ = locate(pages, f"Table {no}", body)
        if not label:
            misses.append(("tbl", no, old))
            label = old
        tbls.append((no, cap, label))

    src = (ROOT / "scripts/report_fill/content.py").read_text()

    def block(name, rows, width):
        body_ = "\n".join(f" ({', '.join(repr(v) for v in r)})," for r in rows)
        return f"{name} = [\n{body_}\n]"

    for name, rows in (("TOC", toc), ("FIGURES", figs), ("TABLES", tbls)):
        start = src.index(f"{name} = [")
        end = src.index("\n]", start) + 2
        src = src[:start] + block(name, rows, 0) + src[end:]
    (ROOT / "scripts/report_fill/content.py").write_text(src)

    changed = sum(1 for (a, b, c), (d, e, f) in zip(C.TOC, toc) if c != f)
    print(f"  contents: {changed} of {len(toc)} page numbers corrected")
    print(f"  figures:  {sum(1 for a, b in zip(C.FIGURES, figs) if a[2] != b[2])} corrected")
    print(f"  tables:   {sum(1 for a, b in zip(C.TABLES, tbls) if a[2] != b[2])} corrected")
    if misses:
        print(f"  !! {len(misses)} not found, left as they were:")
        for kind, what, old in misses:
            print(f"     {kind} {what!r} (kept {old})")


if __name__ == "__main__":
    main()
