#!/usr/bin/env bash
# Build the literature review as both .docx and .pdf.
#
#   bash scripts/build_litreview.sh
#
# The review matrix is eight columns wide, so it gets its own landscape pages in
# the PDF via a CSS named page. In portrait at a readable font it simply will not
# fit, and shrinking the type far enough to make it fit makes it unreadable.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=artifacts/report
mkdir -p "$OUT"

cat > "$OUT/lit_meta.yaml" <<'YAML'
title: "Literature Review"
subtitle: "Self-Supervised AI for ETL Error Detection and Root-Cause Localization"
author: "Akshat"
date: "September 2026"
lang: en-GB
YAML

# Drop the markdown title block; the metadata supplies the title page.
tail -n +5 docs/LITERATURE_REVIEW.md > "$OUT/lit_body.md"

# ---- Word ----
pandoc "$OUT/lit_body.md" \
  --from markdown+pipe_tables \
  --to docx \
  --metadata-file="$OUT/lit_meta.yaml" \
  --toc --toc-depth=2 \
  -o "$OUT/Literature_Review.docx"

# ---- PDF, via HTML so WeasyPrint can do the page layout ----
cat > "$OUT/lit.css" <<'CSS'
@page { size: A4 portrait; margin: 20mm 18mm; 
        @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
@page wide { size: A4 landscape; margin: 14mm 12mm; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 19pt; margin: 0 0 4pt; }
h1.title { font-size: 22pt; text-align: center; margin-top: 55mm; }
p.subtitle { text-align: center; font-size: 13pt; color: #444; }
p.author, p.date { text-align: center; font-size: 11pt; color: #444; margin: 2pt; }
h2 { font-size: 14pt; margin-top: 20pt; border-bottom: 1px solid #ccc;
     padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 14pt; page-break-after: avoid; }
blockquote { border-left: 3px solid #d0a020; background: #fdf8ec; margin: 10pt 0;
             padding: 7pt 11pt; font-size: 10pt; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 3.5pt 5pt; text-align: left;
         vertical-align: top; }
/* Without this a row straddles the page break and renders with half its cells
   blank, which reads as missing data rather than as a page turn. */
tr { page-break-inside: avoid; break-inside: avoid; }
thead { display: table-header-group; }
th { background: #eee; font-weight: bold; }
/* The review matrix is the only very wide table: give it landscape pages. */
table.matrix { font-size: 7.2pt; page: wide; }
table.matrix th, table.matrix td { padding: 2pt 3pt; }
/* Return to portrait after the landscape matrix, otherwise the prose that
   follows it keeps running across landscape pages. */
.resume-portrait { page: auto; break-before: page; height: 0; }
ul { margin: 4pt 0 4pt 16pt; }
li { margin: 1.5pt 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 16pt 0; }
CSS

pandoc "$OUT/lit_body.md" \
  --from markdown+pipe_tables \
  --to html5 --standalone \
  --metadata-file="$OUT/lit_meta.yaml" \
  --toc --toc-depth=2 \
  --css=lit.css \
  -o "$OUT/lit.html"

# Tag the widest table so the CSS can move it onto landscape pages. It is the
# only one with a "Key Contribution" header, which makes it easy to find.
python3 - "$OUT/lit.html" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
def tag(m):
    if "Key Contribution" not in m.group(0):
        return m.group(0)
    return (m.group(0).replace("<table>", '<table class="matrix">', 1)
            + '<div class="resume-portrait"></div>')
s = re.sub(r"<table>.*?</table>", tag, s, flags=re.S)
p.write_text(s)
PY

weasyprint "$OUT/lit.html" "$OUT/Literature_Review.pdf"

echo "wrote $OUT/Literature_Review.docx"
echo "wrote $OUT/Literature_Review.pdf"
