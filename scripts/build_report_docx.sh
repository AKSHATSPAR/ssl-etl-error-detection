#!/usr/bin/env bash
# Build the submittable report from docs/REPORT.md.
#
#   bash scripts/build_report_docx.sh
#
# Pandoc rather than a hand-written docx script: the report is long and
# table-heavy, and pandoc maps headings, tables and code blocks correctly
# without a few hundred lines of layout code that would then need maintaining
# every time a number changes.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=artifacts/report
mkdir -p "$OUT"

# A reference doc lets Word styles be adjusted later without touching the source.
if [ ! -f "$OUT/reference.docx" ]; then
  pandoc -o "$OUT/reference.docx" --print-default-data-file reference.docx > /dev/null 2>&1 \
    || pandoc --print-default-data-file reference.docx > "$OUT/reference.docx"
fi

cat > "$OUT/metadata.yaml" <<'YAML'
title: "Self-Supervised AI for ETL Error Detection and Root-Cause Localization"
subtitle: "B.Tech Deep Learning Project"
author: "Akshat Sparsh (23BDS0149) and Anuprakash Dubey (23BDS0162)"
date: "October 2026"
lang: en-GB
YAML

# Strip the markdown title block, which is eight lines including the byline.
# The metadata block supplies the title page, so leaving it in duplicates it.
tail -n +9 docs/REPORT.md > "$OUT/body.md"

# --metadata-file, not a positional input: passing the YAML as an input file
# renders it as body text instead of populating the title page.
# No --number-sections either, because the headings are already numbered and
# pandoc would stack a second set on top ("2 1. Introduction").
# --resource-path so the figure paths in REPORT.md, which are relative to the
# repo root, still resolve after body.md is written into artifacts/report/.
pandoc "$OUT/body.md" \
  --from markdown+pipe_tables+backtick_code_blocks+implicit_figures \
  --resource-path=".:artifacts/results:artifacts/dataset_report" \
  --to docx \
  --metadata-file="$OUT/metadata.yaml" \
  --reference-doc="$OUT/reference.docx" \
  --toc --toc-depth=2 \
  -o "$OUT/ETL_Error_Detection_Report.docx"

echo "wrote $OUT/ETL_Error_Detection_Report.docx"
