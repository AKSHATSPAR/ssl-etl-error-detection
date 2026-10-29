#!/usr/bin/env bash
# Assemble everything for the day into one folder outside the repository.
#
#   bash scripts/package_submission.sh
#
# Three sub-folders, numbered so they sort in the order they are needed. The
# folder lives outside the repo on purpose: it is a snapshot for handing in, not
# a source of truth, and the canonical copies stay under artifacts/.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-$HOME/Downloads/ETL_Project_Submission}"
rm -rf "$DEST"
mkdir -p "$DEST/1_SUBMIT_THESE" "$DEST/2_YOUR_PREPARATION" \
         "$DEST/3_ON_THE_DAY/figures"

# ---- 1. What gets handed in ----
cp artifacts/report/ETL_Error_Detection_Report.docx "$DEST/1_SUBMIT_THESE/"
cp artifacts/report/Literature_Review.pdf          "$DEST/1_SUBMIT_THESE/"
cp artifacts/report/Literature_Review.docx         "$DEST/1_SUBMIT_THESE/"
cp artifacts/slides/ETL_Project_Viva.pptx          "$DEST/1_SUBMIT_THESE/"
cp artifacts/slides/ETL_Review2_Deck.pptx            "$DEST/1_SUBMIT_THESE/"
cp artifacts/web/pipeline_demo.html                "$DEST/1_SUBMIT_THESE/"
cat > "$DEST/1_SUBMIT_THESE/CODE_REPOSITORY.txt" <<'TXT'
Self-Supervised AI for ETL Error Detection and Root-Cause Localization
B.Tech Deep Learning Project, October 2026

Akshat Sparsh      23BDS0149
Anuprakash Dubey   23BDS0162

Source code, tests and all result files:

    https://github.com/AKSHATSPAR/ssl-etl-error-detection

The repository is private. Either add the examiner as a collaborator or make it
public before submission, otherwise the link will not open for them.

Everything in this folder was generated from that repository. To reproduce any
number, see Appendix A of the report.
TXT

# ---- 2. Preparation, not for submission ----
cp artifacts/report/Complete_Explainer.pdf            "$DEST/2_YOUR_PREPARATION/"
cp artifacts/report/Presentation_and_Demo_Guide.pdf   "$DEST/2_YOUR_PREPARATION/"
cp artifacts/report/Speaking_Script.pdf               "$DEST/2_YOUR_PREPARATION/"

# ---- 3. Everything needed in the room ----
cp artifacts/web/pipeline_demo.html "$DEST/3_ON_THE_DAY/"
cp artifacts/slides/ETL_Project_Viva.pptx "$DEST/3_ON_THE_DAY/"
cp artifacts/slides/ETL_Review2_Deck.pptx "$DEST/3_ON_THE_DAY/"
cp artifacts/slides/bw/*.png "$DEST/3_ON_THE_DAY/figures/"
cp artifacts/dataset_report/fig*.png artifacts/results/fig*.png "$DEST/3_ON_THE_DAY/figures/"
cat > "$DEST/3_ON_THE_DAY/SETUP.txt" <<'TXT'
SETUP, ON THIS MAC, BEFORE THEY WALK IN
=======================================

Open these three things and leave them open.

1. The demo, in a browser tab. Double-click pipeline_demo.html in this folder.
   Leave it on a broken run. Do not press "Show what actually happened" yet.

2. The slides. Double-click ETL_Project_Viva.pptx. Use presenter view so you
   can see the speaker notes; every slide has them.

3. A terminal in the project folder, with the demo command typed but NOT run,
   so you only have to press Enter if someone asks to see it run for real:

       cd /Users/akshat/Downloads/dl_project
       PYTHONPATH=src .venv/bin/python scripts/demo.py --n 2

   The visual demo needs none of this. It is one file and it opens by
   double-clicking. The terminal command is only for anyone who asks to see
   it run against the real model.

Then:

   - Do Not Disturb on. A notification mid-demo is the most common failure.
   - Charger plugged in.
   - Close everything else.
   - Test the projector before the room fills. If the amber highlights in the
     diagram look washed out, say "the highlighted ones", not "the orange ones".

ORDER
-----
Slides 1 to 18, then Cmd+Tab to the browser tab, run the two-minute demo
walkthrough (pages 2 to 4 of the Demo Guide), then questions.

IF SOMETHING BREAKS
-------------------
Demo will not open:   drag the HTML file onto an open browser window.
Slides will not open: the figures/ folder has every picture as a PNG.
Both dead:            fig2_propagation.png alone carries the main idea.
TXT

echo "packaged into $DEST"
find "$DEST" -type f | sed "s|$DEST/|  |" | sort
echo
du -sh "$DEST" | awk '{print "total: "$1}'
