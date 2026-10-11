#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# example.sh - a complete, self-contained tour of the `iftff` command-line tool.
#
# It runs the three things the tool does, in the order you would normally use
# them, on one small bundled amyloid structure (examples/8IZZ.pdb, 740 atoms):
#
#   1. minimize  a structure with the stock AMBER force field   (the baseline)
#   2. train     a force field from that structure  ->  trained.xml
#   3. minimize  the same structure with YOUR trained force field
#   4. md        run a short molecular-dynamics trajectory with the trained FF
#
# Everything it needs ships with the package - no downloads, no extra data.
# The whole run takes well under a minute on a laptop CPU.
#
# Prerequisite:  pip install -e .        (installs the `iftff` command)
# Run it with:   bash example.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDB="$HERE/examples/8IZZ.pdb"           # the input structure (a small amyloid)
OUT="$HERE/example_output"              # everything this script writes lands here
BASE_XML="$HERE/iftff/data/amber99sb.xml"   # stock AMBER, shipped with the package
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=========================================================="
echo " STEP 1/4  Minimize with the ORIGINAL AMBER force field"
echo "           (how far does the experimental structure move"
echo "            when relaxed under stock AMBER? = the baseline)"
echo "=========================================================="
iftff minimize --pdb "$PDB" --xml "$BASE_XML" --output-dir "$OUT/min_original"

echo
echo "=========================================================="
echo " STEP 2/4  Train a force field from this structure"
echo "           (produces a trained XML; normally you point"
echo "            --pdb-dir at a whole folder of structures)"
echo "=========================================================="
iftff train \
    --pdb-dir "$HERE/examples" \
    --n-epochs 3 \
    --reg-lambda 0.01 \
    --output-xml "$OUT/trained.xml" \
    --checkpoint-dir "$OUT/checkpoints"

echo
echo "=========================================================="
echo " STEP 3/4  Minimize again, now with YOUR trained force field"
echo "           (the structure should sit closer to a minimum:"
echo "            smaller RMSD than step 1)"
echo "=========================================================="
iftff minimize --pdb "$PDB" --xml "$OUT/trained.xml" --output-dir "$OUT/min_trained"

echo
echo "=========================================================="
echo " STEP 4/4  Short MD with the trained force field"
echo "           (500 steps; writes a trajectory you can open"
echo "            in PyMOL/VMD and a report.txt with drift)"
echo "=========================================================="
iftff md --pdb "$PDB" --xml "$OUT/trained.xml" --output-dir "$OUT/md" \
    --n-steps 500 --traj-interval 100

echo
echo "=========================================================="
echo " Done. Look in:  $OUT"
echo "   min_original/8IZZ/  - baseline minimized structure + report"
echo "   trained.xml         - the force field you just trained"
echo "   min_trained/8IZZ/   - minimized under the trained FF"
echo "   md/8IZZ/            - trajectory.pdb + report.txt"
echo "=========================================================="
