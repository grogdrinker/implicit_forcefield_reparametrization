# iftff — implicit-function-theorem force-field optimization

`iftff` trains molecular-mechanics **force fields** so that experimentally
determined structures sit at **minima** of the modeled energy surface, and then
lets you run **energy minimization** and **molecular dynamics** with the force
field you trained.

It re-parametrizes the standard **AMBER** force field by attaching one
dimensionless multiplier to every parameter (bonds, angles, charges, van der
Waals) — `1.0` means "unchanged" — and learns those multipliers by
back-propagating through the *result* of energy minimization using the
**implicit function theorem**, so it never has to differentiate through the
minimizer's iterations. The result is a normal OpenMM force-field XML you can
use anywhere.

---

## Install

OpenMM and PDBFixer come from **conda** (PDBFixer isn't on PyPI, and OpenMM's
GPU builds live on conda-forge); everything else is installed by pip:

```sh
conda create -n iftff python=3.11
conda activate iftff
conda install -c conda-forge openmm pdbfixer      # the two conda-only deps
pip install -e .                                  # the iftff package + tool
```

That installs the `iftff` Python package, its pip dependencies (NumPy, PyTorch,
tqdm), and the **`iftff` command-line tool**. A CUDA GPU is used automatically
if available, but everything also runs on CPU.

Check it worked:

```sh
iftff --help
```

---

## Run the example first

A complete, self-contained example ships in the repo. It needs nothing but the
install above — one small amyloid structure (`examples/8IZZ.pdb`) is bundled.

```sh
bash example.sh
```

In under a minute it runs the whole workflow on that structure and writes the
results to `example_output/`:

1. **minimize** it with stock AMBER (the baseline);
2. **train** a force field from it → `example_output/trained.xml`;
3. **minimize** it again with the trained force field;
4. run a short **MD** trajectory with the trained force field.

Open `example.sh` — it is heavily commented and is the fastest way to learn the
three commands below.

---

## Prefer a window? `iftff gui`

Install the GUI with:

```sh
pip install -e ".[gui]"
```

then launch it:

```sh
iftff gui
```

It opens a small graphical interface with one tab per task (Train / Minimize /
MD): pick your files, adjust any parameter (the common ones are shown; the rest
live under **Advanced options**), and press **Run** — the output streams live in
the log at the bottom. It runs the same commands as the CLI and prints each one.

## The three commands

The tool does three things. The natural order is **train → minimize / md**:
train a force field from your structures, then use that force field.

```
iftff train     --pdb-dir <dir>  --output-xml <ff.xml>   # structures -> trained force field
iftff minimize  --pdb <pdb>      [--xml <ff.xml>]         # energy-minimize one structure
iftff md        --pdb <pdb>      [--xml <ff.xml>]         # run molecular dynamics
iftff gui                                                 # the same, in a window
```

Every command prints its full option list with `iftff <command> --help`.

**The default force field is the bundled reparametrised one** (reg=0.01) — the
recommended production force field this method produces. So `--xml` is optional
for `minimize` and `md`; pass it only to use a different force field (a force
field you trained, or stock AMBER via `--xml $(python -c "import iftff;print(iftff.forcefield('amber99sb'))")`).
The five bundled reparametrised force fields ship inside the package
(`iftff/data/AmberReparametrised_*.xml`).

### 1. `iftff train` — learn a force field from structures

Point it at a folder of PDB files; it learns parameter multipliers so those
structures sit at energy minima, and writes a trained force-field XML.

```sh
iftff train \
    --pdb-dir  my_structures/ \   # folder of PDB files to fit
    --n-epochs 10 \               # outer optimization epochs
    --lr       1e-3 \             # Adam learning rate
    --reg-lambda 0.01 \           # keep parameters near stock AMBER (0 = no penalty)
    --output-xml my_ff.xml        # the trained force field
```

**Input:** a directory of PDBs. **Output:** `my_ff.xml` (the trained force
field, a normal OpenMM XML), a per-epoch checkpoint XML in `--checkpoint-dir`,
and `per_protein_losses.tsv` (the per-structure RMSD each epoch).

`--reg-lambda` controls how far training may pull parameters from stock AMBER:
higher = stay closer to the original, lower = fit the structures more
aggressively. Start around `0.01`.

### 2. `iftff minimize` — energy-minimize one structure

```sh
iftff minimize --pdb my_protein.pdb --output-dir out/      # uses the default (reg=0.01) FF
iftff minimize --pdb my_protein.pdb --xml my_ff.xml --output-dir out/   # or your own FF
```

Fixes the structure (adds missing atoms/hydrogens with PDBFixer),
energy-minimizes it with the force field, prints `Minimized: RMSD=… nm` (how far
the heavy atoms moved from the input, Kabsch-aligned, hydrogens excluded), and
writes the minimized PDB — then stops.

### 3. `iftff md` — run molecular dynamics

```sh
iftff md --pdb my_protein.pdb --output-dir out/ \
    --n-steps 250000 --traj-interval 1000
```

Same preparation and minimization, then Langevin MD (defaults: 300 K, friction
1 ps⁻¹, 2 fs timestep). `--n-steps 250000` at 2 fs is 0.5 ns. Prints the final
RMSD and drift and writes `trajectories/trajectory.pdb` plus a `report.txt`.
Change the ensemble with `--temperature`, `--friction`, `--timestep`.

---

## License & citation

Licensed under the **Apache License 2.0** — see the [`LICENSE`](LICENSE) file.
If you use this in academic work, please cite the accompanying paper (see the
repository page).
