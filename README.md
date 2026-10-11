# implicit_forcefield_reparametrization

Bilevel reparametrization of molecular-mechanics force fields via the implicit
function theorem (IFT). Starting from stock AMBER99SB, `iftff` tunes the
force-field parameters so that experimental structures sit at minima of the
energy landscape — using OpenMM for the energy minimization and PyTorch for the
gradients.

## Installation

```bash
conda install -c conda-forge openmm pdbfixer   # not on PyPI
pip install -e .                               # installs the `iftff` command
pip install -e ".[gui]"                        # optional graphical interface
```

Requires Python ≥ 3.9.

## Quick start

```bash
bash example.sh     # minimize → train → minimize → short MD, on a bundled structure
```

or call the commands directly:

```bash
iftff minimize --pdb examples/8IZZ.pdb --xml iftff/data/amber99sb.xml --output-dir out/min
iftff train    --pdb-dir examples --n-epochs 3 --reg-lambda 0.01 --output-xml out/trained.xml
iftff md       --pdb examples/8IZZ.pdb --xml out/trained.xml --output-dir out/md
iftff gui      # the above, in a window
```

Five ready-to-use reparametrized force fields ship in `iftff/data/`
(unregularized + λ = 0.005 / 0.01 / 0.05 / 0.1).

## Package layout

```
iftff/
  cli.py            `iftff <command>` entry point (train | md | minimize | gui)
  train.py          outer loop: reparametrize a force field from a set of PDBs
  run.py            minimize / run MD with a given force-field XML
  __init__.py       public API + forcefield() path resolver
  data/*.xml        stock amber99sb + the five reparametrized force fields
  gui/              Tkinter graphical interface
  src/
    force_field.py  differentiable (PyTorch) AMBER energy and forces
    ift.py          d RMSD / d parameters via the implicit function theorem
    openmm_utils.py build systems, energy-minimize, run MD (OpenMM)
    topology.py     PDB → topology and atom-class mappings
    xml_io.py       read / write OpenMM force-field XML
    config.py       TrainingConfig / SimulationConfig
    utils.py        Kabsch RMSD, checkpoints
examples/8IZZ.pdb   small amyloid used by example.sh
```

## How it works

Two nested loops. The **inner** loop energy-minimizes each structure under the
current parameters (OpenMM). The **outer** loop measures the RMSD between the
minimized and the experimental structure and updates the parameters to reduce it
(Adam). The outer gradient is obtained from the converged minimum alone via the
implicit function theorem,

```
∇θ L = − (∂²F / ∂X∂θ)ᵀ H⁻¹ ∇X L ,   H = ∂²F / ∂X²  at the minimum,
```

so its cost is independent of the number of minimization steps and the minimizer
is used as a black box. Parameters are trained as multipliers on the stock AMBER
values (1.0 = unchanged); dihedral and 1–4 nonbonded terms are held fixed.

## Python API

```python
import iftff

iftff.forcefield("amber99sb")     # path to stock AMBER XML (training starts here)
iftff.forcefield("lambda0.01")    # path to a reparametrized force field
iftff.DEFAULT_FORCEFIELD          # recommended production force field (λ = 0.01)

from iftff import (
    create_base_system_and_modeller,  # PDB + XML → OpenMM system
    minimize_with_system,             # energy-minimize to a stationary point
    compute_ift_gradient,             # d RMSD / d parameters (IFT)
    AmberForceFieldMP,                # differentiable AMBER force field
    kabsch_rmsd,                      # heavy-atom RMSD after Kabsch alignment
    TrainingConfig, SimulationConfig,
    load_xml, dump_xml,               # read / write force-field XML
)
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
