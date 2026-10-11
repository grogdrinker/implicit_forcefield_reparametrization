#!/usr/bin/env python3
import argparse
import gc
import os
import sys
import time

import numpy as np
from openmm import app, unit, LocalEnergyMinimizer, LangevinIntegrator, Context, Platform
from pdbfixer import PDBFixer

from iftff import kabsch_rmsd, DEFAULT_FORCEFIELD

# Fixed integrator seed for reproducible trajectories. This is deliberately not
# a user-facing option: the random seed is not something the user should change.
_FIXED_SEED = 42


def build_parser():
    p = argparse.ArgumentParser(description='Run OpenMM simulation with a trained force field')
    p.add_argument('--pdb', required=True, help='Input PDB file')
    p.add_argument('--xml', default=DEFAULT_FORCEFIELD,
                   help='Force field XML file (default: the bundled reg=0.01 '
                        'reparametrised force field)')
    p.add_argument('--output-dir', default='./simulation_output', help='Output directory')
    p.add_argument('--temperature', type=float, default=300.0, help='Temperature (K)')
    p.add_argument('--friction', type=float, default=1.0, help='Friction coefficient (1/ps)')
    p.add_argument('--timestep', type=float, default=0.002, help='Integration timestep (ps)')
    p.add_argument('--n-steps', type=int, default=250000, help='Number of MD steps')
    p.add_argument('--traj-interval', type=int, default=10000, help='Steps between trajectory frames')
    p.add_argument('--min-tolerance', type=float, default=1.0, help='Minimization tolerance')
    p.add_argument('--min-max-iterations', type=int, default=1000, help='Max minimization iterations')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device: auto (default), cuda, or cpu')
    p.add_argument('--minimize-only', action='store_true',
                   help='Only fix and energy-minimize; write the minimized structure and skip MD.')
    return p


def compute_rmsd(pos_a, pos_b, atom_indices=None):
    # RMSD is always computed over heavy atoms only (hydrogens excluded).
    return kabsch_rmsd(pos_a, pos_b, atom_indices=atom_indices).item()


def heavy_atom_indices(topology):
    """Indices of non-hydrogen atoms in an OpenMM topology (for RMSD)."""
    import torch as _torch
    idx = [i for i, a in enumerate(topology.atoms())
           if a.element is None or a.element.symbol != 'H']
    return _torch.tensor(idx, dtype=_torch.long)


def resolve_platform(device):
    if device == 'cpu':
        return Platform.getPlatformByName('CPU')
    if device == 'cuda':
        try:
            return Platform.getPlatformByName('CUDA')
        except Exception:
            print('CUDA platform not available, falling back to auto')
    for name in ['CUDA', 'OpenCL', 'CPU']:
        for i in range(Platform.getNumPlatforms()):
            if Platform.getPlatform(i).getName() == name:
                return Platform.getPlatform(i)
    return Platform.getPlatform(0)

def fix_pdb(pdb_path, output_dir, ph=7.0):
    # Strip HETATM records (waters, ions, ligands - none of the globular structures
    # here have a chain-embedded modified residue like MSE that would need PDBFixer's
    # replaceNonstandardResidues() instead of outright removal) before PDBFixer ever
    # sees them. The bare amber99sb.xml used here has no water/ion parameters loaded
    # alongside it, so createSystem() fails on any of them - and some (e.g. DOD, UNX)
    # even crash PDBFixer's addMissingHydrogens(), which tries to download an online
    # CCD definition for unrecognized residue names.
    with open(pdb_path) as f:
        lines = [line for line in f if not line.startswith('HETATM')]
    stripped_path = os.path.join(output_dir, 'stripped_input.pdb')
    with open(stripped_path, 'w') as f:
        f.writelines(lines)

    fixer = PDBFixer(filename=stripped_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    fixed_path = os.path.join(output_dir, 'fixed_input.pdb')
    with open(fixed_path, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return app.PDBFile(fixed_path)

def main():
    _run(build_parser().parse_args())


def main_minimize():
    args = build_parser().parse_args()
    args.minimize_only = True
    _run(args)


def _run(args):

    if not os.path.exists(args.pdb):
        print(f'PDB file not found: {args.pdb}')
        sys.exit(1)
    if not os.path.exists(args.xml):
        print(f'XML file not found: {args.xml}')
        sys.exit(1)

    pdb_id = os.path.splitext(os.path.basename(args.pdb))[0]
    out_dir = os.path.join(args.output_dir, pdb_id)
    traj_dir = os.path.join(out_dir, 'trajectories')
    os.makedirs(traj_dir, exist_ok=True)

    pdb = fix_pdb(args.pdb, out_dir)

    temperature = args.temperature * unit.kelvin
    friction = args.friction / unit.picoseconds
    timestep = args.timestep * unit.picoseconds

    platform = resolve_platform(args.device)
    print(f'Platform: {platform.getName()}')
    print(f'PDB: {args.pdb}')
    print(f'XML: {args.xml}')
    print(f'Output: {out_dir}')
    print(f'Temperature: {args.temperature} K, Steps: {args.n_steps}, Timestep: {args.timestep} ps')
    print('=' * 60)

    start_t = time.time()

    ff = app.ForceField(args.xml)

    modeller = app.Modeller(pdb.topology, pdb.getPositions())
    topology = modeller.getTopology()
    init_pos = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    n_atoms = topology.getNumAtoms()
    heavy_idx = heavy_atom_indices(topology)   # RMSD over heavy atoms only

    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff)

    integrator = LangevinIntegrator(temperature, friction, timestep)
    context = Context(system, integrator, platform)
    context.setPositions(modeller.getPositions())

    LocalEnergyMinimizer.minimize(context, tolerance=args.min_tolerance, maxIterations=args.min_max_iterations)
    state = context.getState(getPositions=True, getEnergy=True)
    min_pos = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    min_pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    rmsd_min = compute_rmsd(min_pos, init_pos, atom_indices=heavy_idx)

    min_pdb_path = os.path.join(traj_dir, 'minimized.pdb')
    with open(min_pdb_path, 'w') as f:
        app.PDBFile.writeFile(topology, state.getPositions(), f)
    print(f'  Minimized: RMSD={rmsd_min:.4f} nm, PE={min_pe:.0f} kJ/mol')

    if args.minimize_only:
        print('  Minimize-only requested; skipping MD.')
        return

    del context

    traj_path = os.path.join(traj_dir, 'trajectory.pdb')
    traj_integrator = LangevinIntegrator(temperature, friction, timestep)
    traj_integrator.setRandomNumberSeed(_FIXED_SEED)
    traj_context = Context(system, traj_integrator, platform)
    traj_context.setPositions(min_pos)

    n_frames = 0
    model_idx = 0
    with open(traj_path, 'w') as f:
        app.PDBFile.writeHeader(topology, f)
        state = traj_context.getState(getPositions=True)
        app.PDBFile.writeModel(topology, state.getPositions(), f, modelIndex=model_idx)
        n_frames += 1
        model_idx += 1

        for step in range(1, args.n_steps + 1):
            traj_integrator.step(1)
            if step % args.traj_interval == 0:
                state = traj_context.getState(getPositions=True)
                app.PDBFile.writeModel(topology, state.getPositions(), f, modelIndex=model_idx)
                n_frames += 1
                model_idx += 1
            if step % 50000 == 0:
                elapsed = time.time() - start_t
                print(f'  Step {step}/{args.n_steps} ({elapsed:.0f}s)')

        state = traj_context.getState(getPositions=True, getEnergy=True)
        final_pos = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        final_pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        rmsd_final = compute_rmsd(final_pos, init_pos, atom_indices=heavy_idx)
        drift = compute_rmsd(final_pos, min_pos, atom_indices=heavy_idx)

    elapsed = time.time() - start_t

    report_path = os.path.join(out_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(f'pdb\t{args.pdb}\n')
        f.write(f'force_field\t{args.xml}\n')
        f.write(f'n_atoms\t{n_atoms}\n')
        f.write(f'rmsd_min\tnm\t{rmsd_min:.4f}\n')
        f.write(f'pe_min\tkJ/mol\t{min_pe:.0f}\n')
        f.write(f'rmsd_final\tnm\t{rmsd_final:.4f}\n')
        f.write(f'drift\tnm\t{drift:.4f}\n')
        f.write(f'pe_final\tkJ/mol\t{final_pe:.0f}\n')
        f.write(f'n_frames\t{n_frames}\n')
        f.write(f'elapsed\ts\t{elapsed:.1f}\n')

    print(f'  Final: RMSD={rmsd_final:.4f} nm, Drift={drift:.4f} nm, PE={final_pe:.0f} kJ/mol')
    print(f'  Frames: {n_frames}')
    print(f'  Time: {elapsed:.1f}s')
    print(f'  Report: {report_path}')
    print(f'  Trajectory: {traj_path}')
    print('Done.')

    del traj_context, traj_integrator
    gc.collect()

if __name__ == '__main__':
    main()
