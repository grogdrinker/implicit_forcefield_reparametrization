#!/usr/bin/env python3
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from openmm import app, unit

from iftff import (
    DATA_DIR,
    AmberForceFieldMP,
    TrainingConfig,
    create_base_system_and_modeller,
    create_topology,
    compute_ift_gradient,
    dump_xml,
    kabsch_rmsd,
    minimize_with_pytorch,
    minimize_with_system,
    modify_system_forces,
    save_checkpoint,
)

# Default location for a trained force field: the package's data/ folder, the
# same place the bundled reparametrised force fields live.
_DEFAULT_OUTPUT_XML = os.path.join(DATA_DIR, 'AmberReparametrised.xml')


from tqdm import tqdm


def build_parser():
    p = argparse.ArgumentParser(description='Train amyloid force field with IFT')
    p.add_argument('--pdb-dir', required=True, help='Directory with PDB files')
    p.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    p.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device: auto (default), cuda, or cpu')
    p.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    p.add_argument('--output-xml', default=_DEFAULT_OUTPUT_XML,
                   help='Final XML output path (default: the package data/ folder)')
    p.add_argument('--inner-min-tolerance', type=float, default=0.001)
    p.add_argument('--inner-min-max-iterations', type=int, default=100000)
    p.add_argument('--cg-max-iter', type=int, default=200)
    p.add_argument('--cg-damping', type=float, default=1.0)
    p.add_argument('--cg-fallback-damping', type=float, default=10.0)
    p.add_argument('--cg-tolerance', type=float, default=1e-8)
    p.add_argument('--reg-lambda', type=float, default=0.0, help='L2 regularization strength on FF params (default: 0.0). Parameters pulled toward 1.0.')
    p.add_argument('--gradient-clip-norm', type=float, default=1.0)
    p.add_argument('--save-interval', type=int, default=10)
    return p


def main():
    args = build_parser().parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    config = TrainingConfig(
        n_epochs=args.n_epochs,
        outer_lr=args.lr,
        device=device,
        inner_min_tolerance=args.inner_min_tolerance,
        inner_min_max_iterations=args.inner_min_max_iterations,
        cg_max_iter=args.cg_max_iter,
        cg_damping=args.cg_damping,
        cg_fallback_damping=args.cg_fallback_damping,
        cg_tolerance=args.cg_tolerance,
        gradient_clip_norm=args.gradient_clip_norm,
        reg_lambda=args.reg_lambda,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
    )

    pdb_dir = Path(args.pdb_dir)
    pdb_paths = sorted([p for p in pdb_dir.glob('*.pdb') if p.stat().st_size > 0])
    if not pdb_paths:
        print(f'No PDB files found in {args.pdb_dir}')
        sys.exit(1)

    print(f'PDB directory: {args.pdb_dir} ({len(pdb_paths)} files)')
    print(f'Device: {device}')
    print(f'Epochs: {config.n_epochs}, LR: {config.outer_lr}')
    print(f'Checkpoint dir: {config.checkpoint_dir}')
    print(f'Output XML: {args.output_xml}')
    print('=' * 60)

    total_start = time.time()

    torch.manual_seed(config.seed)
    ff = AmberForceFieldMP().to(device)
    for p in ff.parameters():
        p.data.fill_(1.0)

    optimizer = torch.optim.Adam(ff.parameters(), lr=config.outer_lr)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_xml)), exist_ok=True)

    losses = []
    mult_history = []

    per_prot_path = os.path.join(config.checkpoint_dir, 'per_protein_losses.tsv')
    per_prot_file = open(per_prot_path, 'w')
    per_prot_file.write('epoch\tpdb_id\tn_atoms\tloss\trmsd_before\trmsd_after\tforce_norm\n')

    for epoch in range(config.n_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        n_processed = 0

        pbar = tqdm(pdb_paths, desc=f'Epoch {epoch}/{config.n_epochs}')
        for pdb_path in pbar:
            pdb_id = pdb_path.stem
            n_atoms = 0
            loss_val = float('nan')
            rmsd_before = 0.0
            rmsd_after = 0.0
            force_norm = 0.0

            try:
                topo = create_topology(str(pdb_path), device='cpu')
                n_atoms = topo['charges_base'].shape[0]
            except Exception as e:
                print(f'\n  SKIP {pdb_id}: topology error: {e}')
                gc.collect()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            try:
                system, modeller = create_base_system_and_modeller(str(pdb_path))
                modify_system_forces(system, ff, topo)
            except Exception as e:
                print(f'\n  SKIP {pdb_id}: system error: {e}')
                del topo
                gc.collect()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            initial_positions = np.array(modeller.getPositions().value_in_unit(unit.nanometer))

            try:
                min_pos, forces = minimize_with_system(
                    system, modeller,
                    tolerance=config.inner_min_tolerance,
                    max_iterations=config.inner_min_max_iterations,
                    initial_positions=initial_positions,
                )
            except Exception as e:
                print(f'\n  SKIP {pdb_id}: minimization error: {e}')
                del topo, system, modeller
                gc.collect()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            orig_pos = topo['orig_pos']
            atom_idx = topo.get('orig_atom_indices')
            rmsd_before = kabsch_rmsd(torch.from_numpy(initial_positions), torch.from_numpy(orig_pos), atom_indices=atom_idx).item()
            rmsd_after = kabsch_rmsd(torch.from_numpy(min_pos), torch.from_numpy(orig_pos), atom_indices=atom_idx).item()
            force_norm = np.linalg.norm(forces)

            try:
                loss_val = compute_ift_gradient(
                    ff, min_pos, orig_pos, topo,
                    device=device,
                    cg_damping=config.cg_damping,
                    cg_fallback_damping=config.cg_fallback_damping,
                    cg_max_iter=config.cg_max_iter,
                    cg_tol=config.cg_tolerance,
                )
            except RuntimeError as e:
                if 'out of memory' not in str(e):
                    raise
                print(f'\n  SKIP {pdb_id}: CUDA out of memory')
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            if config.reg_lambda > 0:
                for p in ff.parameters():
                    if p.grad is not None:
                        p.grad.add_(config.reg_lambda * 2.0 * (p.data - 1.0))

            torch.nn.utils.clip_grad_norm_(ff.parameters(), config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss_val
            n_processed += 1

            per_prot_file.write(f'{epoch}\t{pdb_id}\t{n_atoms}\t{loss_val:.6f}\t{rmsd_before:.4f}\t{rmsd_after:.4f}\t{force_norm:.4e}\n')
            per_prot_file.flush()
            pbar.set_postfix({'loss': f'{loss_val:.6f}', 'pdb': pdb_id})

            del topo, system, modeller, min_pos, forces, orig_pos, atom_idx, initial_positions
            gc.collect()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if n_processed == 0:
            print('No proteins processed, skipping epoch.')
            continue

        avg_loss = epoch_loss / n_processed
        losses.append(avg_loss)

        for p in ff.parameters():
            p.data.clamp_(min=0.0)
        ff.bond_r0_mult.data.clamp_(min=1e-6)
        ff.angle_theta0_mult.data.clamp_(min=1e-6)
        ff.bond_k_mult.data.clamp_(min=0.0)
        ff.angle_k_mult.data.clamp_(min=0.0)

        mults = ff.get_multipliers()
        mult_history.append(mults)

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch:3d}/{config.n_epochs} | loss={avg_loss:.6f} nm | {epoch_time:.0f}s')

        epoch_xml = os.path.join(config.checkpoint_dir, f'epoch_{epoch}.xml')
        dump_xml(ff, epoch_xml)

        if epoch % config.save_interval == 0 or epoch == config.n_epochs - 1:
            ckpt_path = os.path.join(config.checkpoint_dir, f'epoch_{epoch}.pt')
            save_checkpoint(ff, epoch, losses, mult_history, ckpt_path)

    per_prot_file.close()

    final_mults = ff.get_multipliers()
    dump_xml(ff, args.output_xml)

    log_path = os.path.join(config.checkpoint_dir, 'training_log.tsv')
    log_keys = sorted(final_mults.keys())
    with open(log_path, 'w') as f:
        f.write('epoch\tavg_loss\t' + '\t'.join(log_keys) + '\n')
        for ep, (avg_l, mults) in enumerate(zip(losses, mult_history)):
            f.write(f'{ep}\t{avg_l:.8f}')
            for k in log_keys:
                f.write(f'\t{mults[k]:.6f}')
            f.write('\n')

    total_time = time.time() - total_start
    hours, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    print()
    print('=' * 60)
    print('  TRAINING SUMMARY')
    print('=' * 60)
    print(f'  PDBs:            {len(pdb_paths)} ({n_processed} processed / {len(pdb_paths) - n_processed} skipped)')
    print(f'  Device:          {device}')
    print(f'  Epochs:          {config.n_epochs}')
    print(f'  Total time:      {total_time:.1f}s ({int(hours)}h {int(mins)}m {int(secs)}s)')
    print()
    print(f'  Per-epoch loss (RMSD, nm):')
    for ep, avg_l in enumerate(losses):
        print(f'    Epoch {ep:3d}:  {avg_l:.8f}')
    print()
    print(f'  Final global multipliers:')
    for k in sorted(final_mults.keys()):
        print(f'    {k:20s} {final_mults[k]:.8f}')
    print()
    print(f'  Output:')
    print(f'    Final XML:     {args.output_xml}')
    print(f'    Training log:  {log_path}')
    print(f'    Per-protein:   {per_prot_path}')
    print(f'    Checkpoints:   {config.checkpoint_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
