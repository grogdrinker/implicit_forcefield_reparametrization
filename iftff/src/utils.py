import torch
import os


def kabsch_rmsd(P, Q, atom_indices=None):
    """RMSD between P and Q after optimal rigid-body (rotation + translation) superposition."""
    if not isinstance(P, torch.Tensor):
        P = torch.from_numpy(P)
    if not isinstance(Q, torch.Tensor):
        Q = torch.from_numpy(Q)
    P = P.float()
    Q = Q.float()
    if atom_indices is not None:
        P = P[atom_indices]
        Q = Q[atom_indices]

    P_centered = P - P.mean(dim=0, keepdim=True)
    Q_centered = Q - Q.mean(dim=0, keepdim=True)

    H = P_centered.transpose(0, 1) @ Q_centered
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(0, 1) @ U.transpose(0, 1)))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vt.transpose(0, 1) @ D @ U.transpose(0, 1)

    P_aligned = P_centered @ R.transpose(0, 1)
    # Standard RMSD: mean of per-atom squared distances (sum over x,y,z per atom,
    # then average over the N atoms), not the mean over all 3N coordinate
    # components -- the latter underestimates the RMSD by a factor of sqrt(3).
    return torch.sqrt(((P_aligned - Q_centered) ** 2).sum(dim=1).mean())


def save_checkpoint(ff_pytorch, epoch, losses, mult_history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': ff_pytorch.state_dict(),
        'losses': losses,
        'mult_history': mult_history,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, ff_pytorch, device='cuda'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    ff_pytorch.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def save_training_log(path, epoch, loss, multipliers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(f'{epoch},{loss:.8f}')
        for k, v in multipliers.items():
            f.write(f',{v:.8f}')
        f.write('\n')


def init_training_log(path, mult_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('epoch,loss,' + ','.join(mult_names) + '\n')
