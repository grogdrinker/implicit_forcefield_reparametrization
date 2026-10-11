import gc

import torch

from .utils import kabsch_rmsd


def cg_solve(hvp, b, max_iter=100, tol=1e-8, damping=1000.0, return_info=False):
    b = b.detach()
    b_norm = b.norm()
    if b_norm < 1e-30:
        return (torch.zeros_like(b), 0.0) if return_info else torch.zeros_like(b)

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = r @ r
    rs_new = rs_old

    for _ in range(max_iter):
        Ap = hvp(p).detach() + damping * p
        pAp = p @ Ap
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    rel_res = rs_new.sqrt() / b_norm
    if return_info:
        return x.detach(), rel_res.item()
    return x.detach()


def compute_ift_gradient(ff_pytorch, min_pos, orig_pos, topo, device='cuda',
                         cg_damping=1.0, cg_fallback_damping=10.0,
                         cg_max_iter=200, cg_tol=1e-8):
    param_list = list(ff_pytorch.parameters())

    orig_tensor = torch.from_numpy(orig_pos).to(device)
    min_tensor = torch.from_numpy(min_pos).to(device).requires_grad_(True)

    topo_device = {}
    for k, v in topo.items():
        if isinstance(v, torch.Tensor):
            topo_device[k] = v.to(device)
        else:
            topo_device[k] = v

    atom_indices = topo.get('orig_atom_indices')
    if atom_indices is not None:
        atom_indices = atom_indices.to(device)
    rmsd = kabsch_rmsd(min_tensor, orig_tensor, atom_indices=atom_indices)
    loss_val = rmsd.item()

    energy = ff_pytorch._compute_energy(min_tensor, topo_device)

    grad_x = torch.autograd.grad(energy, min_tensor, create_graph=True, retain_graph=True)[0]

    def hvp(v_flat):
        v_shaped = v_flat.detach().view_as(grad_x)
        Hv = torch.autograd.grad(grad_x, min_tensor, grad_outputs=v_shaped,
                                 retain_graph=True, create_graph=False)[0]
        return Hv.detach().view(-1)

    dL_dx = torch.autograd.grad(rmsd, min_tensor)[0].view(-1)

    v_flat, rel_res = cg_solve(hvp, dL_dx, damping=cg_damping, max_iter=cg_max_iter,
                               tol=cg_tol, return_info=True)
    if rel_res > 0.01:
        v_flat, rel_res = cg_solve(hvp, dL_dx, damping=cg_fallback_damping, max_iter=cg_max_iter,
                                   tol=cg_tol, return_info=True)

    param_grads = torch.autograd.grad(
        grad_x, param_list,
        grad_outputs=v_flat.view_as(grad_x),
        retain_graph=False, allow_unused=True
    )

    del grad_x, energy
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for p, g in zip(param_list, param_grads):
        if g is not None:
            g_val = g.detach()
            if torch.isnan(g_val).any():
                continue
            if p.grad is None:
                p.grad = -g_val
            else:
                p.grad = p.grad + (-g_val)

    del hvp, param_grads, v_flat, dL_dx
    del min_tensor, orig_tensor, topo_device
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return loss_val
