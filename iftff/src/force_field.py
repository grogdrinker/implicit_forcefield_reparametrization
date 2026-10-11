import torch
import torch.nn as nn

from .topology import get_class_mappings


class AmberForceFieldMP(nn.Module):
    COULOMB = 138.935458

    def __init__(self, xml_path=None):
        super().__init__()

        mappings = get_class_mappings(xml_path)
        class_to_idx = mappings['class_to_idx']
        bond_pair_to_idx = mappings['bond_pair_to_idx']
        angle_triple_to_idx = mappings['angle_triple_to_idx']

        n_atom_classes = len(class_to_idx)
        n_bond_pairs = len(bond_pair_to_idx)
        n_angle_triples = len(angle_triple_to_idx)

        self.bond_k_mult = nn.Parameter(torch.tensor(1.0))
        self.bond_r0_mult = nn.Parameter(torch.tensor(1.0))
        self.angle_k_mult = nn.Parameter(torch.tensor(1.0))
        self.angle_theta0_mult = nn.Parameter(torch.tensor(1.0))
        self.lj_eps_mult = nn.Parameter(torch.tensor(1.0))
        self.lj_sig_mult = nn.Parameter(torch.tensor(1.0))
        self.charge_mult = nn.Parameter(torch.tensor(1.0))

        self.class_charge_scale = nn.Parameter(torch.ones(n_atom_classes))
        self.class_sigma_scale = nn.Parameter(torch.ones(n_atom_classes))
        self.class_epsilon_scale = nn.Parameter(torch.ones(n_atom_classes))

        self.bond_class_k_scale = nn.Parameter(torch.ones(n_bond_pairs))
        self.bond_class_r0_scale = nn.Parameter(torch.ones(n_bond_pairs))

        self.angle_class_k_scale = nn.Parameter(torch.ones(n_angle_triples))
        self.angle_class_theta0_scale = nn.Parameter(torch.ones(n_angle_triples))

    def forward(self, positions, topo):
        return self._compute_energy(positions, topo)

    def _compute_energy(self, pos, topo, device=None):
        if isinstance(pos, torch.Tensor):
            target_device = pos.device if device is None else torch.device(device)
            pos = pos.to(target_device)
        else:
            target_device = torch.device(device or 'cuda')

        topo = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in topo.items()}

        bi, bj = topo['bond_i'], topo['bond_j']
        bk = topo['bond_k_base'] * self.bond_k_mult * self.bond_class_k_scale[topo['bond_class']]
        br0 = topo['bond_r0_base'] * self.bond_r0_mult * self.bond_class_r0_scale[topo['bond_class']]
        r = torch.norm(pos[bi] - pos[bj], dim=1)
        energy_bond = torch.sum(0.5 * bk * (r - br0) ** 2)

        ai, aj, ak = topo['angle_i'], topo['angle_j'], topo['angle_k']
        ak_base = topo['angle_k_base'] * self.angle_k_mult * self.angle_class_k_scale[topo['angle_class']]
        ath0 = topo['angle_theta0_base'] * self.angle_theta0_mult * self.angle_class_theta0_scale[topo['angle_class']]

        v1 = pos[ai] - pos[aj]
        v2 = pos[ak] - pos[aj]
        cos_theta = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8)
        theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6))
        energy_angle = torch.sum(0.5 * ak_base * (theta - ath0) ** 2)

        di, dj, dk, dl = topo['dih_i'], topo['dih_j'], topo['dih_k'], topo['dih_l']
        dk_base = topo['dih_k_base']
        dphi0 = topo['dih_phi0_base']

        b1 = pos[dj] - pos[di]
        b2 = pos[dk] - pos[dj]
        b3 = pos[dl] - pos[dk]
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        n1 = n1 / (torch.norm(n1, dim=1, keepdim=True) + 1e-8)
        n2 = n2 / (torch.norm(n2, dim=1, keepdim=True) + 1e-8)
        x = torch.sum(n1 * n2, dim=1)
        b2_norm = torch.norm(b2, dim=1, keepdim=True)
        y = torch.sum(torch.linalg.cross(n1, b2 / (b2_norm + 1e-8)) * n2, dim=1)
        phi = torch.atan2(y, x)
        energy_dih = torch.sum(dk_base * (1 + torch.cos(topo['dih_periodicity'] * phi - dphi0)))

        atom_class = topo['atom_class']
        q = topo['charges_base'] * self.charge_mult * self.class_charge_scale[atom_class]
        sig = topo['lj_sigma_base'] * self.lj_sig_mult * self.class_sigma_scale[atom_class]
        eps = topo['lj_epsilon_base'] * self.lj_eps_mult * self.class_epsilon_scale[atom_class]

        ni, nj = topo['nonbond_i'], topo['nonbond_j']
        r_nb = torch.norm(pos[ni] - pos[nj], dim=1) + 1e-8
        sig_ij = (sig[ni] + sig[nj]) / 2
        eps_ij = torch.sqrt(eps[ni] * eps[nj] + 1e-8)
        q_ij = q[ni] * q[nj]
        energy_nonbond = torch.sum(self.COULOMB * q_ij / r_nb
                                   + 4 * eps_ij * ((sig_ij / r_nb) ** 12 - (sig_ij / r_nb) ** 6))

        ei, ej = topo['exc_14_i'], topo['exc_14_j']

        if ei.shape[0] > 0:
            ci_14 = topo['exc_14_class_i']
            cj_14 = topo['exc_14_class_j']

            q14_scale = (self.charge_mult ** 2) * self.class_charge_scale[ci_14] * self.class_charge_scale[cj_14]
            eps14_scale = self.lj_eps_mult * torch.sqrt(self.class_epsilon_scale[ci_14]
                                                         * self.class_epsilon_scale[cj_14] + 1e-8)

            r_14 = torch.norm(pos[ei] - pos[ej], dim=1) + 1e-8
            q_14 = topo['exc_14_cp'] * q14_scale
            sig_i = self.lj_sig_mult * self.class_sigma_scale[ci_14] * topo['lj_sigma_base'][ei]
            sig_j = self.lj_sig_mult * self.class_sigma_scale[cj_14] * topo['lj_sigma_base'][ej]
            sig_14 = (sig_i + sig_j) / 2
            eps_14 = topo['exc_14_eps'] * eps14_scale
            energy_14 = torch.sum(self.COULOMB * q_14 / r_14
                                  + 4 * eps_14 * ((sig_14 / r_14) ** 12 - (sig_14 / r_14) ** 6))
        else:
            energy_14 = torch.tensor(0.0, device=target_device)

        return energy_bond + energy_angle + energy_dih + energy_nonbond + energy_14

    def get_multipliers(self):
        return {
            'bond_k_mult': self.bond_k_mult.item(),
            'bond_r0_mult': self.bond_r0_mult.item(),
            'angle_k_mult': self.angle_k_mult.item(),
            'angle_theta0_mult': self.angle_theta0_mult.item(),
            'lj_eps_mult': self.lj_eps_mult.item(),
            'lj_sig_mult': self.lj_sig_mult.item(),
            'charge_mult': self.charge_mult.item(),
        }
