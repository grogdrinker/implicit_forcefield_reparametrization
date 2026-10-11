import io
import math
import os

import torch
import numpy as np
import openmm
from openmm import app, unit
from openmm.app.internal import compiled

_PARAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_DEFAULT_XML = os.path.join(_PARAMS_DIR, 'amber99sb.xml')

_PREFERRED_TEMPLATES = {
    'CYS': ['CYS', 'CYM', 'CYX'],
    'CYM': ['CYM', 'CYS', 'CYX'],
    'CYX': ['CYX', 'CYS', 'CYM'],
}

_RESIDUE_FIXES = {
    'PTR': ('TYR', {'P', 'O1P', 'O2P', 'O3P'}),
}


def _fix_pdb_lines(pdb_lines):
    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')):
            resname = line[17:20].strip()
            if resname in _RESIDUE_FIXES:
                new_name, remove_atoms = _RESIDUE_FIXES[resname]
                atomname = line[12:16].strip()
                if atomname in remove_atoms:
                    continue
                line = line[:17] + new_name + line[20:]
        yield line


def extract_layers(pdb_path, n_layers=3):
    pdb = app.PDBFile(pdb_path)
    positions = np.array(pdb.getPositions().value_in_unit(unit.nanometer))
    ca_idx = [i for i, a in enumerate(pdb.topology.atoms()) if a.name == 'CA']
    if len(ca_idx) < 3:
        return None
    ca_pos = positions[ca_idx]
    centered = ca_pos - ca_pos.mean(axis=0)
    cov = centered.T @ centered
    _, evecs = np.linalg.eigh(cov)
    fiber_axis = evecs[:, -1]
    atoms_list = list(pdb.topology.atoms())
    chain_data = []
    for chain in pdb.topology.chains():
        chain_ca = [positions[i] for i in ca_idx if atoms_list[i].residue.chain.id == chain.id]
        if chain_ca:
            chain_data.append((np.mean(chain_ca, axis=0) @ fiber_axis, chain.id))
    chain_data.sort(key=lambda x: x[0])
    n = len(chain_data)
    if n % 10 != 0:
        return None
    chains_per_layer = n // 10
    keep = n_layers * chains_per_layer
    start = (n - keep) // 2
    selected = set(cid for _, cid in chain_data[start:start + keep])
    result = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and len(line) > 21 and line[21] in selected:
                result.append(line)
            elif line.startswith(('END', 'ENDMDL')):
                result.append(line)
    return result


def _load_pdb(pdb_path, extract_layers_flag=False):
    if extract_layers_flag:
        layers = extract_layers(pdb_path)
        if layers is None:
            return None
    else:
        with open(pdb_path) as f:
            layers = list(f)
    fixed = list(_fix_pdb_lines(layers))
    return app.PDBFile(io.StringIO(''.join(fixed)))


def create_base_system_and_modeller(pdb_path, base_xml=None, extract_layers_flag=False):
    if base_xml is None:
        base_xml = _DEFAULT_XML
    pdb = _load_pdb(pdb_path, extract_layers_flag=extract_layers_flag)
    if pdb is None:
        raise ValueError(f"Chain count not divisible by 10: {pdb_path}")
    modeller = app.Modeller(pdb.topology, pdb.getPositions())

    import random
    random.seed(42)
    modeller.addHydrogens()
    ff = app.ForceField(base_xml)
    topology = modeller.getTopology()
    bondedToAtom = ff._buildBondedToAtomList(topology)

    residueTemplates = {}
    for residue in topology.residues():
        if residue.name in ('CYS', 'CYM', 'CYX'):
            try:
                ff._getResidueTemplateMatches(residue, bondedToAtom, ignoreExternalBonds=True)
            except Exception:
                preferred = _PREFERRED_TEMPLATES.get(residue.name, [residue.name])
                for name in preferred:
                    if name in ff._templates:
                        t = ff._templates[name]
                        m = compiled.matchResidueToTemplate(residue, t, bondedToAtom, ignoreExternalBonds=True)
                        if m is not None:
                            residueTemplates[residue] = name
                            break

    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff,
                              ignoreExternalBonds=True, residueTemplates=residueTemplates)
    return system, modeller


def modify_system_forces(system, ff_pytorch, topo):
    with torch.no_grad():
        charge_mult = ff_pytorch.charge_mult.item()
        lj_sig_mult = ff_pytorch.lj_sig_mult.item()
        lj_eps_mult = ff_pytorch.lj_eps_mult.item()

        class_charge_scale = ff_pytorch.class_charge_scale.cpu().numpy()
        class_sigma_scale = ff_pytorch.class_sigma_scale.cpu().numpy()
        class_epsilon_scale = ff_pytorch.class_epsilon_scale.cpu().numpy()

        bond_k_mult = ff_pytorch.bond_k_mult.item()
        bond_r0_mult = ff_pytorch.bond_r0_mult.item()
        bond_class_k_scale = ff_pytorch.bond_class_k_scale.cpu().numpy()
        bond_class_r0_scale = ff_pytorch.bond_class_r0_scale.cpu().numpy()

        angle_k_mult = ff_pytorch.angle_k_mult.item()
        angle_theta0_mult = ff_pytorch.angle_theta0_mult.item()
        angle_class_k_scale = ff_pytorch.angle_class_k_scale.cpu().numpy()
        angle_class_theta0_scale = ff_pytorch.angle_class_theta0_scale.cpu().numpy()

    atom_class = topo['atom_class'].cpu().numpy()

    bond_i = topo['bond_i'].cpu().numpy()
    bond_j = topo['bond_j'].cpu().numpy()
    bond_class_arr = topo['bond_class'].cpu().numpy()
    bond_class_map = {}
    for b in range(len(bond_i)):
        bond_class_map[(int(bond_i[b]), int(bond_j[b]))] = bond_class_arr[b]

    angle_i = topo['angle_i'].cpu().numpy()
    angle_j = topo['angle_j'].cpu().numpy()
    angle_k = topo['angle_k'].cpu().numpy()
    angle_class_arr = topo['angle_class'].cpu().numpy()
    angle_class_map = {}
    for a in range(len(angle_i)):
        angle_class_map[(int(angle_i[a]), int(angle_j[a]), int(angle_k[a]))] = angle_class_arr[a]

    # Fallback scale factors (AMBER's standard SCEE=1.2 / SCNB=2.0), only used for
    # exceptions where the original combining-rule ratio can't be inferred (e.g. a
    # zero charge product), so behavior on amber99sb is unchanged.
    DEFAULT_COULOMB14_SCALE = 1.0 / 1.2
    DEFAULT_LJ14_SCALE = 0.5

    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        name = type(force).__name__

        if name == 'HarmonicBondForce':
            for b in range(force.getNumBonds()):
                a1, a2, length, k = force.getBondParameters(b)
                kv = k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2)
                lenv = length.value_in_unit(unit.nanometer)
                bc = bond_class_map.get((a1, a2), 0)
                new_k = kv * bond_k_mult * float(bond_class_k_scale[bc])
                new_len = lenv * bond_r0_mult * float(bond_class_r0_scale[bc])
                new_len = max(0.01, new_len)
                new_k = max(0.0, new_k)
                force.setBondParameters(b, a1, a2,
                                        new_len * unit.nanometer,
                                        new_k * unit.kilojoule_per_mole / unit.nanometer ** 2)

        elif name == 'HarmonicAngleForce':
            for a in range(force.getNumAngles()):
                a1, a2, a3, theta, k = force.getAngleParameters(a)
                kv = k.value_in_unit(unit.kilojoule_per_mole / unit.radian ** 2)
                thetav = theta.value_in_unit(unit.radian)
                ac = angle_class_map.get((a1, a2, a3), 0)
                new_k = kv * angle_k_mult * float(angle_class_k_scale[ac])
                new_theta = thetav * angle_theta0_mult * float(angle_class_theta0_scale[ac])
                new_theta = max(0.01, min(new_theta, np.pi - 0.01))
                new_k = max(0.0, new_k)
                force.setAngleParameters(a, a1, a2, a3,
                                         new_theta * unit.radian,
                                         new_k * unit.kilojoule_per_mole / unit.radian ** 2)

        elif name == 'NonbondedForce':
            orig_particle_params = []
            for i in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(i)
                qv = q.value_in_unit(unit.elementary_charge)
                sigv = sig.value_in_unit(unit.nanometer)
                epsv = eps.value_in_unit(unit.kilojoule_per_mole)
                orig_particle_params.append((qv, sigv, epsv))
                c = atom_class[i]
                new_q = qv * charge_mult * float(class_charge_scale[c])
                new_sig = sigv * lj_sig_mult * float(class_sigma_scale[c])
                new_eps = epsv * lj_eps_mult * float(class_epsilon_scale[c])
                force.setParticleParameters(i, new_q, new_sig, new_eps)

            for e in range(force.getNumExceptions()):
                p1, p2, orig_cp, orig_sig, orig_eps = force.getExceptionParameters(e)
                orig_cpv = orig_cp.value_in_unit(unit.elementary_charge ** 2)
                orig_epsv = orig_eps.value_in_unit(unit.kilojoule_per_mole)
                if abs(orig_cpv) < 1e-15 and abs(orig_epsv) < 1e-15:
                    force.setExceptionParameters(e, p1, p2, 0.0, 1.0, 0.0)
                else:
                    # Derive the actual 1-4 scale factors implied by this FF's own
                    # (unscaled) exception values, instead of assuming AMBER's fixed
                    # SCEE/SCNB constants. Falls back to those constants only when the
                    # ratio can't be inferred (e.g. a zero charge product).
                    q1_0, _, e1_0 = orig_particle_params[p1]
                    q2_0, _, e2_0 = orig_particle_params[p2]
                    combo_cp0 = q1_0 * q2_0
                    combo_eps0 = math.sqrt(max(0.0, e1_0 * e2_0))
                    coulomb_scale = (orig_cpv / combo_cp0 if abs(combo_cp0) > 1e-15
                                     else DEFAULT_COULOMB14_SCALE)
                    lj_eps_scale = (orig_epsv / combo_eps0 if combo_eps0 > 1e-15
                                    else DEFAULT_LJ14_SCALE)

                    q1, s1, e1 = force.getParticleParameters(p1)
                    q2, s2, e2 = force.getParticleParameters(p2)
                    q1v = q1.value_in_unit(unit.elementary_charge)
                    q2v = q2.value_in_unit(unit.elementary_charge)
                    s1v = s1.value_in_unit(unit.nanometer)
                    s2v = s2.value_in_unit(unit.nanometer)
                    e1v = e1.value_in_unit(unit.kilojoule_per_mole)
                    e2v = e2.value_in_unit(unit.kilojoule_per_mole)
                    new_cp = q1v * q2v * coulomb_scale
                    new_sig = (s1v + s2v) / 2.0
                    new_eps = math.sqrt(max(1e-30, e1v * e2v)) * lj_eps_scale
                    force.setExceptionParameters(e, p1, p2, new_cp, new_sig, new_eps)

        elif name == 'PeriodicTorsionForce':
            pass


def minimize_with_system(system, modeller, tolerance=0.001, max_iterations=100000, initial_positions=None,
                          return_energy=False):
    integrator = openmm.LangevinIntegrator(300, 1, 0.001)
    context = openmm.Context(system, integrator)
    if initial_positions is not None:
        context.setPositions(initial_positions * unit.nanometer)
    else:
        context.setPositions(modeller.getPositions())

    prev_rms = float('inf')
    for _ in range(5):
        openmm.LocalEnergyMinimizer.minimize(context, tolerance=tolerance, maxIterations=max_iterations)
        state = context.getState(getForces=True)
        forces = np.array(state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
        rms = np.sqrt(np.mean(forces ** 2))
        if prev_rms - rms < 0.001 * prev_rms:
            break
        prev_rms = rms

    state = context.getState(getPositions=True, getForces=True, getEnergy=return_energy)
    minimized_pos = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    forces = np.array(state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
    if return_energy:
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        del context, integrator
        return minimized_pos, forces, energy
    del context, integrator
    return minimized_pos, forces


def minimize_with_pytorch(ff_pytorch, topo, modeller, tolerance=0.001, max_iterations=1000, device='cuda', initial_positions=None):
    from openmm import unit as _unit
    if initial_positions is not None:
        init_pos = np.array(initial_positions)
    else:
        init_pos = np.array(modeller.getPositions().value_in_unit(_unit.nanometer))
    pos = torch.from_numpy(init_pos).to(device).requires_grad_(True)

    topo_device = {}
    for k, v in topo.items():
        if isinstance(v, torch.Tensor):
            topo_device[k] = v.to(device)
        elif k == '_openmm_topology':
            continue
        else:
            topo_device[k] = v

    grad_norm_threshold = tolerance * (pos.shape[0] * 3) ** 0.5

    def closure():
        optimizer.zero_grad()
        energy = ff_pytorch._compute_energy(pos, topo_device)
        energy.backward()
        torch.nn.utils.clip_grad_norm_([pos], 1000)
        return energy

    optimizer = torch.optim.LBFGS(
        [pos],
        max_iter=max_iterations,
        tolerance_grad=grad_norm_threshold,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn='strong_wolfe',
    )

    optimizer.step(closure)

    energy_post = ff_pytorch._compute_energy(pos, topo_device)
    force_tensor = -torch.autograd.grad(energy_post, pos, create_graph=False)[0]

    minimized_pos = pos.detach().cpu().numpy()
    forces = force_tensor.detach().cpu().numpy()
    return minimized_pos, forces


def _best_platform():
    for name in ['CUDA', 'OpenCL', 'CPU']:
        try:
            return openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return openmm.Platform.getPlatformByName('Reference')

def run_base_md_perturbation(system, positions, n_steps=20, temperature=300.0,
                              friction=1.0, step_size=0.001, platform=None, seed=42):
    integrator = openmm.LangevinIntegrator(temperature, friction, step_size)
    integrator.setRandomNumberSeed(seed)
    platform_obj = _best_platform() if platform is None else openmm.Platform.getPlatformByName(platform)
    context = openmm.Context(system, integrator, platform_obj)
    context.setPositions(positions * unit.nanometer)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(n_steps)
    state = context.getState(getPositions=True)
    new_positions = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    del context, integrator
    return new_positions


def compute_energy_with_system(system, positions):
    integrator = openmm.LangevinIntegrator(300, 1, 0.001)
    context = openmm.Context(system, integrator)
    context.setPositions(positions * unit.nanometer)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context, integrator
    return energy
