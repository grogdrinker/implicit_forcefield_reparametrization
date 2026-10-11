import io
import os
import random
import torch
import numpy as np
import xml.etree.ElementTree as ET
from openmm import app, unit
from openmm.app.internal import compiled

_PARAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_DEFAULT_XML = os.path.join(_PARAMS_DIR, 'amber99sb.xml')

_CLASS_MAPPINGS = {}

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


def _resolve_template(ff, residue, bondedToAtom, ignoreExternalBonds=True):
    try:
        result = ff._getResidueTemplateMatches(residue, bondedToAtom, ignoreExternalBonds=ignoreExternalBonds)
        return result[0], result[1]
    except Exception:
        all_templates = ff._templates
        res_name = residue.name
        preferred = _PREFERRED_TEMPLATES.get(res_name, [res_name])
        for templ_name in preferred:
            if templ_name in all_templates:
                template = all_templates[templ_name]
                match = compiled.matchResidueToTemplate(
                    residue, template, bondedToAtom,
                    ignoreExternalBonds=ignoreExternalBonds
                )
                if match is not None:
                    return template, match
        raise


def get_class_mappings(xml_path=None):
    global _CLASS_MAPPINGS
    if xml_path is None:
        xml_path = _DEFAULT_XML
    if xml_path in _CLASS_MAPPINGS:
        return _CLASS_MAPPINGS[xml_path]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    mappings = {
        'type_number_to_class': {},
        'class_to_idx': {},
        'bond_pair_to_idx': {},
        'angle_triple_to_idx': {},
        'dih_quad_to_idx': {},
    }

    type_name_to_class = {}
    for t in root.find('AtomTypes').findall('Type'):
        type_name = t.get('name')
        atom_class = t.get('class')
        type_name_to_class[type_name] = atom_class

    all_classes = sorted(set(type_name_to_class.values()))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class_name = {i: c for c, i in class_to_idx.items()}

    mappings['type_number_to_class'] = type_name_to_class
    mappings['class_to_idx'] = class_to_idx
    mappings['idx_to_class_name'] = idx_to_class_name

    def _resolve_to_class(name):
        if name in type_name_to_class:
            return type_name_to_class[name]
        return name

    bond_force = root.find('HarmonicBondForce')
    if bond_force is not None:
        for bond in bond_force.findall('Bond'):
            raw_c1 = bond.get('class1') or bond.get('type1')
            raw_c2 = bond.get('class2') or bond.get('type2')
            if raw_c1 and raw_c2:
                c1 = _resolve_to_class(raw_c1)
                c2 = _resolve_to_class(raw_c2)
                pair = tuple(sorted([c1, c2]))
                if pair not in mappings['bond_pair_to_idx']:
                    mappings['bond_pair_to_idx'][pair] = len(mappings['bond_pair_to_idx'])

    angle_force = root.find('HarmonicAngleForce')
    if angle_force is not None:
        for angle in angle_force.findall('Angle'):
            raw_c1 = angle.get('class1') or angle.get('type1')
            raw_c2 = angle.get('class2') or angle.get('type2')
            raw_c3 = angle.get('class3') or angle.get('type3')
            if raw_c1 and raw_c2 and raw_c3:
                c1 = _resolve_to_class(raw_c1)
                c2 = _resolve_to_class(raw_c2)
                c3 = _resolve_to_class(raw_c3)
                outer = sorted([c1, c3])
                triple = (outer[0], c2, outer[1])
                if triple not in mappings['angle_triple_to_idx']:
                    idx = len(mappings['angle_triple_to_idx'])
                    mappings['angle_triple_to_idx'][triple] = idx

    dih_force = root.find('PeriodicTorsionForce')
    if dih_force is not None:
        for dih in dih_force.findall('Proper'):
            raw_c1 = dih.get('class1') or dih.get('type1')
            raw_c2 = dih.get('class2') or dih.get('type2')
            raw_c3 = dih.get('class3') or dih.get('type3')
            raw_c4 = dih.get('class4') or dih.get('type4')
            if raw_c1 and raw_c2 and raw_c3 and raw_c4:
                c1 = _resolve_to_class(raw_c1)
                c2 = _resolve_to_class(raw_c2)
                c3 = _resolve_to_class(raw_c3)
                c4 = _resolve_to_class(raw_c4)
                quad = (c1, c2, c3, c4)
                if quad not in mappings['dih_quad_to_idx']:
                    idx = len(mappings['dih_quad_to_idx'])
                    mappings['dih_quad_to_idx'][quad] = idx

    _CLASS_MAPPINGS[xml_path] = mappings
    return mappings


def create_topology(pdb_path, xml_path=None, device='cpu', extract_layers_flag=False):
    mappings = get_class_mappings(xml_path)
    type_name_to_class = mappings['type_number_to_class']
    class_to_idx = mappings['class_to_idx']
    bond_pair_to_idx = mappings['bond_pair_to_idx']
    angle_triple_to_idx = mappings['angle_triple_to_idx']
    dih_quad_to_idx = mappings['dih_quad_to_idx']

    pdb = _load_pdb(pdb_path, extract_layers_flag=extract_layers_flag)
    if pdb is None:
        raise ValueError(f"Chain count not divisible by 10: {pdb_path}")

    orig_atom_keys = set()
    for atom in pdb.topology.atoms():
        orig_atom_keys.add((atom.residue.chain.id, atom.residue.id, atom.name))

    modeller = app.Modeller(pdb.topology, pdb.getPositions())

    random.seed(42)
    modeller.addHydrogens()

    # RMSD is measured over the deposited HEAVY atoms only -- hydrogens are never
    # included (they are absent from most deposits and added inconsistently, and
    # their positions are not experimentally reliable).
    orig_atom_indices = []
    for i, atom in enumerate(modeller.getTopology().atoms()):
        key = (atom.residue.chain.id, atom.residue.id, atom.name)
        if key in orig_atom_keys and (atom.element is None or atom.element.symbol != 'H'):
            orig_atom_indices.append(i)

    n_original = len(orig_atom_indices)

    ff_base = app.ForceField(xml_path or _DEFAULT_XML)
    topology = modeller.getTopology()
    bondedToAtom = ff_base._buildBondedToAtomList(topology)

    residueTemplates = {}
    for residue in topology.residues():
        if residue.name in ('CYS', 'CYM', 'CYX'):
            preferred = _PREFERRED_TEMPLATES.get(residue.name, [residue.name])
            for name in preferred:
                if name in ff_base._templates:
                    t = ff_base._templates[name]
                    m = compiled.matchResidueToTemplate(residue, t, bondedToAtom, ignoreExternalBonds=True)
                    if m is not None:
                        residueTemplates[residue] = name
                        break

    system = ff_base.createSystem(topology, nonbondedMethod=app.NoCutoff,
                                   ignoreExternalBonds=True, residueTemplates=residueTemplates)

    atoms_list = list(topology.atoms())
    atom_idx_to_class = {}

    for residue in topology.residues():
        try:
            template_data, atom_matches = _resolve_template(
                ff_base, residue, bondedToAtom, ignoreExternalBonds=True
            )
            for atom_idx_in_res, atom in enumerate(residue.atoms()):
                t_idx = atom_matches[atom_idx_in_res]
                type_number = str(template_data.atoms[t_idx].type)
                atom_class = type_name_to_class.get(type_number, 'C')
                atom_idx_to_class[atom.index] = class_to_idx.get(atom_class, 0)
        except Exception:
            for atom in residue.atoms():
                atom_idx_to_class[atom.index] = 0

    atom_class_idx_to_name = {v: k for k, v in class_to_idx.items()}
    n_atoms = len(atoms_list)

    bond_i, bond_j, bond_k, bond_r0, bond_class = [], [], [], [], []
    angle_i, angle_j, angle_k, angle_k_val, angle_theta0, angle_class = [], [], [], [], [], []
    dih_i, dih_j, dih_k, dih_l, dih_k_val, dih_phi0, dih_per, dih_class = [], [], [], [], [], [], [], []
    charges, sigmas, epsilons = [], [], []
    exc_i, exc_j, exc_cp, exc_sig, exc_eps = [], [], [], [], []

    for force in system.getForces():
        name = type(force).__name__

        if name == 'HarmonicBondForce':
            for idx in range(force.getNumBonds()):
                a1, a2, r0, k = force.getBondParameters(idx)
                bond_i.append(a1)
                bond_j.append(a2)
                bond_k.append(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2))
                bond_r0.append(r0.value_in_unit(unit.nanometer))

                c1_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a1, 0), 'C')
                c2_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a2, 0), 'C')
                pair = tuple(sorted([c1_name, c2_name]))
                bond_class.append(bond_pair_to_idx.get(pair, 0))

        elif name == 'HarmonicAngleForce':
            for idx in range(force.getNumAngles()):
                a1, a2, a3, theta0, k = force.getAngleParameters(idx)
                angle_i.append(a1)
                angle_j.append(a2)
                angle_k.append(a3)
                angle_k_val.append(k.value_in_unit(unit.kilojoule_per_mole / unit.radian ** 2))
                angle_theta0.append(theta0.value_in_unit(unit.radian))

                c1_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a1, 0), 'C')
                c2_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a2, 0), 'C')
                c3_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a3, 0), 'C')
                outer = sorted([c1_name, c3_name])
                triple = (outer[0], c2_name, outer[1])
                angle_class.append(angle_triple_to_idx.get(triple, 0))

        elif name == 'PeriodicTorsionForce':
            for idx in range(force.getNumTorsions()):
                a1, a2, a3, a4, per, phase, k = force.getTorsionParameters(idx)
                dih_i.append(a1)
                dih_j.append(a2)
                dih_k.append(a3)
                dih_l.append(a4)
                dih_k_val.append(k.value_in_unit(unit.kilojoule_per_mole))
                dih_phi0.append(phase.value_in_unit(unit.radian))
                dih_per.append(float(per))

                c1_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a1, 0), 'C')
                c2_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a2, 0), 'C')
                c3_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a3, 0), 'C')
                c4_name = atom_class_idx_to_name.get(atom_idx_to_class.get(a4, 0), 'C')
                quad = (c1_name, c2_name, c3_name, c4_name)
                dih_class.append(dih_quad_to_idx.get(quad, 0))

        elif name == 'NonbondedForce':
            for idx in range(force.getNumParticles()):
                q, s, e = force.getParticleParameters(idx)
                charges.append(q.value_in_unit(unit.elementary_charge))
                sigmas.append(s.value_in_unit(unit.nanometer))
                epsilons.append(e.value_in_unit(unit.kilojoule_per_mole))

            for idx in range(force.getNumExceptions()):
                p1, p2, cp, s, e = force.getExceptionParameters(idx)
                exc_i.append(p1)
                exc_j.append(p2)
                exc_cp.append(cp.value_in_unit(unit.elementary_charge ** 2))
                exc_sig.append(s.value_in_unit(unit.nanometer))
                exc_eps.append(e.value_in_unit(unit.kilojoule_per_mole))

    atom_class_array = [atom_idx_to_class.get(i, 0) for i in range(n_atoms)]

    i_idx, j_idx = torch.triu_indices(n_atoms, n_atoms, offset=1)

    exc_set = set()
    exc_14_i, exc_14_j, exc_14_cp, exc_14_sig, exc_14_eps = [], [], [], [], []
    exc_14_class_i, exc_14_class_j = [], []
    exc_14_set = set()

    for i, j, cp, s, e in zip(exc_i, exc_j, exc_cp, exc_sig, exc_eps):
        e_val = float(e) if hasattr(e, '__float__') else e
        cp_val = float(cp) if hasattr(cp, '__float__') else cp
        if e_val > 1e-10 or abs(cp_val) > 1e-10:
            exc_14_i.append(i)
            exc_14_j.append(j)
            exc_14_cp.append(cp_val)
            exc_14_sig.append(float(s) if hasattr(s, '__float__') else s)
            exc_14_eps.append(e_val)
            exc_14_class_i.append(atom_idx_to_class.get(i, 0))
            exc_14_class_j.append(atom_idx_to_class.get(j, 0))
            exc_14_set.add((min(i, j), max(i, j)))
        else:
            exc_set.add((min(i, j), max(i, j)))

    mask = torch.tensor([(int(i), int(j)) not in exc_set and (int(i), int(j)) not in exc_14_set
                         for i, j in zip(i_idx.tolist(), j_idx.tolist())], dtype=torch.bool)
    nonbond_i = i_idx[mask]
    nonbond_j = j_idx[mask]

    topo = {
        'bond_i': torch.LongTensor(bond_i),
        'bond_j': torch.LongTensor(bond_j),
        'bond_k_base': torch.tensor(bond_k, dtype=torch.float32),
        'bond_r0_base': torch.tensor(bond_r0, dtype=torch.float32),
        'bond_class': torch.LongTensor(bond_class),

        'angle_i': torch.LongTensor(angle_i),
        'angle_j': torch.LongTensor(angle_j),
        'angle_k': torch.LongTensor(angle_k),
        'angle_k_base': torch.tensor(angle_k_val, dtype=torch.float32),
        'angle_theta0_base': torch.tensor(angle_theta0, dtype=torch.float32),
        'angle_class': torch.LongTensor(angle_class),

        'dih_i': torch.LongTensor(dih_i),
        'dih_j': torch.LongTensor(dih_j),
        'dih_k': torch.LongTensor(dih_k),
        'dih_l': torch.LongTensor(dih_l),
        'dih_k_base': torch.tensor(dih_k_val, dtype=torch.float32),
        'dih_phi0_base': torch.tensor(dih_phi0, dtype=torch.float32),
        'dih_periodicity': torch.tensor(dih_per, dtype=torch.float32),
        'dih_class': torch.LongTensor(dih_class),

        'atom_class': torch.LongTensor(atom_class_array),
        'charges_base': torch.tensor(charges, dtype=torch.float32),
        'lj_sigma_base': torch.tensor(sigmas, dtype=torch.float32),
        'lj_epsilon_base': torch.tensor(epsilons, dtype=torch.float32),

        'nonbond_i': nonbond_i,
        'nonbond_j': nonbond_j,

        'exc_14_i': torch.LongTensor(exc_14_i),
        'exc_14_j': torch.LongTensor(exc_14_j),
        'exc_14_cp': torch.tensor(exc_14_cp, dtype=torch.float32),
        'exc_14_sig': torch.tensor(exc_14_sig, dtype=torch.float32),
        'exc_14_eps': torch.tensor(exc_14_eps, dtype=torch.float32),
        'exc_14_class_i': torch.LongTensor(exc_14_class_i),
        'exc_14_class_j': torch.LongTensor(exc_14_class_j),
    }

    for k, v in topo.items():
        if isinstance(v, torch.Tensor):
            topo[k] = v.to(device)

    topo['orig_pos'] = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    topo['n_original'] = n_original
    topo['orig_atom_indices'] = torch.tensor(orig_atom_indices, dtype=torch.long)
    topo['_openmm_topology'] = modeller.getTopology()

    return topo
