import os
import math
import torch
import xml.etree.ElementTree as ET
from openmm import app

from .topology import get_class_mappings, _DEFAULT_XML


def dump_xml(ff, xml_path, base_xml=None):
    if base_xml is None:
        base_xml = _DEFAULT_XML
    os.makedirs(os.path.dirname(os.path.abspath(xml_path)), exist_ok=True)

    tree = ET.parse(base_xml)
    root = tree.getroot()
    mappings = get_class_mappings(base_xml)
    type_name_to_class = mappings['type_number_to_class']
    class_to_idx = mappings['class_to_idx']
    bond_pair_to_idx = mappings['bond_pair_to_idx']
    angle_triple_to_idx = mappings['angle_triple_to_idx']

    with torch.no_grad():
        charge_mult = ff.charge_mult.item()
        lj_sig_mult = ff.lj_sig_mult.item()
        lj_eps_mult = ff.lj_eps_mult.item()
        class_charge_scale = ff.class_charge_scale.cpu().numpy()
        class_sigma_scale = ff.class_sigma_scale.cpu().numpy()
        class_epsilon_scale = ff.class_epsilon_scale.cpu().numpy()

        bond_k_mult = ff.bond_k_mult.item()
        bond_r0_mult = ff.bond_r0_mult.item()
        bond_class_k_scale = ff.bond_class_k_scale.cpu().numpy()
        bond_class_r0_scale = ff.bond_class_r0_scale.cpu().numpy()

        angle_k_mult = ff.angle_k_mult.item()
        angle_theta0_mult = ff.angle_theta0_mult.item()
        angle_class_k_scale = ff.angle_class_k_scale.cpu().numpy()
        angle_class_theta0_scale = ff.angle_class_theta0_scale.cpu().numpy()

    for child in root:
        if child.tag == 'HarmonicBondForce':
            for bond in child.findall('Bond'):
                c1 = bond.get('class1')
                c2 = bond.get('class2')
                length = float(bond.get('length'))
                k = float(bond.get('k'))
                if c1 and c2:
                    pair = tuple(sorted([c1, c2]))
                    bcidx = bond_pair_to_idx.get(pair, 0)
                    new_length = length * bond_r0_mult * float(bond_class_r0_scale[bcidx])
                    new_k = k * bond_k_mult * float(bond_class_k_scale[bcidx])
                    bond.set('length', f'{new_length:.10f}')
                    bond.set('k', f'{new_k:.10f}')

        elif child.tag == 'HarmonicAngleForce':
            for angle in child.findall('Angle'):
                c1 = angle.get('class1')
                c2 = angle.get('class2')
                c3 = angle.get('class3')
                theta = float(angle.get('angle'))
                k = float(angle.get('k'))
                if c1 and c2 and c3:
                    outer = sorted([c1, c3])
                    triple = (outer[0], c2, outer[1])
                    aidx = angle_triple_to_idx.get(triple, 0)
                    new_theta = theta * angle_theta0_mult * float(angle_class_theta0_scale[aidx])
                    new_k = k * angle_k_mult * float(angle_class_k_scale[aidx])
                    angle.set('angle', f'{new_theta:.10f}')
                    angle.set('k', f'{new_k:.10f}')

        elif child.tag == 'NonbondedForce':
            for atom_node in child.findall('Atom'):
                type_name = atom_node.get('type')
                charge = float(atom_node.get('charge'))
                sigma = float(atom_node.get('sigma'))
                epsilon = float(atom_node.get('epsilon'))
                if type_name and type_name in type_name_to_class:
                    class_name = type_name_to_class[type_name]
                    cidx = class_to_idx.get(class_name, 0)
                    new_charge = charge * charge_mult * float(class_charge_scale[cidx])
                    new_sigma = sigma * lj_sig_mult * float(class_sigma_scale[cidx])
                    new_epsilon = epsilon * lj_eps_mult * float(class_epsilon_scale[cidx])
                    atom_node.set('charge', f'{new_charge:.10f}')
                    atom_node.set('sigma', f'{new_sigma:.10f}')
                    atom_node.set('epsilon', f'{new_epsilon:.10f}')

    tree.write(xml_path, xml_declaration=True, encoding='UTF-8')


def load_xml(xml_path):
    return app.ForceField(xml_path)
