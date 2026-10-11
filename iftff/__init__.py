import os as _os

# Directory holding the bundled force-field XMLs (base AMBER + the reparametrised
# force fields this program produces). Shipped as package data, so it resolves
# correctly whether the package is installed normally or in editable mode.
DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'data')

# The reparametrised force fields -- the main outputs of the method. Keys are the
# regularization condition; values are the bundled XML filenames.
REPARAMETRISED_FORCEFIELDS = {
    'unregularized': 'AmberReparametrised_unregularized.xml',
    'lambda0.005':   'AmberReparametrised_lambda0005.xml',
    'lambda0.01':    'AmberReparametrised_lambda001.xml',
    'lambda0.05':    'AmberReparametrised_lambda005.xml',
    'lambda0.1':     'AmberReparametrised_lambda01.xml',
}


def forcefield(name='amber99sb'):
    """Absolute path to a bundled force-field XML.

    `name` may be a filename ('amber99sb.xml'), a bare name ('amber99sb'), or a
    reparametrised-condition key ('unregularized', 'lambda0.01', ...).
    """
    if name in REPARAMETRISED_FORCEFIELDS:
        name = REPARAMETRISED_FORCEFIELDS[name]
    if not name.endswith('.xml'):
        name += '.xml'
    return _os.path.join(DATA_DIR, name)


# The default force field for RUNNING MD / minimization: the reg=0.01
# reparametrised force field (the recommended production force field). This is
# NOT the base used for training -- training always starts from stock amber99sb
# (see _DEFAULT_XML), which is the parameter set being reparametrised.
DEFAULT_FORCEFIELD = forcefield('lambda0.01')


from .src.config import TrainingConfig, SimulationConfig
from .src.force_field import AmberForceFieldMP
from .src.topology import create_topology, get_class_mappings, _DEFAULT_XML
from .src.openmm_utils import (
    create_base_system_and_modeller,
    modify_system_forces,
    minimize_with_system,
    minimize_with_pytorch,
    run_base_md_perturbation,
)
from .src.ift import compute_ift_gradient
from .src.utils import kabsch_rmsd, save_checkpoint, load_checkpoint
from .src.xml_io import dump_xml, load_xml

__all__ = [
    'DATA_DIR', 'REPARAMETRISED_FORCEFIELDS', 'forcefield', 'DEFAULT_FORCEFIELD',
    'TrainingConfig', 'SimulationConfig',
    'AmberForceFieldMP',
    'create_topology', 'get_class_mappings', '_DEFAULT_XML',
    'create_base_system_and_modeller', 'modify_system_forces',
    'minimize_with_system', 'minimize_with_pytorch',
    'run_base_md_perturbation',
    'compute_ift_gradient',
    'kabsch_rmsd', 'save_checkpoint', 'load_checkpoint',
    'dump_xml', 'load_xml',
]
