import sys
import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    n_epochs: int = 50
    outer_lr: float = 0.001
    device: str = 'auto'
    max_atoms: int = sys.maxsize
    inner_min_tolerance: float = 0.001
    inner_min_max_iterations: int = 100000
    use_pytorch_minimizer: bool = False
    cg_max_iter: int = 200
    cg_damping: float = 1.0
    cg_fallback_damping: float = 10.0
    cg_tolerance: float = 1e-8
    gradient_clip_norm: float = 1.0
    reg_lambda: float = 0.0
    checkpoint_dir: str = 'checkpoints'
    save_interval: int = 10
    seed: int = 42


@dataclass
class SimulationConfig:
    temperature: float = 300.0
    friction: float = 1.0
    timestep: float = 0.002
    n_steps: int = 250000
    traj_interval: int = 10000
    min_tolerance: float = 1.0
    min_max_iterations: int = 1000
    output_dir: str = './simulation_output'
    seed: int = 42
