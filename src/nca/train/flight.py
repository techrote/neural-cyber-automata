"""Trainer-only learner construction and fail-closed preparation status."""
from nca.adapters.flight import finite_vector, velocity_action
from nca.registry import registry
from nca.train.flight_accounting import execution_gate, preparation_declaration


def build_policy():
    """CPU model only; factory does not optimize or collect host data."""
    import torch
    return torch.nn.Sequential(torch.nn.Linear(54, 64, device='cpu'), torch.nn.Tanh(),
        torch.nn.Linear(64, 64, device='cpu'), torch.nn.Tanh(), torch.nn.Linear(64, 2, device='cpu'), torch.nn.Tanh())


def export_preprocessing(flat, outputs):
    # Host schema uses published fixed scaling; no tuning/final-fitted statistics.
    return finite_vector(flat, 54), velocity_action(outputs)


def training_admission():
    """Evaluate the immutable preparation declaration; it is deliberately blocked."""
    gate = execution_gate(preparation_declaration(), current_evidence={})
    return dict(gate, reason=';'.join(gate['reasons']))


registry.register('policy', '4ly-direct-preparation', 1, build_policy,
                  capabilities=('cpu_model_factory', 'no_execution_admission'))
