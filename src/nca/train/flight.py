"""Trainer-only learner construction and fail-closed preparation status."""
from nca.adapters.flight import finite_vector, velocity_action
from nca.registry import registry


def build_policy():
    """CPU model only; factory does not optimize or collect host data."""
    import torch
    return torch.nn.Sequential(torch.nn.Linear(54, 64, device='cpu'), torch.nn.Tanh(),
        torch.nn.Linear(64, 64, device='cpu'), torch.nn.Tanh(), torch.nn.Linear(64, 2, device='cpu'), torch.nn.Tanh())


def export_preprocessing(flat, outputs):
    # Host schema uses published fixed scaling; no tuning/final-fitted statistics.
    return finite_vector(flat, 54), velocity_action(outputs)


def training_admission():
    """No user-controlled flag can turn an unfinished dependency into admission."""
    return {'admitted': False, 'reason': '4ly003_accepted_host_and_teacher_required',
            'final_access': False, 'training_enabled': False}


registry.register('policy', '4ly-direct-preparation', 1, build_policy,
                  capabilities=('cpu_model_factory', 'no_execution_admission'))
