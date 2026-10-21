"""Record-level differential privacy via Opacus DP-SGD.

Guarantee scope
---------------
Each simulated institution runs DP-SGD (per-sample clipping + Gaussian noise,
Poisson subsampling) on its *own* records. The accountant tracks every noisy
step a client has taken across *all* rounds it participated in, so the
reported epsilon is a cumulative, record-level (example-level) guarantee for a
single record held by that institution, under add/remove-one-record adjacency.

Client datasets are disjoint, so a given patient record only ever enters one
client's mechanism; the run-level figure we report is the maximum per-client
epsilon. This is NOT client-level (institution-level) DP: it does not hide
whether an entire institution participated.
"""

from __future__ import annotations

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def compute_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """Epsilon of `steps` compositions of the subsampled Gaussian mechanism (RDP accountant)."""
    if steps <= 0:
        return 0.0
    acc = RDPAccountant()
    acc.history = [(noise_multiplier, sample_rate, steps)]
    return float(acc.get_epsilon(delta=delta))


def noise_for_target_epsilon(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """Smallest noise multiplier reaching `target_epsilon` (planning aid, computed by Opacus)."""
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon, target_delta=delta, sample_rate=sample_rate,
            steps=steps, accountant="rdp",
        )
    )


def make_private(model, optimizer, loader, noise_multiplier: float, max_grad_norm: float, secure_mode: bool):
    """Wrap model/optimizer/loader with Opacus (Poisson sampling, flat clipping)."""
    engine = PrivacyEngine(accountant="rdp", secure_mode=secure_mode)
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,
    )
    return engine, model, optimizer, loader
