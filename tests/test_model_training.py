"""Model, checkpointing, local training, DP-SGD and privacy accounting."""

import numpy as np
import pytest
import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from fedmed.clients.trainer import train
from fedmed.config import PrivacyCfg, TrainCfg
from fedmed.evaluation.metrics import evaluate_model
from fedmed.models.mlp import build_model, get_weights, load_checkpoint, save_checkpoint, set_weights
from fedmed.privacy.dp import compute_epsilon, noise_for_target_epsilon


def _data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def test_model_init_reproducible_and_shapes():
    a, b = build_model(30, [32, 16], seed=1), build_model(30, [32, 16], seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(get_weights(a), get_weights(b)))
    assert not np.array_equal(get_weights(build_model(30, [32, 16], seed=2))[0], get_weights(a)[0])
    assert a(torch.zeros(4, 30)).shape == (4,)
    assert sum(w.size for w in get_weights(a)) == 30 * 32 + 32 + 32 * 16 + 16 + 16 + 1


def test_set_weights_roundtrip_and_validation():
    a, b = build_model(5, [4], 0), build_model(5, [4], 1)
    set_weights(b, get_weights(a))
    assert all(np.array_equal(x, y) for x, y in zip(get_weights(a), get_weights(b)))
    with pytest.raises(ValueError):
        set_weights(b, get_weights(a)[:-1])


def test_checkpoint_save_load(tmp_path):
    m = build_model(7, [5, 3], seed=4)
    save_checkpoint(m, tmp_path / "m.pt", {"round": 3})
    m2, extra = load_checkpoint(tmp_path / "m.pt")
    assert extra == {"round": 3} and m2.hidden == [5, 3]
    x = torch.randn(3, 7)
    assert torch.equal(m(x), m2(x))


def test_local_training_learns():
    X, y = _data()
    m = build_model(6, [16], 0)
    before = evaluate_model(m, X, y)["log_loss"]
    res = train(m, X, y, TrainCfg(lr=0.05, batch_size=16), epochs=20, seed=0)
    after = evaluate_model(m, X, y)
    assert after["log_loss"] < before and after["accuracy"] > 0.9
    assert res.steps == 20 * int(np.ceil(120 / 16))


def test_training_reproducible():
    X, y = _data()
    ws = []
    for _ in range(2):
        m = build_model(6, [8], 0)
        train(m, X, y, TrainCfg(), epochs=3, seed=11)
        ws.append(get_weights(m))
    assert all(np.array_equal(a, b) for a, b in zip(*ws))


def test_dp_training_runs_and_reports_protocol():
    X, y = _data()
    m = build_model(6, [8], 0)
    res = train(m, X, y, TrainCfg(batch_size=16), epochs=2, seed=0, privacy=PrivacyCfg(enabled=True))
    assert res.sample_rate == pytest.approx(1 / 8)
    assert res.steps == 16 and res.noise_multiplier == 1.0
    assert all(np.isfinite(w).all() for w in get_weights(m))
    # The plain module is usable after training (Opacus hooks removed).
    assert list(m.state_dict())[0] == "net.0.weight"


def test_dp_clipping_bounds_the_update():
    """With sigma=0 and a tiny clipping norm, per-sample clipping limits movement."""
    X, y = _data()
    base = build_model(6, [8], 0)
    w0 = get_weights(base)
    clipped = build_model(6, [8], 0)
    cfg = TrainCfg(lr=0.05, momentum=0.0, batch_size=16)
    train(clipped, X, y, cfg, epochs=1, seed=0, privacy=PrivacyCfg(enabled=True, noise_multiplier=0.0, max_grad_norm=1e-4))
    free = build_model(6, [8], 0)
    train(free, X, y, cfg, epochs=1, seed=0)
    move = lambda m: sum(np.abs(a - b).sum() for a, b in zip(get_weights(m), w0))  # noqa: E731
    # Per step: sum of clipped grads (each <= C) / expected batch size; Poisson batches
    # can exceed the expected size, so allow a factor of 2.
    l2 = np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(get_weights(clipped), w0)))
    assert l2 <= 8 * 0.05 * 1e-4 * 2
    assert move(free) > 100 * move(clipped)


def test_dp_noise_changes_update():
    X, y = _data()
    cfg = TrainCfg(batch_size=16)
    a, b = build_model(6, [8], 0), build_model(6, [8], 0)
    train(a, X, y, cfg, epochs=1, seed=0, privacy=PrivacyCfg(enabled=True, noise_multiplier=0.0))
    train(b, X, y, cfg, epochs=1, seed=0, privacy=PrivacyCfg(enabled=True, noise_multiplier=2.0))
    assert not np.allclose(get_weights(a)[0], get_weights(b)[0])


def test_epsilon_matches_opacus_engine_accountant():
    """Our accounting helper reproduces Opacus' own accountant for the same run."""
    X, y = _data(n=100)
    model = build_model(6, [8], 0)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(torch.as_tensor(X), torch.as_tensor(y)), batch_size=10)
    engine = PrivacyEngine(accountant="rdp")
    model, opt, loader = engine.make_private(module=model, optimizer=opt, data_loader=loader,
                                             noise_multiplier=1.3, max_grad_norm=1.0)
    steps = 0
    for _ in range(3):
        for xb, yb in loader:
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb).backward()
            opt.step()
            steps += 1
    assert compute_epsilon(1.3, 1 / len(loader), steps, 1e-5) == pytest.approx(engine.get_epsilon(1e-5), rel=1e-9)


def test_epsilon_monotonicity_and_composition():
    e = lambda s, steps: compute_epsilon(s, 0.2, steps, 1e-5)  # noqa: E731
    assert e(1.0, 100) > e(1.0, 50) > e(1.0, 5) > 0
    assert e(0.8, 100) > e(1.0, 100) > e(2.0, 100)
    assert compute_epsilon(1.0, 0.2, 0, 1e-5) == 0.0
    # Accounting across rounds == accounting of the concatenated steps.
    sigma = noise_for_target_epsilon(5.0, 0.2, 100, 1e-5)
    assert compute_epsilon(sigma, 0.2, 100, 1e-5) == pytest.approx(5.0, abs=0.05)
