"""Dataset parsing, splitting, preprocessing (leakage) and partitioning."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from conftest import requires_heart, requires_wdbc
from fedmed.config import ExperimentCfg
from fedmed.data.datasets import WDBC_FEATURES, load_heart_disease, load_wdbc
from fedmed.data.download import SOURCES, sha256_of
from fedmed.data.partition import make_partition
from fedmed.data.preprocessing import Preprocessor
from fedmed.data.splits import make_split
from fedmed.experiments.prepare import prepare
from fedmed.paths import RAW_DIR


# ---------------------------------------------------------------- parsing
@requires_wdbc
def test_wdbc_archive_checksum():
    assert sha256_of(RAW_DIR / "wdbc" / "wdbc.zip") == SOURCES["wdbc"].sha256


@requires_wdbc
def test_wdbc_structure_and_labels():
    ds = load_wdbc()
    assert ds.X.shape == (569, 30)
    assert list(ds.X.columns) == WDBC_FEATURES
    assert "id" not in ds.X.columns
    assert int(ds.y.sum()) == 212 and int((ds.y == 0).sum()) == 357  # 212 M / 357 B per wdbc.names
    assert not ds.X.isna().any().any()
    assert ds.X["radius_mean"].iloc[0] == pytest.approx(17.99)  # first row of wdbc.data, label M
    assert ds.y[0] == 1


@requires_heart
def test_heart_disease_structure():
    ds = load_heart_disease()
    assert len(ds.y) == 920
    sites, counts = np.unique(ds.groups, return_counts=True)
    assert dict(zip(sites, counts)) == {"cleveland": 303, "hungary": 294, "switzerland": 123, "va_long_beach": 200}
    assert int(ds.y.sum()) == 139 + 106 + 115 + 149
    assert not {"slope", "ca", "thal", "num"} & set(ds.X.columns)
    assert ds.X.loc[ds.groups == "switzerland", "chol"].isna().all()  # chol==0 -> missing
    assert (ds.X["chol"].dropna() > 0).all()


# ---------------------------------------------------------------- splits
def test_split_disjoint_stratified_reproducible():
    y = np.array([0] * 70 + [1] * 30)
    s1, s2 = make_split(y, seed=3), make_split(y, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(s1.as_dict().values(), s2.as_dict().values()))
    allidx = np.concatenate([s1.train, s1.val, s1.test])
    assert len(allidx) == len(set(allidx)) == 100
    assert y[s1.test].mean() == pytest.approx(0.3, abs=0.05)
    assert not np.array_equal(make_split(y, seed=4).test, s1.test)


# ---------------------------------------------------------------- preprocessing
def test_federated_stats_equal_pooled_and_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, size=(90, 4))
    parts = [X[:20], X[20:55], X[55:]]
    fed = Preprocessor.fit_federated(parts, list("abcd"))
    pooled = Preprocessor.fit(X, list("abcd"))
    sk = StandardScaler().fit(X)
    np.testing.assert_allclose(fed.mean, sk.mean_)
    np.testing.assert_allclose(fed.scale, sk.scale_)
    np.testing.assert_allclose(fed.transform(X), pooled.transform(X), atol=1e-6)


def test_mean_imputation_uses_observed_values():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
    pre = Preprocessor.fit(X, ["a", "b"])
    assert pre.mean[1] == pytest.approx(6.0)
    assert pre.transform(X)[0, 1] == pytest.approx(0.0)  # imputed to the mean -> 0 after scaling


@requires_wdbc
def test_no_test_leakage_in_preprocessing():
    """Preprocessing statistics depend only on training rows."""
    cfg = ExperimentCfg()
    prep = prepare(cfg)
    X = prep.dataset.X.to_numpy()
    expected = Preprocessor.fit(X[prep.split.train], list(prep.dataset.X.columns))
    np.testing.assert_allclose(prep.preprocessor.mean, expected.mean)
    np.testing.assert_allclose(prep.preprocessor.scale, expected.scale)
    with_test = Preprocessor.fit(X[np.concatenate([prep.split.train, prep.split.test])], list(prep.dataset.X.columns))
    assert not np.allclose(prep.preprocessor.mean, with_test.mean)


@requires_wdbc
def test_test_set_identical_across_modes_and_partitions():
    a = prepare(ExperimentCfg(mode="centralized"))
    b = ExperimentCfg(mode="federated", seed=5)
    b.partition.strategy, b.partition.alpha, b.partition.n_clients = "dirichlet", 0.5, 10
    b = prepare(b)
    assert np.array_equal(a.split.test, b.split.test)
    np.testing.assert_array_equal(a.y_test, b.y_test)


# ---------------------------------------------------------------- partitioning
def _pools(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    split = make_split(y, seed=0)
    return y, split.as_dict()


@pytest.mark.parametrize("strategy,alpha", [("iid", None), ("dirichlet", 0.5), ("quantity_skew", 1.0)])
@pytest.mark.parametrize("k", [3, 5, 10])
def test_partition_is_disjoint_cover_and_reproducible(strategy, alpha, k):
    y, pools = _pools()
    p1 = make_partition(pools, y, strategy, k, seed=1, alpha=alpha)
    p2 = make_partition(pools, y, strategy, k, seed=1, alpha=alpha)
    for pool, idx in pools.items():
        got = np.concatenate([c[pool] for c in p1.client_indices])
        assert sorted(got.tolist()) == sorted(idx.tolist())  # every record exactly once
        for c1, c2 in zip(p1.client_indices, p2.client_indices):
            assert np.array_equal(c1[pool], c2[pool])
    assert min(len(c["train"]) for c in p1.client_indices) >= 8
    summary = p1.summary(y)
    assert sum(r["n_train"] for r in summary) == len(pools["train"])


def test_iid_balanced_dirichlet_skewed():
    y, pools = _pools(600)
    iid = make_partition(pools, y, "iid", 5, seed=0)
    skew = make_partition(pools, y, "dirichlet", 5, seed=0, alpha=0.1)
    frac = lambda p: np.array([r["pos_frac_train"] for r in p.summary(y)])  # noqa: E731
    assert frac(iid).std() < 0.03
    assert frac(skew).std() > 0.15


def test_natural_partition_by_site():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 10)
    groups = np.array(["a", "a", "b", "b", "c", "c", "a", "b"] * 10)
    split = make_split(y, groups, seed=0)
    p = make_partition(split.as_dict(), y, "natural", 0, seed=0, groups=groups, min_train_per_client=1)
    assert p.client_names == ["a", "b", "c"]
    for name, c in zip(p.client_names, p.client_indices):
        assert set(groups[c["train"]]) == {name}
