from abtest import mde_proportions, sample_size_means, sample_size_proportions


def test_sample_size_matches_textbook():
    # 5% → 6% conversion, alpha .05 two-sided, power .8 ≈ 8,000 per arm (classic example)
    n = sample_size_proportions(0.05, 0.01)
    assert 7500 < n < 8700


def test_bigger_effect_needs_fewer_users():
    assert sample_size_proportions(0.05, 0.02) < sample_size_proportions(0.05, 0.01)


def test_means_sample_size():
    assert sample_size_means(sigma=10, mde=1) == 1570  # 2*((1.96+0.84)*10/1)^2 ≈ 1570


def test_mde_inverts_sample_size():
    n = sample_size_proportions(0.10, 0.01)
    assert abs(mde_proportions(0.10, n) - 0.01) < 0.0015
