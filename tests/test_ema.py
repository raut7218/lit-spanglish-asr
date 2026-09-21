import pytest

torch = pytest.importorskip("torch")

from lit.train import EMA, selection_score  # noqa: E402


def test_ema_warmup_forgets_the_untrained_init():
    """A plain 0.999 EMA started at the init keeps 0.999**1000 = 37% of it after 1000 steps (the v2 bug)."""
    p = torch.nn.Parameter(torch.zeros(3))
    ema = EMA([p], 0.999)
    for _ in range(1000):
        p.data.fill_(1.0)
        ema.update([p])
    assert ema.shadow[0].min() > 0.99


def test_ema_decay_is_capped_and_grows():
    ema = EMA([torch.nn.Parameter(torch.zeros(1))], 0.999)
    ema.n = 0
    assert ema.current_decay() < 0.2
    ema.n = 100_000
    assert ema.current_decay() == 0.999


def test_selection_score_is_dev_weighted_and_ignores_missing_sets():
    res = {"dev": 0.10, "holdout_turn": 0.20, "holdout_window": 0.9}
    assert selection_score(res) == pytest.approx(0.7 * 0.10 + 0.3 * 0.20)
    assert selection_score({"holdout_turn": 0.2}) == pytest.approx(0.2)
    assert selection_score(res, {"holdout_turn": 1.0}) == pytest.approx(0.2)
