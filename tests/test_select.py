"""Selection-rule and convergence benchmarks (§6.6, §6.7, §10).

`select_best`/`check_convergence` are pure functions over `TrialOutcome`
objects, so these are hand-built directly rather than produced by a real
training run -- a handful of deliberately-constructed scenarios is sufficient
to exercise the logical cases the design doc calls out.
"""

from __future__ import annotations

import numpy as np
import pytest

from hp_agent.config import TuningConfig
from hp_agent.evaluate import TrialOutcome
from hp_agent.select import check_convergence, compute_noise_floor, select_best


def _trial(iteration, metric, std, overfit, fit_time=1.0, status="ok"):
    train_metric = metric + overfit if metric is not None and overfit is not None else None
    return TrialOutcome(
        iteration=iteration, source="seed_design", config={"x": iteration}, status=status,
        primary_metric=metric, metric_std=std, train_metric=train_metric,
        overfit_gap=overfit, fit_time_s=fit_time if status == "ok" else None,
    )


# --------------------------------------------------------------------------- #
# select_best: never a raw metric-max (§6.6)
# --------------------------------------------------------------------------- #
def test_select_best_census_reference_picks_low_overfit_within_noise_floor():
    """§14's own numbers: trial 23 scores highest but is badly overfit; trial 19
    is within the noise floor of it and far less overfit -- must win."""
    trials = [
        _trial(19, metric=0.795, std=0.006, overfit=0.018),
        _trial(23, metric=0.798, std=0.006, overfit=0.041),  # highest metric, high overfit
        _trial(7, metric=0.760, std=0.02, overfit=0.02),      # clearly worse, outside noise floor
    ]
    picked = select_best(trials)
    assert picked.iteration == 19
    top_scorer = max(trials, key=lambda t: t.primary_metric)
    assert picked.iteration != top_scorer.iteration
    assert picked.overfit_gap < top_scorer.overfit_gap


def test_select_best_never_picks_the_single_lucky_high_variance_outlier():
    """A one-off trial with the highest raw metric also has the widest fold
    spread (its own metric_std) and worst overfit gap -- a tighter, nearby
    plateau of trials must win instead."""
    trials = [
        _trial(1, metric=0.80, std=0.10, overfit=0.06),   # highest metric, but the noisy outlier
        _trial(2, metric=0.77, std=0.01, overfit=0.02),
        _trial(3, metric=0.775, std=0.01, overfit=0.015),  # least overfit within the outlier's noise floor
        _trial(4, metric=0.60, std=0.02, overfit=0.03),    # clearly worse, outside noise floor
    ]
    picked = select_best(trials)
    assert picked.iteration == 3
    assert picked.iteration != 1


def test_select_best_picks_the_top_trial_when_it_is_also_least_overfit():
    """Sanity check: the rule must not unnecessarily avoid a top trial that
    genuinely deserves to win (no bias against high scores per se)."""
    trials = [
        _trial(1, metric=0.90, std=0.01, overfit=0.01),
        _trial(2, metric=0.85, std=0.01, overfit=0.05),  # outside noise floor of trial 1
    ]
    picked = select_best(trials)
    assert picked.iteration == 1


def test_select_best_tie_breaks_on_fastest_fit_time():
    trials = [
        _trial(1, metric=0.80, std=0.02, overfit=0.02, fit_time=5.0),
        _trial(2, metric=0.80, std=0.02, overfit=0.02, fit_time=2.0),
    ]
    picked = select_best(trials)
    assert picked.iteration == 2


def test_select_best_ignores_failed_and_timeout_trials():
    trials = [
        _trial(1, metric=None, std=None, overfit=None, status="failed"),
        _trial(2, metric=None, std=None, overfit=None, status="timeout"),
        _trial(3, metric=0.70, std=0.01, overfit=0.02),
    ]
    picked = select_best(trials)
    assert picked.iteration == 3


def test_select_best_returns_none_when_nothing_succeeded():
    trials = [_trial(1, metric=None, std=None, overfit=None, status="failed")]
    assert select_best(trials) is None
    assert select_best([]) is None


@pytest.mark.parametrize("seed", range(20))
def test_select_best_randomized_plateau_never_picks_the_lucky_outlier(seed):
    """§10's selection-rule benchmark, verbatim: 'a synthetic objective with a
    known noisy plateau near the optimum ... across 20 seeds, and never the
    single highest-variance outlier trial.' Randomized on top of the
    hand-crafted cases above, rather than in place of them."""
    rng = np.random.default_rng(seed)
    n_plateau = int(rng.integers(4, 9))
    plateau_metric = 0.80
    trials = []
    it = 1
    for _ in range(n_plateau):
        m = plateau_metric + rng.normal(0.0, 0.003)
        std = abs(rng.normal(0.006, 0.001)) + 0.001
        overfit = abs(rng.normal(0.02, 0.008))
        trials.append(_trial(it, metric=m, std=std, overfit=overfit))
        it += 1
    # the lucky outlier: a strictly higher raw metric, but far noisier and more
    # overfit than anything in the plateau -- must never be selected
    outlier_metric = plateau_metric + 0.05 + abs(rng.normal(0.0, 0.01))
    outlier = _trial(it, metric=outlier_metric, std=0.12, overfit=0.15)
    trials.append(outlier)
    it += 1
    # a few clearly-worse trials, well outside the noise floor either way
    for _ in range(int(rng.integers(1, 4))):
        m = plateau_metric - 0.15 - abs(rng.normal(0.0, 0.02))
        trials.append(_trial(it, metric=m, std=0.01, overfit=0.03))
        it += 1

    picked = select_best(trials)
    assert picked is not None
    assert picked.iteration != outlier.iteration, (
        f"seed {seed}: selected the lucky high-variance outlier (trial {outlier.iteration})"
    )
    best_metric = max(t.primary_metric for t in trials)
    noise_floor = compute_noise_floor(trials)
    assert best_metric - picked.primary_metric <= noise_floor + 1e-12, (
        f"seed {seed}: picked config is outside the noise floor of the best"
    )
    # never picks a config whose overfit gap is the worst among everything evaluated
    worst_overfit = max(t.overfit_gap for t in trials if t.overfit_gap is not None)
    if len(set(t.overfit_gap for t in trials)) > 1:
        assert picked.overfit_gap < worst_overfit, (
            f"seed {seed}: selected the single most-overfit trial in the whole history"
        )


# --------------------------------------------------------------------------- #
# check_convergence: patience + noise floor + minimum-trials floor (§6.7)
# --------------------------------------------------------------------------- #
def test_still_improving_sequence_never_reports_converged():
    cfg = TuningConfig(
        model=None, patience=8, convergence_window=6,
        convergence_tol=0.005, min_trials_before_convergence=8,
    )
    seed_k = 5
    trials = [_trial(i, metric=0.5 + 0.01 * i, std=0.002, overfit=0.02) for i in range(1, 31)]

    for k in range(1, len(trials) + 1):
        decision = check_convergence(trials[:k], cfg, seed_k)
        assert not decision.converged, f"falsely converged at trial {k} on a still-improving sequence"


def test_flat_sequence_converges_within_a_bounded_number_of_trials_past_the_floor():
    """A seed batch that genuinely improves, followed by a flat run of
    proposals with no further improvement, must converge -- and must do so
    within a small, bounded number of trials past `min_trials_before_convergence`,
    not immediately and not never."""
    cfg = TuningConfig(
        model=None, patience=8, convergence_window=6,
        convergence_tol=0.005, min_trials_before_convergence=8,
    )
    seed_k = 5
    min_floor = cfg.min_trials_before_convergence

    seed_metrics = [0.60, 0.64, 0.68, 0.70, 0.71]
    seed_overfits = [0.05, 0.045, 0.04, 0.035, 0.03]
    trials = [_trial(i, m, std=0.01, overfit=o) for i, (m, o) in
              enumerate(zip(seed_metrics, seed_overfits), start=1)]
    n_total = 16
    for i in range(len(trials) + 1, n_total + 1):
        trials.append(_trial(i, metric=0.71, std=0.01, overfit=0.03))  # flat: no improvement

    # never converges before the minimum-trials floor
    for k in range(1, min_floor):
        decision = check_convergence(trials[:k], cfg, seed_k)
        assert not decision.converged, f"converged before the floor at trial {k}"

    # converges somewhere reasonably soon after the floor (bounded, not immediate, not never)
    first_converged_at = None
    for k in range(min_floor, n_total + 1):
        decision = check_convergence(trials[:k], cfg, seed_k)
        if decision.converged:
            first_converged_at = k
            first_reason = decision.reason
            break
    assert first_converged_at is not None, "flat sequence never converged"
    assert first_reason in ("patience_exhausted", "noise_floor_plateau")
    bound = min_floor + cfg.patience + cfg.convergence_window
    assert min_floor <= first_converged_at <= bound, (
        f"converged at {first_converged_at}, expected within [{min_floor}, {bound}]"
    )
