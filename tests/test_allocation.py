"""Need score, rankings and group assignment (Milestone 3), at full size with synthetic inputs."""

import pytest
import torch

from density.checks import allocation as checks

DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"))]


def assert_rows(rows):
    failures = [r for r in rows if r["passed"] is not True]
    assert not failures, failures


@pytest.mark.parametrize("device", DEVICES)
def test_need_score(device):
    assert_rows(checks.check_t8_unit(device))


def test_ties_at_the_group_boundary():
    assert_rows(checks.check_t9())


def test_registered_group_sizes_at_full_size():
    assert_rows(checks.check_full_size_sizes())


def test_importance_times_need_invariants():
    assert_rows(checks.check_t12())


def test_literal_t12_fails_for_fractional_coverage():
    assert checks.t12_counterexample()["masked_token_fell"]


def test_coverage_rule_ignores_need():
    assert checks.check_t13()["passed"]


def test_binary_roi_first():
    assert checks.check_t14()["passed"]


def test_labels_assigned_once_and_frozen():
    assert_rows(checks.check_assignment())


def test_negative_controls_fail():
    assert not all(r["passed"] for r in checks.check_t9(checks.ascending_need, label="ascending"))
    assert not checks.check_t13(checks.g_seeded_permutation)["passed"]
