"""The exploratory C7 check (Chase, 2026-10-08): the inverse ranking, its definition check, the region shares of C3,
C4 and C7 from one g, and the registered decision rule."""

import pytest
import torch

from density import exploratory as X
from density.allocation import assign_groups, rank_need
from density.conditions import CONDITIONS, lookup
from density.hsa import HIGH


def toy(n=12):
    g = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.2, 0.6, 0.5, 0.3, 0.95, 0.05, 0.4, 0.4], dtype=torch.float32)[:n]
    m = torch.tensor([0.0, 0.0, 1.0, 0.5, 0.0, 0.05, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=torch.float32)[:n]  # region: 2, 3, 8 (m >= 0.1)
    return g, m


def test_c7_ranks_outside_tokens_by_need_and_region_tokens_last():
    g, m = toy()
    order = X.rank_outside_then_need(g, m).tolist()
    assert set(order[-3:]) == {2, 3, 8}  # the region (b = 1) ranks after every outside token
    assert order[:9] == [0, 5, 6, 10, 11, 7, 4, 1, 9]  # outside: g descending; 0.4 ties by ascending token ID
    labels = assign_groups(X.rank_outside_then_need(g, m), 4)
    assert X.check_c7(labels, m) == {"full_step_region_tokens": 0, "full_step_tokens_with_m_above_0": 1}  # token 5, m = 0.05 < 0.1
    with pytest.raises(RuntimeError, match="region tokens would get full steps"):
        X.check_c7(assign_groups(X.rank_outside_then_need(g, m), 10), m)  # only 9 outside tokens for 10 full-step places


def test_region_shares_of_c3_c4_and_c7_come_from_one_g():
    g, m = toy()
    group_iterations = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)  # HIGH runs 4 iterations, LOW 2
    labels_c3 = assign_groups(rank_need(g), 4)
    labels_c7 = assign_groups(X.rank_outside_then_need(g, m), 4)
    out = X.same_g_diagnostics(g, m, group_iterations, 4, labels_c7, labels_c3, lam=10.0)
    c = out["conditions"]
    assert c["C7"]["full_step_roi_tokens"] == 0 and c["C7"]["s_roi"] < c["C3"]["s_roi"] <= c["C4@10"]["s_roi"]
    assert c["C3"]["changed_tokens"] == 0 and c["C7"]["labels_sha256"] != c["C3"]["labels_sha256"]
    # C7's region share is the floor: every region token runs the LOW group's 2 iterations, of 4 x 4 + 8 x 2 evaluations
    assert c["C7"]["s_roi"] == pytest.approx(3 * 2 / (4 * 4 + 8 * 2))


@pytest.mark.parametrize("r, outcome", [(1.0, "within 3 percent"), (0.97, "within 3 percent"), (1.03, "within 3 percent"),
                                        (1.031, "in between"), (1.10, "in between"), (1.1001, "the lever is real"), (0.9, "in between")])
def test_the_registered_decision_rule(r, outcome):
    assert outcome in X.classify(r)


def test_c7_stays_out_of_the_registered_conditions():
    assert "C7" not in CONDITIONS and lookup("C7") is X.C7 and lookup("C4") is CONDITIONS["C4"]
    assert X.C7.steps == 40 and X.C7.hsa and X.C7.mask == "roi" and X.COMPARE_LAMBDA == 10.0 and X.PRESET == "HSA-50"
