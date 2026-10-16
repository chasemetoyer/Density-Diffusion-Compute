"""Allocation diagnostics, protocol equations 17-20 (Milestone 3)."""

import pytest

from density.checks import allocation as checks


@pytest.mark.parametrize("index", range(3))
def test_diagnostics(index):
    result = checks.check_t16()[index]
    assert result["passed"], result
