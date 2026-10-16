"""Pixel-to-token mask mapping, protocol equations 10-12 (Milestone 3)."""

import pytest
import torch

from density.checks import allocation as checks
from density.token_map import cell_coverage, frame_groups


@pytest.mark.parametrize("index", range(5))
def test_token_mapping(index):
    result = checks.check_t15()[index]
    assert result["passed"], result


def test_rejects_sizes_that_do_not_map():
    with pytest.raises(ValueError):
        cell_coverage(torch.zeros(1, 20, 32))
    with pytest.raises(ValueError):
        frame_groups(16)
