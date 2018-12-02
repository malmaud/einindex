import einindex
import pytest
import torch


@pytest.fixture
def pattern():
    return "i, [i,j]->i,j"


def test_parse(pattern):
    res = einindex.parse(pattern)
    return res


def test_apply(pattern):
    parse = einindex.parse(pattern)
    x = torch.tensor([[1, 2], [3, 4]]).float()
    idx = torch.tensor([[1, 0], [0, 1]])
    res = einindex.apply(parse, x, idx)
    return res
