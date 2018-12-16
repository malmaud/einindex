import einindex
import pytest
import torch


@pytest.fixture
def pattern():
    return "i j, [j]ik ->ik"


def test_parse(pattern):
    res = einindex.parse(pattern)
    return res
