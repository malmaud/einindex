import einindex
import pytest
import torch


def test_parse():
    res = einindex.parse("i j, [j]i ->i")
    return res


def tensor_equal(t1, t2):
    if not isinstance(t1, torch.Tensor):
        t1 = torch.tensor(t1)
    if not isinstance(t2, torch.Tensor):
        t2 = torch.tensor(t2)
    return torch.all(t1 == t2).item() == 1


def test_apply():
    main = torch.tensor([[0, 1, 2], [3, 4, 5]])
    idx = torch.tensor([1, 2])
    res = einindex.index("i j, [j]i->i", main, idx)
    assert tensor_equal(res, [1, 5])

    idx = torch.tensor([[1, 0], [1, 2]])
    res = einindex.index("i j, [j]i k->i j", main, idx)
    assert tensor_equal(res, [[1, 0], [4, 5]])

    idx = torch.tensor([1, 0, 0])
    res = einindex.index("i j,  [i]j->j", main, idx)
    assert tensor_equal(res, [3, 1, 2])

    main = torch.tensor([0, 1, 5])
    idx = torch.tensor([0, 2])
    res = einindex.index("i,[i]j->j", main, idx)
    assert tensor_equal(res, [0, 5])

