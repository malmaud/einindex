import torch
import typing
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import lark
import itertools
from functools import lru_cache
import copy


@dataclass
class Var:
    var: str

    def __eq__(self, other):
        return self.var == other.var

    def __hash__(self):
        return hash(("Var", self.var))


@dataclass
class VarList:
    vars: List[Var]


@dataclass
class IndexParam:
    index: Var
    index_vars: VarList


@dataclass
class Source:
    main: VarList
    indices: IndexParam


@dataclass
class IndexExpr:
    source: Source
    target: VarList


@lru_cache()
def load_grammar():
    grammar_text = Path("grammar.lark").read_text()
    grammar = lark.Lark(grammar_text)
    return grammar


class Transformer(lark.Transformer):
    def varlist(self, vars):
        return VarList(vars=[Var(var.value) for var in vars])

    def indexexpr(self, expr):
        return Var(expr[1].value)

    def indexparam(self, params):
        return IndexParam(index=params[0], index_vars=params[1])

    def target(self, t):
        return t[0]

    def source(self, t):
        return Source(main=t[0], indices=t[2])

    def start(self, t):
        return IndexExpr(source=t[0], target=t[2])


def parse(pattern):
    grammar = load_grammar()
    tree = grammar.parse(pattern)
    new_tree = Transformer().transform(tree)
    return new_tree


def apply_pattern(pattern: IndexExpr, main, indices):
    pattern = copy.deepcopy(pattern)
    main_vars = pattern.source.main.vars
    index_target = pattern.source.indices.index
    index_vars = pattern.source.indices.index_vars.vars
    idx = 0
    splice_out: List[int] = []
    while len(main_vars) > len(index_vars):
        if idx < len(index_vars) and main_vars[idx] == index_vars[idx]:
            idx += 1
            continue
        splice_out.append(idx)
        index_vars.insert(idx, main_vars[idx])
        slices: List = []
        for i in range(len(index_vars)):
            if i == idx:
                slices.append(None)
            else:
                slices.append(slice(None))
        indices = indices[tuple(slices)]
        idx += 1
    index_dim = main_vars.index(index_target)
    result = main.gather(index_dim, indices)
    splices: List = []
    for dim in range(len(main_vars)):
        if dim in splice_out:
            splices.append(0)
        else:
            splices.append(slice(None))
    result = result[tuple(splices)]
    return result


def index(pattern: str, main, indices):
    return apply_pattern(parse(pattern), main, indices)
