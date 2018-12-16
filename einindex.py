import re
import torch
import typing
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import lark
import itertools
from functools import lru_cache


@dataclass
class Var:
    var: str


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
