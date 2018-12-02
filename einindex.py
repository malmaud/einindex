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
class VarList:
    level: int
    vars: List[str]


@dataclass
class IndexPattern:
    source: List[VarList]
    target: VarList


@lru_cache()
def load_grammar():
    grammar_text = Path("grammar.lark").read_text()
    grammar = lark.Lark(grammar_text)
    return grammar


class Transformer(lark.Transformer):
    def vargroup(self, vars):
        return VarList(level=0, vars=[var.value for var in vars])

    def bgroup(self, args):
        varlist = args[1]
        return VarList(level=varlist.level + 1, vars=varlist.vars)

    def start(self, args):
        source, _, target = args
        return IndexPattern(source=source.children, target=target.children)


def parse(pattern):
    grammar = load_grammar()
    tree = grammar.parse(pattern)
    new_tree = Transformer().transform(tree)
    return new_tree


def apply(pattern: IndexPattern, x, idx):
    source = pattern.source
    if len(source) == 1:
        l = source[0]
        if l.level == 1:
            if len(l.vars) == 1:
                return x[idx]
    return apply_general(pattern, x, idx)


def apply_general(pattern: IndexPattern, x, idx):
    var_set = {}
    for var_list in pattern.source:
        if var_list.level == 1:
            for var_idx, var_name in enumerate(var_list.vars):
                var_set[var_name] = var_idx
            break
    out = x.new_zeros(size=idx.size())
    size_set = [range(dim) for dim in idx.size()]
    for i in itertools.product(*size_set):
        idx_tuple = []
        for var_list in pattern.source:
            if var_list.level == 0:
                for var in var_list.vars:
                    pos = var_set[var]
                    idx_tuple.append(i[pos])
            elif var_list.level == 1:
                inner_tuple = []
                for var in var_list.vars:
                    pos = var_set[var]
                    inner_tuple.append(i[pos])
                index_value = idx[tuple(inner_tuple)]
                idx_tuple.append(index_value)
        out[i] = x[tuple(idx_tuple)]
    return out


def index(pattern: str, x, idx):
    return apply(parse(pattern), x, idx)
