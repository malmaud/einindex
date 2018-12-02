import re
import torch
import typing
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import lark


@dataclass
class VarList:
    level: int
    vars: List[str]


@dataclass
class IndexPattern:
    source: List[VarList]
    target: VarList


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
