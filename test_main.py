import einindex
import pytest


@pytest.fixture
def pattern():
    return "a b,b->c"


def test_parse(pattern):
    pattern = "a b,b->c"
    einindex.parse(pattern)

