from cytoolz.utils_test import module_doctest

import cytoolz
import cytoolz.dicttoolz
import cytoolz.functoolz
import cytoolz.itertoolz
import cytoolz.recipes


def test_doctest():
    assert module_doctest(cytoolz) is True
    assert module_doctest(cytoolz.dicttoolz) is True
    assert module_doctest(cytoolz.functoolz) is True
    assert module_doctest(cytoolz.itertoolz) is True
    assert module_doctest(cytoolz.recipes) is True
