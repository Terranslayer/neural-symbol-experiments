# -*- coding: utf-8 -*-
import pytest
from backend.core.memory import WorkingMemory

def test_allocate_ok():
    wm = WorkingMemory(6)
    D, R = wm.allocate(2)
    assert (D, R) == (2, 4)
    assert D + R == wm.K

@pytest.mark.parametrize("bad_D", [0, -1, 6, 7])
def test_allocate_invalid(bad_D):
    wm = WorkingMemory(6)
    with pytest.raises(Exception):
        wm.allocate(bad_D)
