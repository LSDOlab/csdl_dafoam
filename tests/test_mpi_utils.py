import numpy as np
import pytest
from fakes import FakeComm

from csdl_dafoam.mpi_utils import gather_array_to_rank0, hash_array_tol


class TwoRankView:
    """What rank ``rank`` sees of a communicator whose ranks hold ``parts``."""

    def __init__(self, parts, rank):
        self.parts, self.rank = parts, rank

    def Get_rank(self):
        return self.rank

    def allgather(self, value):
        return [p.shape[0] for p in self.parts]

    def gather(self, value, root=0):
        return list(self.parts) if self.rank == 0 else None


def test_gather_single_rank_is_identity():
    x = np.arange(12.0).reshape(4, 3)
    full, sizes, ranges = gather_array_to_rank0(x, FakeComm())
    np.testing.assert_array_equal(full, x)
    assert list(sizes) == [4]
    assert [(int(a), int(b)) for a, b in ranges] == [(0, 4)]


def test_gather_concatenates_in_rank_order_and_reports_slices():
    parts = [np.full((2, 3), 1.0), np.full((3, 3), 2.0), np.full((1, 3), 3.0)]
    full, sizes, ranges = gather_array_to_rank0(parts[0], TwoRankView(parts, 0))
    assert full.shape == (6, 3)
    for rank, part in enumerate(parts):
        a, b = ranges[rank]
        np.testing.assert_array_equal(full[a:b], part)
    assert list(sizes) == [2, 3, 1]


def test_gather_non_root_gets_none_but_keeps_ranges():
    parts = [np.zeros((2, 3)), np.ones((3, 3))]
    full, sizes, ranges = gather_array_to_rank0(parts[1], TwoRankView(parts, 1))
    assert full is None
    assert [(int(a), int(b)) for a, b in ranges] == [(0, 2), (2, 5)]


def test_hash_is_stable_and_tolerance_aware():
    x = np.linspace(0, 1, 30).reshape(10, 3)
    assert hash_array_tol(x) == hash_array_tol(x.copy())
    assert hash_array_tol(x) == hash_array_tol(x + 1e-12)
    assert hash_array_tol(x) != hash_array_tol(x + 1e-3)
    assert len(hash_array_tol(x, length=8)) == 8
