"""Small MPI helpers. ``comm`` is any mpi4py-style communicator (``gather``/``allgather``)."""
import hashlib

import numpy as np


def gather_array_to_rank0(x_local, comm):
    """Gather per-rank ``(n_i, dim)`` arrays onto rank 0.

    Returns ``(x_full, sizes, index_ranges)``. ``x_full`` is the ``(sum(n_i), dim)``
    stack on rank 0 and ``None`` elsewhere; ``sizes`` and ``index_ranges`` (the
    ``(start, stop)`` row range each rank owns in ``x_full``) are returned on every rank.
    """
    rank = comm.Get_rank()
    x_local = np.ascontiguousarray(x_local, dtype=np.float64)

    sizes = np.array(comm.allgather(x_local.shape[0]), dtype=np.int32)
    starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    stops = starts + sizes
    index_ranges = list(zip(starts, stops))

    # The surface mesh is small, so a pickle-based gather is fine and keeps this module
    # free of any mpi4py import.
    pieces = comm.gather(x_local, root=0)
    if rank == 0:
        return np.concatenate(pieces, axis=0), sizes, index_ranges
    return None, sizes, index_ranges


def hash_array_tol(arr, tol=1e-8, length=16):
    """Tolerance-aware short SHA-256 hash of an array (used to name cache files).

    Values are rounded to multiples of ``tol`` first, so arrays that differ only by
    round-off noise hash identically.
    """
    rounded = np.round(np.asarray(arr) / tol) * tol
    return hashlib.sha256(rounded.astype(np.float64).tobytes()).hexdigest()[:length]
