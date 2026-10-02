"""Stand-ins for DAFoam, IDWarp and PETSc with *linear, exactly known* physics.

The real solvers cannot run in CI, but everything this package contributes is the wiring
between CSDL and DAFoam's Python API: which arrays go where, which transpose is applied,
which seed multiplies what. A linear fake makes the correct total derivatives available in
closed form, so that wiring can be checked exactly.

Model
-----
inputs   x (volume coords), v = (airspeed, aoa), p, T
residual R(w; inputs) = A w - (Bx x + Bv v + Bp p + BT T)      (so w = A^-1 (...))
function F_k(w; inputs) = c_k . w + d_k . x + e_k . v
"""
import numpy as np


class FakeVec:
    """The few petsc4py.Vec methods the package uses, backed by a numpy array."""

    def __init__(self, array=None):
        self.array = None if array is None else np.array(array, dtype=float)

    def create(self, comm=None):
        return self

    def setSizes(self, sizes, bsize=None):
        self.array = np.zeros(sizes[0])

    def setFromOptions(self):
        pass

    def zeroEntries(self):
        self.array[:] = 0.0

    def set(self, value):
        self.array[:] = value

    def axpy(self, alpha, other):
        self.array += alpha * other.array

    def norm(self):
        return float(np.linalg.norm(self.array))

    def destroy(self):
        pass


class FakeMat:
    """Stores the transposed Jacobian handed to it by ``calcdRdWT``."""

    def __init__(self):
        self.matrix = None

    def create(self, comm=None):
        return self

    def destroy(self):
        pass


class FakeKSP:
    def __init__(self):
        self.pc_matrix = None

    def create(self, comm=None):
        return self

    def destroy(self):
        pass

    def setTolerances(self, **kwargs):
        pass


class FakePETSc:
    COMM_WORLD = None
    DECIDE = -1
    Vec = FakeVec
    Mat = FakeMat
    KSP = FakeKSP


class FakeComm:
    """Single-process stand-in for an mpi4py communicator."""

    rank = 0
    size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        pass

    def bcast(self, value, root=0):
        return value

    def allgather(self, value):
        return [value]

    def gather(self, value, root=0):
        return [value]

    def allreduce(self, value, op=None):
        return value


class FakeLinearDAFoam:
    """Linear fake of a ``PYDAFoam`` instance (flow solver part)."""

    def __init__(self, n_states=5, n_vol=6, adjoint_method="fixedPoint", seed=0, run_directory=None):
        rng = np.random.default_rng(seed)
        self.n_states, self.n_vol = n_states, n_vol
        self.A = np.eye(n_states) * 4.0 + 0.3 * rng.standard_normal((n_states, n_states))
        self.B = {
            "aero_vol_coords": rng.standard_normal((n_states, n_vol)),
            "patch_velocity": rng.standard_normal((n_states, 2)),
            "pressure": rng.standard_normal((n_states, 1)),
            "temperature": rng.standard_normal((n_states, 1)),
        }
        # functions: F = c.w + d.x + e.v
        self.c = {"CD": rng.standard_normal(n_states), "CL": rng.standard_normal(n_states)}
        self.d = {"CD": rng.standard_normal(n_vol), "CL": rng.standard_normal(n_vol)}
        self.e = {"CD": rng.standard_normal(2), "CL": rng.standard_normal(2)}

        self.options = {
            "adjEqnSolMethod": adjoint_method,
            "adjUseColoring": False,
            "writeMinorIterations": False,
            "adjPCLag": 1,
            "adjEqnOption": {"useNonZeroInitGuess": False, "dynAdjustTol": False},
            "useAD": {"mode": "reverse"},
            "inputInfo": {
                "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
                "patch_velocity": {"type": "patchVelocity", "components": ["solver", "function"]},
                "pressure": {"type": "patchVar", "varName": "p", "components": ["solver"]},
                "temperature": {"type": "patchVar", "varName": "T", "components": ["solver"]},
            },
            "function": {"CD": {}, "CL": {}},
        }
        self.run_directory = run_directory
        self.comm = FakeComm()
        self.dRdWTPC = None
        self.ksp = None
        self.primalFail = 0
        self.mesh_ok = 1
        self.adjoint_fail = 0
        self.solver = self  # checkMesh, calcdRdWT, ...
        self.solverAD = self  # createMLRKSPMatrixFree, runFPAdj, ...
        self.calls = []  # (method, cwd) for the working-directory test
        self._inputs = {}
        self._w = np.zeros(n_states)

    # ---- options / sizes -------------------------------------------------------------
    def getOption(self, name):
        return self.options[name]

    def getNLocalAdjointStates(self):
        return self.n_states

    def initializedRdWTMatrixFree(self):
        pass

    # ---- state / input plumbing ------------------------------------------------------
    def set_solver_input(self, input_vals):
        import os

        self.calls.append(("set_solver_input", os.getcwd()))
        self._inputs = {k: np.array(v, dtype=float) for k, v in input_vals.items()}

    def checkMesh(self):
        return self.mesh_ok

    def writeFailedMesh(self):
        pass

    def __call__(self):
        import os

        self.calls.append(("primal", os.getcwd()))
        rhs = sum(self.B[k] @ np.atleast_1d(v) for k, v in self._inputs.items())
        self._w = np.linalg.solve(self.A, rhs)

    def calcPrimalResidualStatistics(self, mode):
        pass

    def getStates(self):
        return self._w.copy()

    def setStates(self, states):
        self._w = np.array(states, dtype=float)

    def evalFunctions(self, funcs):
        x = self._inputs["aero_vol_coords"]
        v = self._inputs["patch_velocity"]
        for k in ("CD", "CL"):
            funcs[k] = float(self.c[k] @ self._w + self.d[k] @ x + self.e[k] @ v)

    # ---- adjoint ---------------------------------------------------------------------
    def calcJacTVecProduct(self, input_name, input_type, jac_input, function_name, kind, seed, product):
        seed = np.atleast_1d(seed)
        if kind == "residual":  # (dR/d input)^T seed
            if input_name == "dafoam_solver_states":
                product[:] = self.A.T @ seed
            else:
                product[:] = -self.B[input_name].T @ seed
        else:  # function
            if input_name == "dafoam_solver_states":
                product[:] = self.c[function_name] * seed[0]
            elif input_name == "aero_vol_coords":
                product[:] = self.d[function_name] * seed[0]
            elif input_name == "patch_velocity":
                product[:] = self.e[function_name] * seed[0]
            else:
                product[:] = 0.0

    def array2Vec(self, array):
        return FakeVec(array)

    def vec2Array(self, vec):
        return vec.array.copy()

    def renameSolution(self, counter):
        return 0.0, True

    def writeAdjointFields(self, name, time, psi):
        pass

    def runFPAdj(self, dFdW, psi):
        psi.array[:] = np.linalg.solve(self.A.T, dFdW.array)
        return self.adjoint_fail

    def calcdRdWT(self, mode, mat):
        mat.matrix = self.A.T.copy()

    def createMLRKSPMatrixFree(self, mat, ksp):
        ksp.pc_matrix = mat

    def solveLinearEqn(self, ksp, dFdW, psi):
        psi.array[:] = np.linalg.solve(ksp.pc_matrix.matrix, dFdW.array)
        return self.adjoint_fail

    def runColoring(self):
        pass


class FakeWarpMesh:
    """Linear IDWarp stand-in: ``x_vol = W x_surf + x_vol0``."""

    def __init__(self, W, x_vol0):
        self.W, self.x_vol0 = W, x_vol0
        self.x_surf = np.zeros(W.shape[1])
        self._dxs = None

    def getSolverGrid(self):
        return self.W @ self.x_surf + self.x_vol0

    def warpMesh(self):
        pass

    def warpDeriv(self, dxV):
        self._dxs = (self.W.T @ np.asarray(dxV)).reshape(-1, 3)

    def getdXs(self):
        return self._dxs


class FakeWarpingDAFoam:
    """Linear fake of the IDWarp-facing part of a ``PYDAFoam`` instance."""

    designSurfacesGroup = "designSurfaces"
    allWallsGroup = "allWalls"

    def __init__(self, n_surf_points=4, n_vol=9, seed=1, run_directory=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_vol, 3 * n_surf_points))
        self.mesh = FakeWarpMesh(self.W, rng.standard_normal(n_vol))
        self.run_directory = run_directory

    def setSurfaceCoordinates(self, coordinates, group_name=None):
        assert group_name == self.designSurfacesGroup
        self.mesh.x_surf = np.asarray(coordinates).flatten()

    def mapVector(self, vec, from_group, to_group):
        assert (from_group, to_group) == (self.allWallsGroup, self.designSurfacesGroup)
        return vec


class FakeAirfoilDAFoam(FakeLinearDAFoam):
    """Linear flow fake plus an identity mesh warp over a given design surface.

    ``x_vol`` is the flattened design surface itself, so the whole chain
    FFD -> surface -> warp -> flow states -> CL/CD stays linear in the surface coordinates
    and the geometry-to-force derivatives can be checked end to end.
    """

    designSurfacesGroup = "designSurfaces"
    allWallsGroup = "allWalls"

    def __init__(self, surface_points, run_directory=None, seed=0):
        n_vol = surface_points.size
        super().__init__(n_states=6, n_vol=n_vol, adjoint_method="fixedPoint", seed=seed, run_directory=run_directory)
        # scale the force functions so CL, CD are O(1) for O(0.1) surface motions
        for k in self.d:
            self.d[k] = self.d[k] / n_vol
        self.surface_points = np.asarray(surface_points, dtype=float)
        self.mesh = FakeWarpMesh(np.eye(n_vol), np.zeros(n_vol))
        self.mesh.x_surf = self.surface_points.flatten()

    def getSurfaceCoordinates(self, group_name=None):
        return self.surface_points.copy()

    def setSurfaceCoordinates(self, coordinates, group_name=None):
        self.mesh.x_surf = np.asarray(coordinates).flatten()

    def mapVector(self, vec, from_group, to_group):
        return vec


class FakeJaxAirfoilDAFoam(FakeLinearDAFoam):
    """Linear flow fake for ``warper="jax"``: DAFoam only owns a (shuffled, duplicated) set of
    volume points, as a decomposed mesh does; the warp itself happens in IDWarp-JAX."""

    def __init__(self, global_points, run_directory=None, seed=0):
        rng = np.random.default_rng(seed)
        # a rank's local mesh: a random subset in random order, plus a few processor-boundary
        # duplicates of other points
        subset = rng.permutation(len(global_points))[: len(global_points) * 2 // 3]
        extra = rng.choice(subset, size=25, replace=False)
        self.local_to_global_truth = np.concatenate([subset, extra])
        self.xv0 = global_points[self.local_to_global_truth].copy()
        n_vol = self.xv0.size
        super().__init__(n_states=6, n_vol=n_vol, adjoint_method="fixedPoint", seed=seed, run_directory=run_directory)
        for k in self.d:
            self.d[k] = self.d[k] / n_vol
        for k in self.B:
            self.B[k] = self.B[k] / np.sqrt(max(n_vol, 1)) if k == "aero_vol_coords" else self.B[k]
