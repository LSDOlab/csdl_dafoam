"""CSDL operations that wrap a DAFoam primal/adjoint solver.

* :func:`instantiate_dafoam` builds the ``PYDAFoam`` object (and optionally IDWarp).
* :class:`DAFoamSolver` is an implicit operation: the OpenFOAM flow residuals
  ``R(w, inputs) = 0`` define the flow states ``w``. The primal solve is DAFoam's SIMPLE
  loop; the adjoint solve (``apply_inverse_jacobian``) is DAFoam's Krylov or fixed-point
  adjoint.
* :class:`DAFoamFunctions` is an explicit operation mapping states and inputs to the
  functions of interest (CD, CL, ...) declared in the DAFoam ``function`` option.

Only reverse mode is implemented.
"""
import logging
import os
import warnings

import csdl_alpha as csdl
import numpy as np

from csdl_dafoam._optional import require
from csdl_dafoam._run_directory import in_run_directory

logger = logging.getLogger(__name__)

__all__ = [
    "DAFoamError",
    "DAFoamSolver",
    "DAFoamFunctions",
    "instantiate_dafoam",
    "instantiateDAFoam",
]


class DAFoamError(RuntimeError):
    """A DAFoam primal solve failed or the mesh was rejected."""


def instantiate_dafoam(options, comm, run_directory=None, mesh_options=None):
    """Create a ``PYDAFoam`` instance for the case in ``run_directory``.

    Parameters
    ----------
    options : dict
        The DAFoam ``daOptions`` dictionary.
    comm : mpi4py.MPI.Comm
        Communicator; its size must equal ``numberOfSubdomains`` in the case's
        ``system/decomposeParDict``.
    run_directory : str or path-like, optional
        OpenFOAM case directory. Defaults to the current directory. The directory is
        remembered as ``instance.run_directory`` and the CSDL operations ``chdir`` into
        it for every DAFoam call; the caller's working directory is restored afterwards.
    mesh_options : dict, optional
        IDWarp options. If given, an ``idwarp.USMesh`` is created and attached.
    """
    dafoam = require("dafoam")

    previous_directory = os.getcwd()
    run_directory = os.path.abspath(run_directory) if run_directory else previous_directory
    os.chdir(run_directory)
    try:
        dafoam_instance = dafoam.PYDAFOAM(options=options, comm=comm)

        if mesh_options is not None:
            idwarp = require("idwarp")
            mesh = idwarp.USMesh(options=mesh_options, comm=comm)
            dafoam_instance.setMesh(mesh)
    finally:
        os.chdir(previous_directory)

    dafoam_instance.run_directory = run_directory
    return dafoam_instance


# Name used by the original (pre-package) scripts.
instantiateDAFoam = instantiate_dafoam


def _petsc():
    return require("petsc4py.PETSc")


class DAFoamSolver(csdl.experimental.CustomImplicitOperation):
    """Implicit operation solving the DAFoam flow residual equations.

    Parameters
    ----------
    dafoam_instance
        Object returned by :func:`instantiate_dafoam`.
    check_mesh : bool
        Run DAFoam's mesh-quality check before each primal solve and raise
        :class:`DAFoamError` if the mesh is rejected (this is what DAFoam's own MPhys
        wrapper does).
    on_adjoint_failure : {"warn", "raise"}
        What to do when the adjoint linear solve does not converge. ``"warn"``
        continues with the (partially converged) adjoint, which is usually good enough
        for an optimizer to keep making progress; ``"raise"`` stops.
    """

    def __init__(self, dafoam_instance, check_mesh=True, on_adjoint_failure="warn"):
        super().__init__()
        if on_adjoint_failure not in ("warn", "raise"):
            raise ValueError("on_adjoint_failure must be 'warn' or 'raise'")

        PETSc = _petsc()
        self.dafoam_instance = dafoam_instance
        self.check_mesh = check_mesh
        self.on_adjoint_failure = on_adjoint_failure
        self.solution_counter = 1

        # initialize the dRdWT matrix-free matrix in DASolver
        dafoam_instance.solverAD.initializedRdWTMatrixFree()

        # create the adjoint vector
        self.num_state_elements = dafoam_instance.getNLocalAdjointStates()
        self.psi = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.psi.setSizes((self.num_state_elements, PETSc.DECIDE), bsize=1)
        self.psi.setFromOptions()
        self.psi.zeroEntries()

        # The coloring is only needed for the Krylov adjoint
        self.runColoring = dafoam_instance.getOption("adjEqnSolMethod") != "fixedPoint"

    def evaluate(self, dafoam_input_variables_group: csdl.VariableGroup):
        logger.debug("DAFoamSolver.evaluate")
        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "solver" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

        return self.create_output("dafoam_solver_states", (self.num_state_elements,))

    def solve_residual_equations(self, input_vals, output_vals):
        logger.debug("DAFoamSolver.solve_residual_equations")
        dafoam_instance = self.dafoam_instance

        with in_run_directory(dafoam_instance):
            # set the solver input, including mesh, boundary etc.
            dafoam_instance.set_solver_input(input_vals)

            # before running the primal, we need to check if the mesh quality is good
            if self.check_mesh and dafoam_instance.solver.checkMesh() != 1:
                dafoam_instance.solver.writeFailedMesh()
                raise DAFoamError("DAFoam mesh check failed; the failed mesh was written to disk.")

            # Run primal
            dafoam_instance()

            # if the primal fails, the states are meaningless
            if dafoam_instance.primalFail != 0:
                raise DAFoamError("DAFoam primal solution failed.")

            # after solving the primal, we need to print its residual info
            if dafoam_instance.getOption("useAD")["mode"] == "forward":
                dafoam_instance.solverAD.calcPrimalResidualStatistics("print")
            else:
                dafoam_instance.solver.calcPrimalResidualStatistics("print")

            # assign the computed flow states to outputs
            states = dafoam_instance.getStates()
            output_vals["dafoam_solver_states"] = states

            # set states
            dafoam_instance.setStates(states)

            # We also need to just calculate the residual for the AD mode to initialize vars
            # like URes. We do not print the residual for AD, though
            dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")

    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        logger.debug("DAFoamSolver.apply_inverse_jacobian")
        if mode == "fwd":
            raise NotImplementedError("forward mode has not been implemented for DAFoamSolver")

        with in_run_directory(self.dafoam_instance):
            self._solve_adjoint(d_outputs, d_residuals)

    def _solve_adjoint(self, d_outputs, d_residuals):
        PETSc = _petsc()
        dafoam_instance = self.dafoam_instance
        adjEqnSolMethod = dafoam_instance.getOption("adjEqnSolMethod")

        # right hand side array from d_outputs
        dFdWArray = d_outputs["dafoam_solver_states"]
        # convert the array to vector
        dFdW = dafoam_instance.array2Vec(dFdWArray)

        # run coloring
        if dafoam_instance.getOption("adjUseColoring") and self.runColoring:
            dafoam_instance.solver.runColoring()
            self.runColoring = False

        if adjEqnSolMethod == "Krylov":
            # solve the adjoint equation using the Krylov method
            # if writeMinorIterations=True, we rename the solution in pyDAFoam.py. So we
            # don't recompute the PC
            if dafoam_instance.getOption("writeMinorIterations"):
                if dafoam_instance.dRdWTPC is None or dafoam_instance.ksp is None:
                    dafoam_instance.dRdWTPC = PETSc.Mat().create(dafoam_instance.comm)
                    dafoam_instance.solver.calcdRdWT(1, dafoam_instance.dRdWTPC)
                    dafoam_instance.ksp = PETSc.KSP().create(dafoam_instance.comm)
                    dafoam_instance.solverAD.createMLRKSPMatrixFree(
                        dafoam_instance.dRdWTPC, dafoam_instance.ksp
                    )

            # otherwise, we need to recompute the PC mat based on adjPCLag
            else:
                # NOTE: this function will be called multiple times (one time for one obj
                # func) in each opt iteration so we don't want to print the total info and
                # recompute PC for each obj, we need to use renamed to check if a recompute
                # is needed. In other words, we only recompute the PC for the first obj func
                # adjoint solution
                solutionTime, renamed = dafoam_instance.renameSolution(self.solution_counter)

                if renamed:
                    if dafoam_instance.comm.rank == 0:
                        print("Driver total derivatives for iteration: %d" % self.solution_counter, flush=True)
                        print("---------------------------------------------", flush=True)
                    self.solution_counter += 1

                # compute the preconditioner matrix for the adjoint linear equation solution
                # and initialize the ksp object. We reinitialize them every adjPCLag
                adjPCLag = dafoam_instance.getOption("adjPCLag")
                if (
                    dafoam_instance.dRdWTPC is None
                    or dafoam_instance.ksp is None
                    or (self.solution_counter - 1) % adjPCLag == 0
                ):
                    if renamed:
                        # calculate the PC mat
                        if dafoam_instance.dRdWTPC is not None:
                            dafoam_instance.dRdWTPC.destroy()
                        dafoam_instance.dRdWTPC = PETSc.Mat().create(dafoam_instance.comm)
                        dafoam_instance.solver.calcdRdWT(1, dafoam_instance.dRdWTPC)
                        # reset the KSP
                        if dafoam_instance.ksp is not None:
                            dafoam_instance.ksp.destroy()
                        dafoam_instance.ksp = PETSc.KSP().create(dafoam_instance.comm)
                        dafoam_instance.solverAD.createMLRKSPMatrixFree(
                            dafoam_instance.dRdWTPC, dafoam_instance.ksp
                        )

            # if useNonZeroInitGuess is False, we will manually reset self.psi to zero
            # this is important because we need the correct psi to update the KSP tolerance
            # in the next line
            if not dafoam_instance.getOption("adjEqnOption")["useNonZeroInitGuess"]:
                self.psi.set(0)
            else:
                # if useNonZeroInitGuess is True, we will assign the previous psi to self.psi
                self.psi = dafoam_instance.array2Vec(d_residuals["dafoam_solver_states"].copy())

            if dafoam_instance.getOption("adjEqnOption")["dynAdjustTol"]:
                # if we want to dynamically adjust the tolerance, call this function. This is
                # mostly used in the block Gauss-Seidel method in two discipline coupling
                # update the KSP tolerances the coupled adjoint before solving
                self._updateKSPTolerances(self.psi, dFdW, dafoam_instance.ksp)

            # actually solving the adjoint linear equation using Petsc
            fail = dafoam_instance.solverAD.solveLinearEqn(dafoam_instance.ksp, dFdW, self.psi)

        elif adjEqnSolMethod == "fixedPoint":
            solutionTime, renamed = dafoam_instance.renameSolution(self.solution_counter)
            if renamed:
                if dafoam_instance.comm.rank == 0:
                    print("Driver total derivatives for iteration: %d" % self.solution_counter, flush=True)
                    print("---------------------------------------------", flush=True)
                self.solution_counter += 1
            # solve the adjoint equation using the fixed-point adjoint approach
            fail = dafoam_instance.solverAD.runFPAdj(dFdW, self.psi)

        else:
            raise RuntimeError("adjEqnSolMethod=%s not valid! Options are: Krylov or fixedPoint" % adjEqnSolMethod)

        # optionally write the adjoint vector as OpenFOAM field format for post-processing
        psi_array = dafoam_instance.vec2Array(self.psi)
        solTimeFloat = (self.solution_counter - 1) / 1e4
        dafoam_instance.writeAdjointFields("function", solTimeFloat, psi_array)

        # convert the solution vector to array and assign it to d_residuals
        d_residuals["dafoam_solver_states"] = dafoam_instance.vec2Array(self.psi)

        if fail:
            message = "DAFoam adjoint linear solve did not converge."
            if self.on_adjoint_failure == "raise":
                raise DAFoamError(message)
            warnings.warn(message + " Continuing with the partially converged adjoint.", RuntimeWarning)

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        logger.debug("DAFoamSolver.compute_jacvec_product")
        if mode == "fwd":
            raise NotImplementedError("forward mode has not been implemented for DAFoamSolver")

        dafoam_instance = self.dafoam_instance

        with in_run_directory(dafoam_instance):
            # assign the states in outputs to the OpenFOAM flow fields
            # NOTE: this is not quite necessary because setStates have been called before in
            # the primal solve; here we call it just to be on the safe side
            dafoam_instance.setStates(output_vals["dafoam_solver_states"])

            if "dafoam_solver_states" in d_residuals:
                # get the reverse mode AD seed from d_residuals
                seed = d_residuals["dafoam_solver_states"]

                # loop over all inputs keys and compute the matrix-vector products accordingly
                inputDict = dafoam_instance.getOption("inputInfo")
                for inputName in list(input_vals.keys()):
                    inputType = inputDict[inputName]["type"]
                    jacInput = input_vals[inputName].copy()
                    product = np.zeros_like(jacInput)
                    dafoam_instance.solverAD.calcJacTVecProduct(
                        inputName,
                        inputType,
                        jacInput,
                        "aero_residuals",
                        "residual",
                        seed,
                        product,
                    )
                    d_inputs[inputName] += product

    def _updateKSPTolerances(self, psi, dFdW, ksp):
        # Here we need to manually update the KSP tolerances because the default relative
        # tolerance will always want to converge the adjoint to a fixed tolerance during the
        # LINGS adjoint solution. However, what we want is to converge just a few orders of
        # magnitude. Here we need to bypass the rTol in Petsc and manually calculate the aTol.
        dafoam_instance = self.dafoam_instance
        # calculate the initial residual for the adjoint before solving
        rArray = np.zeros(self.num_state_elements)
        jacInput = dafoam_instance.getStates()
        seed = dafoam_instance.vec2Array(psi)
        dafoam_instance.solverAD.calcJacTVecProduct(
            "dafoam_solver_states",
            "stateVar",
            jacInput,
            "aero_residuals",
            "residual",
            seed,
            rArray,
        )
        rVec = dafoam_instance.array2Vec(rArray)
        rVec.axpy(-1.0, dFdW)
        # NOTE, this is the norm for the global vec
        rNorm = rVec.norm()

        # read the rTol and aTol from DAOption
        rTol0 = dafoam_instance.getOption("adjEqnOption")["gmresRelTol"]
        aTol0 = dafoam_instance.getOption("adjEqnOption")["gmresAbsTol"]
        # calculate the new absolute tolerance that gives you rTol residual drop
        aTolNew = rNorm * rTol0
        # if aTolNew is smaller than aTol0, assign aTol0 to aTolNew
        if aTolNew < aTol0:
            aTolNew = aTol0
        # assign the atolNew and disable rTol
        ksp.setTolerances(rtol=0.0, atol=aTolNew, divtol=None, max_it=None)


class DAFoamFunctions(csdl.CustomExplicitOperation):
    """Explicit operation evaluating DAFoam functions (CD, CL, ...) from the flow states."""

    def __init__(self, dafoam_instance):
        super().__init__()
        self.dafoam_instance = dafoam_instance

    def evaluate(self, dafoam_solver_states: csdl.Variable, dafoam_input_variables_group: csdl.VariableGroup):
        logger.debug("DAFoamFunctions.evaluate")
        self.declare_input("dafoam_solver_states", dafoam_solver_states)

        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "function" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

        # Read daOptions to get outputs
        dafoam_function_output = csdl.VariableGroup()
        outputDict = self.dafoam_instance.getOption("function")
        for outputName in outputDict.keys():
            setattr(dafoam_function_output, outputName, self.create_output(outputName, (1,)))

        return dafoam_function_output

    def compute(self, input_vals, output_vals):
        logger.debug("DAFoamFunctions.compute")
        dafoam_instance = self.dafoam_instance

        with in_run_directory(dafoam_instance):
            # Update solver states
            dafoam_instance.setStates(input_vals["dafoam_solver_states"])

            funcs = {}
            dafoam_instance.evalFunctions(funcs)

        # Read daOptions to get outputs, and assign them to respective outputs
        outputDict = dafoam_instance.getOption("function")
        for outputName in outputDict.keys():
            output_vals[outputName] = funcs[outputName]

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
        logger.debug("DAFoamFunctions.compute_jacvec_product")
        if mode == "fwd":
            raise NotImplementedError("forward mode has not been implemented for DAFoamFunctions")

        dafoam_instance = self.dafoam_instance

        with in_run_directory(dafoam_instance):
            # Update quantities
            dafoam_instance.setStates(input_vals["dafoam_solver_states"])

            inputDict = dafoam_instance.getOption("inputInfo")
            for functionName in list(d_outputs.keys()):
                seed = d_outputs[functionName]

                # if the seed is zero, do not compute
                if np.all(np.abs(seed) < 1e-12):
                    continue

                for inputName in list(d_inputs.keys()):
                    # compute dFdW * seed
                    if inputName == "dafoam_solver_states":
                        jacInput = input_vals["dafoam_solver_states"]
                        inputType = "stateVar"
                    else:
                        jacInput = input_vals[inputName]
                        inputType = inputDict[inputName]["type"]
                    product = np.zeros_like(jacInput)
                    dafoam_instance.solverAD.calcJacTVecProduct(
                        inputName,
                        inputType,
                        jacInput,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs[inputName] += product
