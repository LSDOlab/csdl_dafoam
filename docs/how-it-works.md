# How it works

## The pieces

| Module | Provides | Role |
| --- | --- | --- |
| `csdl_dafoam.solver` | `instantiate_dafoam`, `DAFoamSolver`, `DAFoamFunctions` | wraps DAFoam's flow solve (implicit operation) and its functions of interest (explicit operation) |
| `csdl_dafoam.jax_warp` | `JaxIDWarp`, `JaxMeshWarper` | IDWarp-JAX: global surface coordinates to this rank's volume coordinates (default) |
| `csdl_dafoam.mesh_warp` | `DAFoamMeshWarper` | the alternative: the Fortran IDWarp attached to the DAFoam instance |
| `csdl_dafoam.inputs` | `compute_dafoam_input_variables` | builds the CSDL variables DAFoam declares as inputs |
| `csdl_dafoam.atmosphere` | `compute_ambient_conditions_group` | differentiable standard atmosphere |
| `csdl_dafoam.geometry` | `load_geometry`, `setup_airfoil_geometry` | STEP import (cached), thickness/camber FFD parameterization |
| `csdl_dafoam.airfoil` | `build_airfoil_model`, `AirfoilModel` | assembles all of the above for a 2D airfoil |
| `csdl_dafoam.cases`, `options`, `foam_io` | bundled data, option dictionaries, a `polyMesh` reader and cell-volume check | |

`build_airfoil_model` is a convenience; for another geometry or parameterization, call the building blocks in
the same order it does (its source is short and commented).

## Data flow of one evaluation

1. **Design variables** (thickness and camber degrees of freedom, angle of attack) are CSDL variables.
2. `setup_airfoil_geometry` moves the control points of an FFD block around the `lsdo_geo` B-spline surface.
3. `geometry.evaluate(projection)` evaluates that surface at the parametric points where the CFD wall mesh was
   projected, giving the new wall coordinates `x_surf`. The projection is computed once and cached.
4. `JaxMeshWarper` passes `x_surf` to IDWarp-JAX, which deforms the whole volume mesh; each rank keeps its own
   points as `x_vol` ({ref}`idwarp-jax`).
5. `compute_dafoam_input_variables` assembles everything DAFoam declares in `inputInfo`: `x_vol`, the patch
   velocity `[airspeed, angle of attack]`, and far-field pressure and temperature from the atmosphere model.
6. `DAFoamSolver` is an *implicit* operation: it runs DAFoam's SIMPLE solve to drive the OpenFOAM residuals
   `R(w, inputs)` to zero and outputs the flow states `w`.
7. `DAFoamFunctions` maps `(w, inputs)` to CL and CD using DAFoam's force functions.

## Derivatives

Reverse mode only. For the implicit operation, CSDL asks for `(dR/dw)^-T` applied to a vector
(`apply_inverse_jacobian`) and for `(dR/d input)^T` applied to a vector (`compute_jacvec_product`); DAFoam's
adjoint provides both. `adjEqnSolMethod` in the DAFoam options selects the Krylov adjoint or the fixed-point
adjoint; the Krylov path follows DAFoam's own `mphys_dafoam.py` (preconditioner re-computation every
`adjPCLag` iterations, optional dynamic tolerance). Forward mode raises `NotImplementedError`.

The chain through the geometry is differentiated by CSDL, and the mesh warp by its own reverse derivative
(IDWarp-JAX's VJP, or the Fortran IDWarp's `warpDeriv`). The
test suite checks the reverse-mode total derivatives of this whole chain against finite differences, with a
fake linear flow solver ({doc}`development`).

(idwarp-jax)=
## Mesh deformation with IDWarp-JAX

[IDWarp-JAX](https://github.com/LSDOlab/idwarp-jax) is LSDO Lab's JAX implementation of IDWarp (inverse-distance
weighted deformation with local rotations). It differs from the Fortran IDWarp in ways that shape the interface:

- **Global, not distributed.** It deforms the whole reconstructed mesh (`constant/polyMesh`) in one process. The
  moving surface is therefore global too: every rank holds all wall points, in ascending OpenFOAM point order, and
  no surface gathering across ranks is needed. DAFoam's per-rank points are located in the global mesh by an exact
  coordinate match (`match_local_points`), which also handles points duplicated across processor boundaries.
- **Reverse pass.** Each rank scatters its volume seed into a global seed (zeros elsewhere), applies the global VJP, and the
  CSDL MPI region sums the per-rank surface contributions. Every rank warps the full mesh, which is cheap next to
  the CFD solve for the sizes this package targets; it would not be for very large 3D meshes.
- **Approximate by design.** Its kd-tree prunes far interactions to a relative tolerance `err_tol` (5e-4 by default), so
  forward and reverse agree to a few tenths of a percent, not to round-off. Tighten `err_tol` through
  `warper_options` if a gradient check needs it.
- **Symmetry: exact y = 0 only.** The kd-tree warper supports only an exact symmetry plane at y = 0. The bundled case
  (span 0 to 0.1 in y) satisfies that; a case with a different symmetry plane needs `warper="idwarp"`.
- **`LdefFact`** (deformation length factor) defaults to 100 here, IDWarp-JAX's own default. On the bundled mesh it
  kept every cell above 86% of its volume for a perturbation about four times the optimizer's first steps
  (against 69% for `LdefFact=1`), with no inverted cell. Cell quality is checked in `tests/test_jax_warp.py` with
  `csdl_dafoam.foam_io.cell_volumes`, and against OpenFOAM v2506's own `checkMesh` in `tests/test_openfoam_mesh.py`
  (DAFoam's mesh check is a copy of OpenFOAM's geometry checks without the cell-determinant test, which a one-cell-thick
  symmetry-closed 2D mesh fails by construction).
- Far-field points move along with the airfoil (by amounts comparable to the wall motion); IDWarp-JAX carries local
  surface rotations to them. The mesh stays valid, but if the far-field position matters for a case, check it.

The Fortran IDWarp remains available with `build_airfoil_model(..., warper="idwarp")`; it is distributed across ranks
and supports other symmetry configurations.

## Optimizers

`AirfoilModel.optimize(algorithm=...)` builds a modOpt `CSDLAlphaProblem` from the CSDL simulator and runs either
**OpenSQP** (default; needs `qpsolvers`, `quadprog` and `highspy`) or **PySLSQP**. modOpt works in scaled design
variables; after the solve the final design is un-scaled and written back into the simulator, which is re-run there.

## MPI

Run under `mpirun -np N`; the case must be decomposed into N subdomains. Each rank owns part of the CFD mesh
and a slice of the design-surface points. The surface points are gathered on rank 0 for the projection (so no
rank sees an empty element set), the cheap geometry set-up runs one rank at a time to avoid races on
`lsdo_geo`'s on-disk cache, and the warp -> solve -> functions chain runs inside CSDL's `enter_mpi_region`,
which sums the per-rank contributions of the global inputs in the reverse pass. CL and CD are global outputs.

## Working directories

DAFoam's Python layer resolves several paths against the process working directory. Every DAFoam call made
through these operations first enters the case directory that `instantiate_dafoam` recorded as
`instance.run_directory`, and restores the previous directory afterwards, so the surrounding script is free
to be anywhere. Geometry caches go to `cache_dir`, never into the case.

## Failure handling

| Event | Behavior |
| --- | --- |
| DAFoam's mesh check rejects the warped mesh | `DAFoamError` (disable with `check_mesh=False`) |
| primal solve fails | `DAFoamError` |
| adjoint linear solve does not converge | `RuntimeWarning` and continue with the partial adjoint; `on_adjoint_failure="raise"` stops |

The original scripts skipped the mesh check ("until debugging is done") and silently continued after a failed
primal solve, leaving the states unset. Raising is deliberate: a failed solve must not feed garbage into an
optimizer.

## Known quirks

- **Camber is normalized by the FFD control-point spacing, not the chord.** `normalized_percent_camber_change_dof`
  of 5 moves a control point by 5% of the distance between the first two chordwise control points: chord/4 for
  the default five points, so 0.0125 chord, not 0.05. This is inherited from the original code (checked
  numerically: a dof of 5 moved the surface by 0.0125). Rescale your bounds if you change the number of
  control points.
- The three thickness/camber degrees of freedom are shared between both span-wise FFD sections, which keeps the
  airfoil two-dimensional.
- `check_mesh`, the Krylov-vs-fixed-point choice and the adjoint tolerances are DAFoam options, not CSDL ones;
  see the DAFoam documentation for their meaning.
