# Bundled data

Everything here ships inside the wheel and is reached through `csdl_dafoam.cases`
(`geometry_path()`, `copy_case()`), not by path.

| Path | What |
| --- | --- |
| `geometry/naca0012_unit_span.stp` | NACA 0012 as a STEP surface, chord 1 m along x, thickness along z, span 0.1 m along y. Imported by `lsdo_geo`. |
| `cases/naca0012/` | OpenFOAM case for `DARhoSimpleCFoam` (compressible, Spalart-Allmaras, wall function). Mesh: 4032 cells, one cell thick, 126 faces on the `wing` patch, far field about 18 chords away. Patches: `wing` (wall), `inout` (far field), `symmetry1`/`symmetry2` (span ends). Flow along +x. `system/decomposeParDict` is set for 4 ranks; `copy_case(..., nprocs=N)` rewrites it. |
| `cases/naca0012_euler_overlay/` | Files laid over `cases/naca0012/` to make the "Euler mode" case: constant viscosity `mu` reduced from 1.8e-5 to 1.8e-9 kg/(m s) and the SA seed `nuTilda`/`nut` scaled by the same factor 1e-4. Reynolds number based on chord goes from about 1.5e7 to 1.5e11. |

Provenance: the geometry and the RANS case are carried over unchanged from the original
`csdl_dafoam` repository (`folder_geo/example_geometries/airfoil_transonic_unitspan_2.stp`
and `folder_cfd/openfoam_naca0012/`). Removed: stale `controlDict~` and a `blockMeshDict`
that does not describe this mesh. `system/createPatchDict` (kept for reference) is the patch-naming step of a
`plot3dToFoam` -> `autoPatch` -> `createPatch` mesh pipeline; the mesh generator itself is not part of the
original repository.
