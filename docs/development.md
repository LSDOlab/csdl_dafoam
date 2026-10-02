# Development

```bash
conda env create -f environment.yml && conda activate csdl-dafoam      # full stack
# or, for everything except DAFoam itself:
pip install -r requirements-lab.txt && pip install -e ".[geometry,opt,test,docs]"
pytest
```

## Test strategy

DAFoam cannot run in ordinary CI, but what this package contributes is *wiring*: which arrays go to which
DAFoam call, which transpose is applied, which seed multiplies what. Wiring errors give plausible-looking
numbers and wrong gradients, so the tests are built to make the right answer known exactly.

`tests/fakes.py` provides stand-ins for DAFoam, IDWarp and PETSc with **linear** physics:

- flow residual `R(w; x, v, p, T) = A w - (Bx x + Bv v + Bp p + BT T)` and functions `F = c.w + d.x + e.v`, so
  the states, function values and every total derivative are available in closed form;
- an identity or random linear mesh warp.

| Test module | What it establishes |
| --- | --- |
| `test_solver.py` | primal, and reverse-mode totals through the implicit operation, match the closed form for **both** the fixed-point and Krylov adjoints; DAFoam calls run in the case directory; failure policies |
| `test_mesh_warp.py` | the Fortran-IDWarp wrapper: warp and its adjoint (the Jacobian equals the warp matrix) |
| `test_jax_warp.py` | **real IDWarp-JAX on the real bundled mesh**: wall follows the prescribed surface, no inverted cells (signed cell volumes), VJP vs finite differences, decomposed-mesh point matching, seed accumulation across ranks |
| `test_inputs.py`, `test_options.py` | input mapping, Mach-to-airspeed, unsupported types, force scaling |
| `test_atmosphere.py` | ISA (with the 300 K sea-level reference) and Sutherland formulas |
| `test_geometry.py` | real `lsdo_geo`: STEP import and cache (keyed by file content, corrupt cache regenerated), projection of the real CFD wall mesh, FFD thickness/camber response, the camber-normalization quirk |
| `test_airfoil_model.py` | the whole `build_airfoil_model` pipeline for **both warpers** (real geometry, FFD, atmosphere, real IDWarp-JAX, CSDL graph with its MPI region, JAX backend, modOpt **OpenSQP** and **PySLSQP**) against the linear fake, including reverse-mode derivatives vs finite differences through geometry, warp and flow |
| `test_examples.py` | the example scripts, end to end, on the fake solver, for each warper/optimizer combination |
| `test_cases.py`, `test_foam_io.py` | bundled data integrity, safe case copying (never deletes a directory it did not create) |
| `test_import.py`, `test_doctor.py` | importable without the heavy stack; helpful errors; the doctor |
| `test_openfoam_mesh.py` | **real OpenFOAM v2506**: IDWarp-JAX meshes through `checkMesh` with DAFoam's criteria, volumes cross-checked against `foam_io.cell_volumes`, a tangled mesh rejected. Marked `openfoam`; skipped without an OpenFOAM (`brew install --cask gerlero/openfoam/openfoam@2506` on macOS) |
| `test_viz.py` | the flow-solution reader on a **real 4-rank DAFoam snapshot of the coarse mesh (100 KB)** (reassembly on the global mesh, deformed points, ordering of the surface distribution) and on synthetic serial fields; plots and the VTK export |
| `test_coarse_mesh.py` | the bundled 336-cell mesh equals what `scripts/make_coarse_airfoil_mesh.py` generates (no drift), its topology and geometry, its wall lies on the STEP surface, OpenFOAM accepts it |
| `test_vm_scripts.py` | static guards on the build and VM scripts: no host mounts (an earlier version exposed the home directory), provisioning present, same package list as the prerequisites script, every download checksummed, pinned sources, stage order |
| `test_dafoam_integration.py` | **real DAFoam**: sane CL/CD, adjoint vs finite differences. Marked `dafoam`; skipped when DAFoam is not importable |

The first run of a test that "passes" is worth a mutation check: while writing `test_solver.py`, flipping the sign of
the residual Jacobian-vector product and dropping the adjoint hand-back were both caught.

Run the real-DAFoam tests inside an environment where DAFoam works (the VM, see {doc}`install-mac`). They use the 336-cell test mesh by default,
so they take seconds; `CSDL_DAFOAM_TEST_CASE=naca0012_euler` uses the 4,032-cell mesh of the examples:

```bash
pytest -m dafoam
mpirun -np 4 python -m pytest -m dafoam -p no:cacheprovider
```

The whole suite takes about a minute and a half; `-m "not slow"` takes about 25 seconds. For a quick loop, `pytest -m "not slow"` skips the handful of tests that take several seconds each
(the example scripts and finite-difference checks); CI runs everything.

## Continuous integration

- `.github/workflows/tests.yml`: pure-pip job (no DAFoam) running the suite on Python 3.9 and 3.12, a wheel check, a docs build, and static checks of the recipes.
- `.github/workflows/pages.yml`: builds the docs (warnings are errors) and publishes them to GitHub Pages on every push to `main` that touches `docs/`, `src/` or `pyproject.toml`.
- `.github/workflows/conda-release.yml`: builds `recipe/recipe.yaml` on a release and publishes `csdl_dafoam`.
- `.github/workflows/conda-stack.yml`: manual dispatch that builds `recipes/*` in dependency order and, with
  `publish: true`, uploads them to the `lsdolab` channel. A full run is several hours.

## Docs

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html      # then open docs/_build/html/index.html
```

The API reference is generated by sphinx-autoapi from the source without importing it, so the docs build with none of the heavy stack installed.
The site is published at <https://lsdolab.github.io/csdl_dafoam/> by `pages.yml`; someone with admin rights must set **Settings, Pages, Source:
"GitHub Actions"** once.

The videos live in `docs/_static/` (`optimization.mp4` for the docs, `optimization.gif` for the README, which GitHub cannot play as a video).
To regenerate them, and the figures in `docs/images/`, run `examples/airfoil_optimization.py --maxiter 100` on a DAFoam machine (about 37 minutes) and
copy `history.png`, `movie.gif`, `movie.mp4` from the working directory.

## Releasing

1. bump `__version__` in `src/csdl_dafoam/__init__.py` and `context.version` in `recipe/recipe.yaml`;
2. `pytest` (and `pytest -m dafoam` in the conda environment);
3. publish a GitHub release; `conda-release.yml` builds and uploads the package.
