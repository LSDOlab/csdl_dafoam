"""Sphinx configuration for the csdl_dafoam documentation.

The API reference is generated statically by sphinx-autoapi (it parses the source and imports nothing), so the docs build on a
machine with none of the heavy runtime stack installed (DAFoam, PETSc, MPI, CSDL, JAX).
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import csdl_dafoam  # noqa: E402  (importable without DAFoam, CSDL, MPI or PETSc)

project = "csdl_dafoam"
author = "LSDOlab"
copyright = "LSDOlab"
release = csdl_dafoam.__version__

extensions = [
    "myst_parser",
    "autoapi.extension",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["colon_fence", "deflist", "html_image"]
myst_heading_anchors = 3

autoapi_dirs = ["../src/csdl_dafoam"]
autoapi_type = "python"
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_options = ["members", "undoc-members", "show-module-summary", "imported-members"]
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_keep_files = False

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_title = f"csdl_dafoam {release}"
html_static_path = ["_static"]
