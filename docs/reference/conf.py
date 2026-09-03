"""Sphinx configuration for the API reference.

Build: ``uv run --group docs sphinx-build -W docs/reference docs/_build/html``.
``-W`` turns every warning into a failure, so a docstring that is not valid
reStructuredText, or a cross-reference to a name that has moved, fails the
build (and the gate, and CI) rather than rendering wrong. The docstrings
are the source: nothing here is written twice.
"""

import polars_online

project = "polars-online"
author = "Hans Gilde"
release = polars_online.__version__
copyright = "Hans Gilde"  # noqa: A001 -- Sphinx's name for it

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

# Members in source order, so a class reads the way its file does; only
# documented members, so an undocumented helper is absent rather than a
# bare signature. `__init__`'s docstring joins the class's.
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
autoclass_content = "both"
# Types stay in the signature (the docstrings describe parameters in prose,
# not ``:param:`` fields, so moving hints to the description would drop
# them), written short: ``DataFrame``, not ``polars.dataframe.frame.DataFrame``.
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

html_theme = "furo"
html_title = f"polars-online {release}"
html_static_path: list[str] = []
# The design notes live beside this directory as Markdown; Sphinx must not
# try to read them.
exclude_patterns = ["_build"]
