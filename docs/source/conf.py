import re

from nukelm import __version__ as release


# -- Project information -----------------------------------------------------

project = "nukelm"
author = "PNNL"

# The short X.Y version.
groups = re.search(r"^([0-9]+\.[0-9]+).*", release)
if groups is not None:
    version = groups.group(1)
else:
    raise ValueError("Could not parse version")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.doctest",
    # "sphinx.ext.todo",
    # "sphinx.ext.coverage",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.ifconfig",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.mathjax",
    # "sphinx_autodoc_typehints",
    # "nbsphinx",
    # "recommonmark",
    # "sphinx_copybutton",
]

# enable cross-linking to other documentation
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3', None),
#     'transformers': ('https://huggingface.co/transformers/', None),
# }


# enable autosummary plugin (table of contents for modules/classes/class
# methods)
# autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
