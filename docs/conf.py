# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory to the path so we can import brainchop
sys.path.insert(0, os.path.abspath('..'))
# Add the docs directory to the path so we can import generate_models
sys.path.insert(0, os.path.abspath('.'))

# Project information
project = 'brainchop'
copyright = '2024, Mike Doan'
author = 'Mike Doan'

# The full version, including alpha/beta/rc tags
release = '0.1.23'
version = '0.1.23'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for Google/NumPy docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx_autodoc_typehints',  # Better type hints support
]

# Napoleon settings for Google/NumPy docstring style
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Template configuration
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# The master toctree document
master_doc = 'index'

# Auto-generate model documentation from models.json
def setup(app):
    """Run custom setup tasks when Sphinx initializes."""
    # Generate models documentation from models.json
    from generate_models import generate_models_rst
    generate_models_rst()