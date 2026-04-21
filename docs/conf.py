"""Sphinx configuration for sw2023 documentation."""

import os
import sys

# Make the package importable without installing
sys.path.insert(0, os.path.abspath('..'))

# ── Project information ───────────────────────────────────────────────────────
project   = 'sw2023'
author    = 'Choonjoo Lee'
copyright = '2026, Choonjoo Lee'
release   = '0.3.2'
version   = '0.3.2'

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    'sphinx.ext.autodoc',       # API docs from docstrings
    'sphinx.ext.napoleon',      # NumPy/Google style docstrings
    'sphinx.ext.viewcode',      # source code links
    'sphinx.ext.intersphinx',   # cross-links to numpy/scipy docs
    'sphinx.ext.autosummary',   # summary tables
]

# ── autodoc settings ─────────────────────────────────────────────────────────
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
    'member-order':     'bysource',
}
autodoc_typehints = 'description'

# ── Napoleon (NumPy-style docstrings) ────────────────────────────────────────
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False
napoleon_use_param         = True
napoleon_use_rtype         = True

# ── intersphinx ──────────────────────────────────────────────────────────────
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# ── HTML output ──────────────────────────────────────────────────────────────
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth':    4,
    'collapse_navigation': False,
    'sticky_navigation':   True,
    'titles_only':         False,
}
html_static_path = ['_static']
html_show_sourcelink = True

# ── Misc ─────────────────────────────────────────────────────────────────────
exclude_patterns  = ['_build', 'Thumbs.db', '.DS_Store']
templates_path    = ['_templates']
