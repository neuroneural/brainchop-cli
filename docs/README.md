# BrainChop Documentation

This directory contains the Sphinx documentation for BrainChop.

## Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r docs/requirements.txt
```

## Building the Documentation

### Quick Build

From the `docs/` directory:

```bash
make html
```

Or from the project root:

```bash
cd docs && make html
```

### Clean Build

To rebuild from scratch:

```bash
make clean html
```

## Viewing Locally

After building, open the documentation in your browser:

```bash
# macOS
open _build/html/index.html

# Linux
xdg-open _build/html/index.html

# Or start a local server
python -m http.server 8000 -d _build/html
# Then visit http://localhost:8000
```

## Live Preview (Auto-reload)

For development with auto-reload on changes:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then visit http://localhost:8000

## Documentation Structure

```
docs/
├── conf.py           # Sphinx configuration
├── index.rst         # Main page
├── installation.rst  # Installation guide
├── usage.rst         # Usage guide (CLI + Python API)
├── models.rst        # Available models (auto-generated)
├── api.rst           # API reference
├── requirements.txt  # Doc build dependencies
├── Makefile          # Build commands
└── generate_models.py # Auto-generates models.rst from models.json
```

## Adding New Pages

1. Create a new `.rst` file (e.g., `advanced.rst`)
2. Add it to the `toctree` in `index.rst`:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   models
   api
   advanced  # new page
```

3. Rebuild: `make html`

## Deployment

The built HTML files in `_build/html/` can be deployed to any static hosting:

- **GitHub Pages**: Copy `_build/html/*` to `gh-pages` branch
- **Read the Docs**: Connect the repo, RTD builds automatically
- **Netlify/Vercel**: Point to `docs/_build/html` as publish directory
