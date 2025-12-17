BrainChop Documentation
=======================

BrainChop is a lightweight, portable brain segmentation tool that runs on pretty much everything.
It leverages tinygrad for efficient ML inference and supports multiple segmentation models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   models
   api

Features
--------

* **Lightweight**: Minimal dependencies, runs on CPU and GPU
* **Portable**: Works on Linux, macOS, and Windows
* **Multiple Models**: Support for various segmentation tasks (tissue, subcortical, atlas-based)
* **Fast**: Optimized inference with BEAM compilation
* **Flexible**: Batch processing, custom models, and extensive CLI options
* **API-First**: Clean Python API for scripting, CLI is a thin wrapper

Quick Start
-----------

Install brainchop:

.. code-block:: bash

   pip install brainchop

**Command Line:**

.. code-block:: bash

   brainchop input.nii.gz -o output.nii.gz

**Python API:**

.. code-block:: python

   from brainchop import load, segment, save

   volume, header = load("input.nii.gz")
   result = segment(volume, "subcortical", header)
   save(result, header, "output.nii.gz")

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`