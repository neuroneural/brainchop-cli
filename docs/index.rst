BrainChop
=========

Portable brain segmentation powered by tinygrad.

Install
-------

.. code-block:: bash

   pip install brainchop

Usage
-----

**CLI:**

.. code-block:: bash

   brainchop input.nii.gz -o output.nii.gz
   brainchop input.nii.gz -m subcortical -o output.nii.gz
   brainchop input.nii.gz --skull-strip -o brain.nii.gz
   brainchop --list  # show available models

**Python:**

.. code-block:: python

   from brainchop import load, segment, save

   vol = load("input.nii.gz")
   result = segment(vol, "subcortical")
   save(result, "output.nii.gz")

Models
------

.. list-table::
   :widths: 20 80

   * - ``tissue_fast``
     - Fast gray/white matter segmentation (default)
   * - ``subcortical``
     - Cortical + 15 subcortical regions, robust to clinical scans
   * - ``DKatlas``
     - Desikan-Killiany atlas (104 labels)
   * - ``mindgrab``
     - Skull stripping / brain extraction
   * - ``aparc50``
     - Cortical parcellation (50 regions)

Run ``brainchop --list`` for the full list.

.. toctree::
   :hidden:

   api
