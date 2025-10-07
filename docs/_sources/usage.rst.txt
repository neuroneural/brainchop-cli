Usage Guide
===========

This guide covers the various ways to use BrainChop for brain segmentation.

Basic Usage
-----------

The simplest way to use BrainChop:

.. code-block:: bash

   brainchop input.nii.gz -o output.nii.gz

This uses the default model (tissue_fast) for fast tissue segmentation.

Selecting Models
----------------

List Available Models
~~~~~~~~~~~~~~~~~~~~~

To see all available segmentation models:

.. code-block:: bash

   brainchop --list

Available models include:

* **tissue_fast**: Fast tissue segmentation (gray/white matter)
* **subcortical**: Robust subcortical segmentation for clinical scans
* **DKatlas**: Desikan-Killiany Atlas with 104 labels
* **mindgrab**: Tissue extraction/skull stripping

Using a Specific Model
~~~~~~~~~~~~~~~~~~~~~~

Specify a model with the ``-m`` flag:

.. code-block:: bash

   brainchop input.nii.gz -m DKatlas -o output.nii.gz

Skull Stripping
~~~~~~~~~~~~~~~

Quick skull stripping with the ``--skull-strip`` flag:

.. code-block:: bash

   brainchop input.nii.gz --skull-strip -o brain.nii.gz

This is an alias for ``-m mindgrab`` and extracts just the brain tissue.

Advanced Options
----------------

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files at once:

.. code-block:: bash

   brainchop file1.nii.gz file2.nii.gz file3.nii.gz -m subcortical

Output files are automatically named as ``{input}_{model}_output_{index}.nii.gz``

Control batch size for memory efficiency:

.. code-block:: bash

   brainchop *.nii.gz -m DKatlas --batch-size 4

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

BrainChop can optimize models for faster inference using BEAM compilation:

.. code-block:: bash

   # First run prompts for optimization
   brainchop input.nii.gz -m tissue_fast

   # Skip optimization prompt
   brainchop input.nii.gz --no-optimize

The optimization is cached per model and batch size for future runs.

Image Preprocessing
~~~~~~~~~~~~~~~~~~~

**Cropping for Speed**

Crop the input to speed up processing (may reduce accuracy):

.. code-block:: bash

   brainchop input.nii.gz --crop -o output.nii.gz

Specify a custom percentile cutoff:

.. code-block:: bash

   brainchop input.nii.gz --crop 5 -o output.nii.gz

**CT Scan Conversion**

Convert CT scans from Hounsfield to Cormack units:

.. code-block:: bash

   brainchop ct_scan.nii.gz --ct -o output.nii.gz

**Inverse Conformation**

Transform output back to original image space:

.. code-block:: bash

   brainchop input.nii.gz -i -o output.nii.gz

Export Options
~~~~~~~~~~~~~~

**Export Class Probability Maps**

Export individual probability maps for each segmentation class:

.. code-block:: bash

   brainchop input.nii.gz --export-classes -o output.nii.gz

This creates separate files: ``output_c0.nii``, ``output_c1.nii``, etc.

**Export Brain Mask (Mindgrab)**

When using mindgrab, optionally save the brain mask:

.. code-block:: bash

   brainchop input.nii.gz -m mindgrab --mask mask.nii.gz -o brain.nii.gz

Control mask border growth:

.. code-block:: bash

   brainchop input.nii.gz -m mindgrab --border 2 --mask mask.nii.gz -o brain.nii.gz

Custom Models
-------------

Use your own trained models:

.. code-block:: bash

   brainchop input.nii.gz --custom /path/to/model_dir -o output.nii.gz

The model directory must contain:

* ``model.json``: Model architecture configuration
* ``model.pth`` or ``model.bin``: Model weights

Local Model Development
~~~~~~~~~~~~~~~~~~~~~~~

Load a local model from the current directory:

.. code-block:: bash

   brainchop input.nii.gz -m . -o output.nii.gz

This looks for ``model.json`` and ``model.pth`` in the current directory.

File URI Loading
~~~~~~~~~~~~~~~~

Load models from file URIs:

.. code-block:: bash

   brainchop input.nii.gz -m file:///path/to/model -o output.nii.gz
   brainchop input.nii.gz -m file://~/models/custom -o output.nii.gz

Updating Models
---------------

Update the model listing from the repository:

.. code-block:: bash

   brainchop --update

This downloads the latest models.json from the GitHub repository.

Common Workflows
----------------

Quick Tissue Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Fast gray/white matter segmentation
   brainchop brain.nii.gz -m tissue_fast -o tissue.nii.gz

Clinical Scan Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Robust segmentation for clinical scans
   brainchop clinical.nii.gz -m subcortical -o segmented.nii.gz

Atlas-based Parcellation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Desikan-Killiany atlas parcellation (104 labels)
   brainchop brain.nii.gz -m DKatlas -o parcellation.nii.gz

Brain Extraction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Extract brain, get mask, then segment
   brainchop t1.nii.gz --skull-strip --mask mask.nii.gz -o brain.nii.gz
   brainchop brain.nii.gz -m DKatlas -o parcellation.nii.gz

Batch Processing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Process all NIfTI files in a directory
   brainchop /data/*.nii.gz -m subcortical --batch-size 8

Environment Variables
---------------------

**BEAM Optimization**

Set the BEAM value manually for compilation optimization:

.. code-block:: bash

   BEAM=2 brainchop input.nii.gz -o output.nii.gz

Higher BEAM values enable more aggressive optimization but may increase compilation time.

Troubleshooting
---------------

Out of Memory
~~~~~~~~~~~~~

If you run out of memory:

1. Reduce batch size: ``--batch-size 1``
2. Enable cropping: ``--crop``
3. Process files individually

Slow First Run
~~~~~~~~~~~~~~

The first run with a new model/batch size triggers compilation. This is normal and cached for subsequent runs.

Model Download Issues
~~~~~~~~~~~~~~~~~~~~~

If models fail to download:

1. Check internet connection
2. Try updating: ``brainchop --update``
3. Download manually from `GitHub repository <https://github.com/neuroneural/brainchop-models>`_

For more help, see the API reference or visit the GitHub repository.