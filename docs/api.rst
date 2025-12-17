API Reference
=============

This page documents the Python API for BrainChop.

Core API
--------

The core API provides a clean, high-level interface for brain segmentation.

NIfTI Class
~~~~~~~~~~~

.. autoclass:: brainchop.NIfTI
   :members:
   :undoc-members:
   :show-inheritance:

Model Class
~~~~~~~~~~~

.. autoclass:: brainchop.Model
   :members:
   :undoc-members:
   :show-inheritance:

ModelInfo Class
~~~~~~~~~~~~~~~

.. autoclass:: brainchop.ModelInfo
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: brainchop.list_models
.. autofunction:: brainchop.skull_strip
.. autofunction:: brainchop.argmax
.. autofunction:: brainchop.largest_component
.. autofunction:: brainchop.export_channels

Quick Start Examples
--------------------

Basic Segmentation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop import NIfTI, Model, list_models

   # List available models
   for m in list_models():
       print(f"{m.name}: {m.description}")

   # Load and segment
   nifti = NIfTI.load("input.nii.gz")
   model = Model("subcortical")
   result = model.segment(nifti)
   result.save("output.nii.gz")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop import NIfTI, Model

   # Load multiple scans (polymorphic API)
   niftis = NIfTI.load(["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"])

   # Segment with memory-efficient sharding
   model = Model("tissue_fast")
   results = model.segment(niftis, shard_size=2)

   # Save all outputs
   for i, result in enumerate(results):
       result.save(f"output_{i}.nii.gz")

Cropping for Speed
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop import NIfTI, Model

   # Load with cropping (faster inference)
   nifti = NIfTI.load("input.nii.gz", crop_percentile=2.0)

   # Segment - output is automatically restored to full size
   model = Model("tissue_fast")
   result = model.segment(nifti)
   result.save("output.nii.gz")

Custom Models
~~~~~~~~~~~~~

.. code-block:: python

   from brainchop import Model

   # Load custom MeshNet model
   model = Model(
       config_path="/path/to/model.json",
       weights_path="/path/to/model.pth"
   )

   # Load custom Spec model (supports funky layers)
   model = Model(
       config_path="/path/to/spec.json",  # contains "forward_pass" key
       weights_path="/path/to/model.pth"
   )

Raw Inference
~~~~~~~~~~~~~

.. code-block:: python

   from brainchop import NIfTI, Model

   nifti = NIfTI.load("input.nii.gz")
   model = Model("tissue_fast")

   # Get raw output (numpy array, before postprocessing)
   raw_output = model(nifti)  # shape: (B, D, H, W)

   # Access underlying tinygrad model
   tinygrad_model = model.tinygrad_model
   tensor_input = nifti.to_tensor()  # shape: (1, 1, D, H, W)
   raw_tensor = tinygrad_model(tensor_input)

WebGPU Export
~~~~~~~~~~~~~

.. code-block:: python

   # IMPORTANT: Must run with WEBGPU=1 env var
   # Example: WEBGPU=1 python my_script.py

   from brainchop import Model

   model = Model("tissue_fast")
   code_path, weights_path = model.export(output_dir="./web_model")
   print(f"Exported to {code_path} and {weights_path}")

Command Line Interface
----------------------

.. automodule:: brainchop.cli
   :members:
   :undoc-members:
   :show-inheritance:

Utilities Module
----------------

.. automodule:: brainchop.utils
   :members:
   :undoc-members:
   :show-inheritance:

Model Management
~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.utils.list_models
.. autofunction:: brainchop.utils.update_models
.. autofunction:: brainchop.utils.find_pth_files

Download and Cache
~~~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.utils.download
.. autofunction:: brainchop.utils.download_model_listing
.. autofunction:: brainchop.utils.load_models

Image Processing
~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.utils.crop_to_cutoff
.. autofunction:: brainchop.utils.pad_to_original_size
.. autofunction:: brainchop.utils.export_classes

Model Loaders
-------------

TinyGrad MeshNet Loader
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: brainchop.tiny_meshnet
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: brainchop.tiny_meshnet.load_meshnet

Model Types (Spec Format)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: brainchop.types
   :members:
   :undoc-members:
   :show-inheritance:

NIfTI Math Operations
---------------------

.. automodule:: brainchop.niimath
   :members:
   :undoc-members:
   :show-inheritance:

Conformation and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.niimath.conform

Labeling Operations
~~~~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.niimath.bwlabel
.. autofunction:: brainchop.niimath.set_header_intent_label

Mask Operations
~~~~~~~~~~~~~~~

.. autofunction:: brainchop.niimath.grow_border

Constants and Configuration
---------------------------

Available Models
~~~~~~~~~~~~~~~~

.. autodata:: brainchop.utils.AVAILABLE_MODELS
   :annotation:

   Dictionary containing all available segmentation models with their metadata.

   Each model entry contains:

   * **folder**: Model storage directory name
   * **description**: Human-readable model description
   * **considerations**: Normalization requirements

Model URLs
~~~~~~~~~~

.. autodata:: brainchop.utils.BASE_URL
   :annotation: = "https://github.com/neuroneural/brainchop-models/raw/main/"

.. autodata:: brainchop.utils.MESHNET_BASE_URL
   :annotation: = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"

.. autodata:: brainchop.utils.MODELS_JSON_URL
   :annotation: = "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"
