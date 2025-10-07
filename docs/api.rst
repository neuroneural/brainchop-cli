API Reference
=============

This page documents the Python API for BrainChop.

Command Line Interface
----------------------

.. automodule:: brainchop.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

The CLI module provides the main entry point for the command-line tool:

.. autofunction:: brainchop.cli.run_cli

Preprocessing
~~~~~~~~~~~~~

.. autofunction:: brainchop.cli.preprocess_input
.. autofunction:: brainchop.cli.preprocess_batch

Inference
~~~~~~~~~

.. autofunction:: brainchop.cli.run_inference
.. autofunction:: brainchop.cli.run_batch_inference

Postprocessing
~~~~~~~~~~~~~~

.. autofunction:: brainchop.cli.postprocess_output
.. autofunction:: brainchop.cli.postprocess_batch_output

Output
~~~~~~

.. autofunction:: brainchop.cli.write_output
.. autofunction:: brainchop.cli.generate_output_filename

Optimization
~~~~~~~~~~~~

.. autofunction:: brainchop.cli.preoptimize
.. autofunction:: brainchop.cli.load_optimization_cache
.. autofunction:: brainchop.cli.save_optimization_cache
.. autofunction:: brainchop.cli.get_best_beam_for_batch_size

Utilities Module
----------------

.. automodule:: brainchop.utils
   :members:
   :undoc-members:
   :show-inheritance:

Model Management
~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.utils.get_model
.. autofunction:: brainchop.utils.get_model_from_custom_path
.. autofunction:: brainchop.utils.list_models
.. autofunction:: brainchop.utils.update_models

Download and Cache
~~~~~~~~~~~~~~~~~~

.. autofunction:: brainchop.utils.download
.. autofunction:: brainchop.utils.download_model_listing
.. autofunction:: brainchop.utils.load_models

Model Detection
~~~~~~~~~~~~~~~

.. autofunction:: brainchop.utils.detect_architecture_version
.. autofunction:: brainchop.utils.find_pth_files
.. autofunction:: brainchop.utils.find_tfjs_files

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

TensorFlow.js MeshNet Loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: brainchop.tfjs_meshnet
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: brainchop.tfjs_meshnet.load_tfjs_meshnet

Model Types
~~~~~~~~~~~

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
   * **parameter_name**: CLI parameter name

Model URLs
~~~~~~~~~~

.. autodata:: brainchop.utils.BASE_URL
   :annotation: = "https://github.com/neuroneural/brainchop-models/raw/main/"

.. autodata:: brainchop.utils.MESHNET_BASE_URL
   :annotation: = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"

.. autodata:: brainchop.utils.MODELS_JSON_URL
   :annotation: = "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"

Usage Examples
--------------

Loading a Model Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop.utils import get_model
   from tinygrad.tensor import Tensor
   import numpy as np

   # Load a model
   model = get_model("tissue_fast")
   
   # Prepare input (BS, 1, H, W, D)
   input_data = np.random.randn(1, 1, 256, 256, 256).astype(np.float32)
   input_tensor = Tensor(input_data)
   
   # Run inference
   output = model(input_tensor)
   
   # Get segmentation labels
   labels = output.argmax(axis=1).numpy()

Using Custom Models
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop.utils import get_model_from_custom_path
   
   # Load custom model
   model = get_model_from_custom_path(
       config_path="/path/to/model.json",
       weights_path="/path/to/model.pth"
   )
   
   # Use like any other model
   output = model(input_tensor)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop.cli import preprocess_batch, run_batch_inference
   import argparse
   
   # Prepare arguments
   args = argparse.Namespace(
       input=["file1.nii.gz", "file2.nii.gz"],
       crop=False,
       comply=False,
       ct=False
   )
   
   # Preprocess batch
   batched_tensor, volumes, headers, crops = preprocess_batch(args.input, args)
   
   # Run inference
   output = run_batch_inference(model, batched_tensor)

Image Cropping
~~~~~~~~~~~~~~

.. code-block:: python

   from brainchop.utils import crop_to_cutoff, pad_to_original_size
   import numpy as np
   
   # Load your 3D image
   image = np.load("brain_volume.npy")
   
   # Crop to 2nd percentile
   cropped, coords = crop_to_cutoff(image, cutoff_percent=2.0)
   
   # Process cropped image
   processed = some_processing_function(cropped)
   
   # Restore to original size
   restored = pad_to_original_size(processed, coords, original_shape=(256, 256, 256))