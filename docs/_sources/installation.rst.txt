Installation
============

Requirements
------------

* Python 3.10 or higher
* pip package manager

BrainChop requires the following core dependencies (automatically installed):

* **tinygrad**: Tiny and portable ML inference engine
* **numpy**: Basic tensor operations
* **requests**: Model downloading

Installation Methods
--------------------

Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install BrainChop is via pip:

.. code-block:: bash

   pip install brainchop

This will install the latest stable version from PyPI.

From Source
~~~~~~~~~~~

To install from source (for development or latest features):

.. code-block:: bash

   git clone https://github.com/neuroneural/brainchop-cli.git
   cd brainchop-cli
   pip install -e .

Using Docker
~~~~~~~~~~~~

BrainChop is also available as a Docker image:

.. code-block:: bash

   git clone https://github.com/neuroneural/brainchop-cli.git
   cd brainchop-cli
   docker build -t brainchop .

To run with Docker:

.. code-block:: bash

   docker run --rm -it --device=nvidia.com/gpu=all \
     -v /path/to/output:/app \
     brainchop input.nii.gz -o output.nii.gz

.. note::
   On some systems (like recent 25.05 nixos), you may need to prepend the docker run 
   command with ``--device=nvidia.com/gpu=all`` for GPU acceleration.

Verification
------------

Verify your installation by checking the version:

.. code-block:: bash

   brainchop --help

This should display the help message with available options.

GPU Support
-----------

BrainChop automatically detects and uses available GPUs when present. The tinygrad 
backend supports:

* **NVIDIA GPUs**: CUDA backend
* **AMD GPUs**: ROCm backend  
* **Apple Silicon**: Metal backend
* **Intel GPUs**: OpenCL backend

No additional configuration is needed - GPU acceleration is automatic when available.

Troubleshooting
---------------

If you encounter issues during installation:

1. **Python version**: Ensure you're using Python 3.10 or higher
2. **Virtual environment**: Consider using a virtual environment to avoid conflicts
3. **Dependencies**: If pip fails, try updating pip: ``pip install --upgrade pip``

For more help, visit the `GitHub Issues <https://github.com/neuroneural/brainchop-cli/issues>`_ page.