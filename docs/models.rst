Available Models
================

BrainChop supports multiple pre-trained segmentation models, each optimized for different tasks.

Model Overview
--------------

.. list-table:: Model Comparison
   :header-rows: 1
   :widths: 20 50 30

   * - Model Name
     - Description
     - Normalization
   * - ``tissue_fast``
     - ⚡ our fastest tissue segmentation model that produces gray and white matter labels
     - min/max normalization
   * - ``subcortical``
     - 🪓 a robust model that can handle clinical scans and other difficult input volumes producing cortical tissue segmentation accompanied by 15 subcortical regions
     - min/max normalization
   * - ``subcortical-mini``
     - 🪓 a smaller version of subcortical that has only 21 channels instead of 30
     - min/max normalization
   * - ``DKatlas``
     - 🔪 segments the brain into the Desikan-Killiany Atlas with 104 labels
     - min/max normalization
   * - ``mindgrab``
     - tissue extraction
     - quantile normalization
   * - ``aparc50``
     - 🧠 cortical parcellation model with 50 anatomical regions
     - quantile normalization (5%-95%)

Detailed Model Information
--------------------------

tissue_fast
~~~~~~~~~~~

⚡ our fastest tissue segmentation model that produces gray and white matter labels

**Details:**

* **Model Folder:** ``model5_gw_ae``
* **Normalization:** min/max normalization
* **CLI Parameter:** ``tissue_fast``

**Usage Example:**

.. code-block:: bash

   brainchop input.nii.gz -m tissue_fast -o output.nii.gz

subcortical
~~~~~~~~~~~

🪓 a robust model that can handle clinical scans and other difficult input volumes producing cortical tissue segmentation accompanied by 15 subcortical regions

**Details:**

* **Model Folder:** ``subcortical``
* **Normalization:** min/max normalization
* **CLI Parameter:** ``subcortical``

**Usage Example:**

.. code-block:: bash

   brainchop input.nii.gz -m subcortical -o output.nii.gz

subcortical-mini
~~~~~~~~~~~~~~~~

🪓 a smaller version of subcortical that has only 21 channels instead of 30

**Details:**

* **Model Folder:** ``model18cls``
* **Normalization:** min/max normalization
* **CLI Parameter:** ``subcortical-mini``

**Usage Example:**

.. code-block:: bash

   brainchop input.nii.gz -m subcortical-mini -o output.nii.gz

DKatlas
~~~~~~~

🔪 segments the brain into the Desikan-Killiany Atlas with 104 labels

**Details:**

* **Model Folder:** ``model21_104class``
* **Normalization:** min/max normalization
* **CLI Parameter:** ``DKatlas``

**Usage Example:**

.. code-block:: bash

   brainchop input.nii.gz -m DKatlas -o output.nii.gz

mindgrab
~~~~~~~~

tissue extraction

**Details:**

* **Model Folder:** ``mindgrab``
* **Normalization:** quantile normalization
* **CLI Parameter:** ``mindgrab``

**Usage Example:**

.. code-block:: bash

   brainchop input.nii.gz -m mindgrab -o output.nii.gz

aparc50
~~~~~~~

🧠 cortical parcellation model with 50 anatomical regions

**Details:**

* **Model Folder:** ``model30chan50cls``
* **Normalization:** quantile normalization (5%-95%)
* **CLI Parameter:** ``aparc50``

**Usage Example:**

.. code-block:: bash

   brainchop input.nii.gz -m aparc50 -o output.nii.gz

Model Sources
-------------

All models are automatically downloaded from the BrainChop model repository:

* **GitHub Repository:** https://github.com/neuroneural/brainchop-models
* **Base URL:** https://github.com/neuroneural/brainchop-models/raw/main/meshnet/

Models are cached locally in ``~/.cache/brainchop/models/`` after first download.

Updating Models
~~~~~~~~~~~~~~~

To update the model listing:

.. code-block:: bash

   brainchop --update

Model Architecture Formats
---------------------------

BrainChop supports two model architecture formats:

**New Architecture Format (.pth weights)**

* Uses PyTorch-style weight format
* Modern JSON-based architecture description
* Better performance and flexibility
* Recommended for new models

**Legacy Architecture Format (.bin weights)**

* Uses TensorFlow.js weight format
* Legacy JSON architecture description
* Maintained for backward compatibility

The architecture format is automatically detected based on the model configuration.
