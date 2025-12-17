API Reference
=============

brainchop provides a minimal Python API with 4 functions.

Quick Start
-----------

.. code-block:: python

   from brainchop import load, segment, save, list_models

   # List models
   print(list_models())

   # Load, segment, save
   volume, header = load("input.nii.gz")
   result = segment(volume, "subcortical", header)
   save(result, header, "output.nii.gz")

Functions
---------

list_models
~~~~~~~~~~~

.. code-block:: python

   list_models() -> dict[str, str]

Returns available models as ``{name: description}``.

load
~~~~

.. code-block:: python

   load(path: str, *, crop: float | None = None, ct: bool = False) -> tuple[np.ndarray, bytes]

Load NIfTI file, conform to 256x256x256.

- ``path``: Path to NIfTI file
- ``crop``: Percentile cutoff for cropping (faster inference)
- ``ct``: Convert CT scans from Hounsfield to Cormack units

Returns ``(volume, header)`` where volume is uint8 ``(256,256,256)``.

segment
~~~~~~~

.. code-block:: python

   segment(
       volume: np.ndarray | list[np.ndarray],
       model: str,
       header: bytes | list[bytes] | None = None,
       shard_size: int = 1,
   ) -> np.ndarray | list[np.ndarray]

Segment brain volume(s).

- ``volume``: Single volume ``(256,256,256)`` uint8 or list of volumes
- ``model``: Model name (e.g., ``"subcortical"``) or path to custom model directory
- ``header``: Optional header(s) for bwlabel postprocessing
- ``shard_size``: Batch size for processing multiple volumes

Returns segmented volume(s) - single array if input was single, list if input was list.

Custom models can be loaded by path:

.. code-block:: python

   # By name (from registry)
   result = segment(volume, "subcortical")

   # By path (custom model directory with model.json + model.pth)
   result = segment(volume, "/path/to/my_model")
   result = segment(volume, ".")  # current directory

   # By file:// URI
   result = segment(volume, "file://~/models/custom")

save
~~~~

.. code-block:: python

   save(volume: np.ndarray, header: bytes, path: str) -> None

Save volume with header to NIfTI file.

- ``volume``: Volume to save
- ``header``: NIfTI header bytes
- ``path``: Output path (``.nii`` or ``.nii.gz``)
