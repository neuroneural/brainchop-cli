API Reference
=============

brainchop provides a minimal Python API with 5 functions.

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

   segment(volume: np.ndarray, model: str, header: bytes | None = None) -> np.ndarray

Segment brain volume.

- ``volume``: Input volume ``(256,256,256)`` uint8
- ``model``: Model name (e.g., ``"subcortical"``, ``"tissue_fast"``)
- ``header``: Optional header for bwlabel postprocessing

Returns segmented volume ``(256,256,256)`` uint8.

segment_batch
~~~~~~~~~~~~~

.. code-block:: python

   segment_batch(
       volumes: list[np.ndarray],
       model: str,
       headers: list[bytes] | None = None,
       shard_size: int = 1,
   ) -> list[np.ndarray]

Segment multiple volumes with optional sharding for memory control.

save
~~~~

.. code-block:: python

   save(volume: np.ndarray, header: bytes, path: str) -> None

Save volume with header to NIfTI file.

- ``volume``: Volume to save
- ``header``: NIfTI header bytes
- ``path``: Output path (``.nii`` or ``.nii.gz``)
