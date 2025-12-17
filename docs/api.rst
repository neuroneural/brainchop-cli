API Reference
=============

brainchop provides a minimal Python API with 4 functions and 1 dataclass.

Quick Start
-----------

.. code-block:: python

   from brainchop import load, segment, save, list_models

   # List models
   print(list_models())

   # Load, segment, save
   vol = load("input.nii.gz")
   result = segment(vol, "subcortical")
   save(result, "output.nii.gz")

Types
-----

Volume
~~~~~~

.. code-block:: python

   @dataclass
   class Volume:
       data: Tensor    # (256, 256, 256) uint8
       header: bytes   # 352-byte NIfTI header

A brain volume with its NIfTI header. Returned by ``load()`` and ``segment()``.

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

   load(path: str, *, crop: float | None = None, ct: bool = False) -> Volume

Load NIfTI file, conform to 256x256x256.

- ``path``: Path to NIfTI file
- ``crop``: Percentile cutoff for cropping (faster inference)
- ``ct``: Convert CT scans from Hounsfield to Cormack units

Returns a ``Volume`` with data Tensor ``(256,256,256)`` and header bytes.

segment
~~~~~~~

.. code-block:: python

   segment(
       volume: Volume | list[Volume],
       model: str,
       shard_size: int = 1,
   ) -> Volume | list[Volume]

Segment brain volume(s).

- ``volume``: Single ``Volume`` or list of ``Volume``s
- ``model``: Model name (e.g., ``"subcortical"``) or path to custom model directory
- ``shard_size``: Batch size for processing multiple volumes

Returns segmented ``Volume``(s) - single if input was single, list if input was list.

Custom models can be loaded by path:

.. code-block:: python

   # By name (from registry)
   result = segment(vol, "subcortical")

   # By path (custom model directory with model.json + model.pth)
   result = segment(vol, "/path/to/my_model")
   result = segment(vol, ".")  # current directory

   # By file:// URI
   result = segment(vol, "file://~/models/custom")

save
~~~~

.. code-block:: python

   save(volume: Volume, path: str) -> None

Save volume to NIfTI file.

- ``volume``: ``Volume`` to save
- ``path``: Output path (``.nii`` or ``.nii.gz``)
