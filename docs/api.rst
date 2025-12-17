API Reference
=============

.. code-block:: python

   from brainchop import Volume, load, segment, save, list_models

Volume
------

.. code-block:: python

   @dataclass
   class Volume:
       data: Tensor    # (256, 256, 256)
       header: bytes   # NIfTI header

load
----

.. code-block:: python

   load(path: str, *, crop: float | None = None, ct: bool = False) -> Volume

Load and conform NIfTI to 256³. Options: ``crop`` percentile, ``ct`` for Hounsfield conversion.

segment
-------

.. code-block:: python

   segment(volume: Volume | list[Volume], model: str, shard_size: int = 1) -> Volume | list[Volume]

Segment volume(s). Pass a list for batch processing with ``shard_size`` controlling memory.

``model`` can be a name (``"subcortical"``) or path to custom model directory.

save
----

.. code-block:: python

   save(volume: Volume, path: str) -> None

Save to ``.nii`` or ``.nii.gz``.

list_models
-----------

.. code-block:: python

   list_models() -> dict[str, str]

Returns ``{name: description}`` of available models.

Example
-------

.. code-block:: python

   from brainchop import load, segment, save

   # Single volume
   vol = load("input.nii.gz")
   result = segment(vol, "subcortical")
   save(result, "output.nii.gz")

   # Batch
   vols = [load(f"scan{i}.nii.gz") for i in range(4)]
   results = segment(vols, "tissue_fast", shard_size=2)
   for i, r in enumerate(results):
       save(r, f"out_{i}.nii.gz")

   # Custom model
   result = segment(vol, "/path/to/model")
   result = segment(vol, ".")  # current directory
