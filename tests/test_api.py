"""
Tests for the brainchop Python API.
"""

import pytest
import numpy as np
from pathlib import Path
from tinygrad.helpers import fetch

from brainchop import NIfTI, Model, list_models, ModelInfo


# Test data
TEST_URL = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
CACHEDIR = Path.home() / ".cache" / "brainchop" / "test_api"
CACHEDIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def test_nifti_path() -> Path:
    """Download and cache test NIfTI file."""
    return Path(fetch(TEST_URL, "t1_crop.nii.gz"))


class TestListModels:
    """Tests for list_models()."""

    def test_returns_list(self):
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_returns_model_info(self):
        models = list_models()
        for m in models:
            assert isinstance(m, ModelInfo)
            assert isinstance(m.name, str)
            assert isinstance(m.description, str)
            assert isinstance(m.folder, str)
            assert m.normalization in ("minmax", "quantile")

    def test_contains_expected_models(self):
        models = list_models()
        names = {m.name for m in models}
        assert "tissue_fast" in names
        assert "subcortical" in names
        assert "mindgrab" in names


class TestNIfTI:
    """Tests for NIfTI class."""

    def test_load_single(self, test_nifti_path):
        nifti = NIfTI.load(str(test_nifti_path))
        assert isinstance(nifti, NIfTI)
        assert isinstance(nifti.volume, np.ndarray)
        assert nifti.volume.shape == (256, 256, 256)
        assert nifti.volume.dtype == np.uint8
        assert isinstance(nifti.header, bytes)
        assert nifti.source_path is not None

    def test_load_list(self, test_nifti_path):
        niftis = NIfTI.load([str(test_nifti_path), str(test_nifti_path)])
        assert isinstance(niftis, list)
        assert len(niftis) == 2
        for n in niftis:
            assert isinstance(n, NIfTI)

    def test_load_with_crop(self, test_nifti_path):
        nifti = NIfTI.load(str(test_nifti_path), crop_percentile=2.0)
        assert isinstance(nifti, NIfTI)
        assert nifti.crop_coords is not None
        # Cropped volume should be smaller or equal
        assert nifti.volume.shape[0] <= 256
        assert nifti.volume.shape[1] <= 256
        assert nifti.volume.shape[2] <= 256

    def test_to_tensor(self, test_nifti_path):
        nifti = NIfTI.load(str(test_nifti_path))
        tensor = nifti.to_tensor()
        assert tensor.shape == (1, 1, 256, 256, 256)

    def test_save(self, test_nifti_path, tmp_path):
        nifti = NIfTI.load(str(test_nifti_path))
        output_path = tmp_path / "output.nii.gz"
        nifti.save(str(output_path))
        assert output_path.exists()


class TestModel:
    """Tests for Model class."""

    def test_load_by_name(self):
        model = Model("tissue_fast")
        assert model.info.name == "tissue_fast"
        assert model.tinygrad_model is not None

    def test_load_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown model"):
            Model("nonexistent_model")

    def test_model_info(self):
        model = Model("tissue_fast")
        info = model.info
        assert isinstance(info, ModelInfo)
        assert info.name == "tissue_fast"

    def test_call_returns_numpy(self, test_nifti_path):
        nifti = NIfTI.load(str(test_nifti_path))
        model = Model("tissue_fast")
        output = model(nifti)
        assert isinstance(output, np.ndarray)
        # Output shape: (B, D, H, W) since PREARGMAX is default
        assert output.shape[0] == 1  # batch size

    def test_segment_single(self, test_nifti_path):
        nifti = NIfTI.load(str(test_nifti_path))
        model = Model("tissue_fast")
        result = model.segment(nifti)
        assert isinstance(result, NIfTI)
        assert result.volume.shape == (256, 256, 256)
        assert result.volume.dtype == np.uint8

    def test_segment_list(self, test_nifti_path):
        niftis = NIfTI.load([str(test_nifti_path), str(test_nifti_path)])
        model = Model("tissue_fast")
        results = model.segment(niftis)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, NIfTI)

    def test_segment_with_shard_size(self, test_nifti_path):
        niftis = NIfTI.load([str(test_nifti_path)] * 4)
        model = Model("tissue_fast")
        results = model.segment(niftis, shard_size=2)
        assert isinstance(results, list)
        assert len(results) == 4

    def test_segment_no_postprocess(self, test_nifti_path):
        nifti = NIfTI.load(str(test_nifti_path))
        model = Model("tissue_fast")
        result = model.segment(nifti, postprocess=False)
        assert isinstance(result, NIfTI)
        # Raw output has float type
        assert result.volume.dtype == np.float32


class TestEndToEnd:
    """End-to-end tests mimicking typical usage."""

    def test_basic_workflow(self, test_nifti_path, tmp_path):
        """Test the basic workflow: load -> segment -> save."""
        # Load
        nifti = NIfTI.load(str(test_nifti_path))

        # Segment
        model = Model("tissue_fast")
        result = model.segment(nifti)

        # Save
        output_path = tmp_path / "segmented.nii.gz"
        result.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_batch_workflow(self, test_nifti_path, tmp_path):
        """Test batch processing workflow."""
        # Load multiple
        paths = [str(test_nifti_path)] * 3
        niftis = NIfTI.load(paths)

        # Segment batch
        model = Model("tissue_fast")
        results = model.segment(niftis, shard_size=2)

        # Save all
        for i, result in enumerate(results):
            output_path = tmp_path / f"output_{i}.nii.gz"
            result.save(str(output_path))
            assert output_path.exists()

    def test_cropped_workflow(self, test_nifti_path, tmp_path):
        """Test workflow with cropping for faster inference."""
        # Load with crop
        nifti = NIfTI.load(str(test_nifti_path), crop_percentile=2.0)

        # Segment
        model = Model("tissue_fast")
        result = model.segment(nifti)

        # Result should be restored to full size
        assert result.volume.shape == (256, 256, 256)

        # Save
        output_path = tmp_path / "cropped_output.nii.gz"
        result.save(str(output_path))
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
