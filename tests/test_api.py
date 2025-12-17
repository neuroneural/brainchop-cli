"""Tests for brainchop API."""

import pytest
from pathlib import Path
from tinygrad import Tensor
from tinygrad.helpers import fetch

from brainchop import load, save, segment, list_models


TEST_URL = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"


@pytest.fixture(scope="module")
def test_nifti_path() -> Path:
    return Path(fetch(TEST_URL, "t1_crop.nii.gz"))


class TestListModels:
    def test_returns_dict(self):
        models = list_models()
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_contains_expected_models(self):
        models = list_models()
        assert "tissue_fast" in models
        assert "subcortical" in models
        assert "mindgrab" in models


class TestLoad:
    def test_load_returns_tensor_and_header(self, test_nifti_path):
        volume, header = load(str(test_nifti_path))
        assert isinstance(volume, Tensor)
        assert volume.shape == (256, 256, 256)
        assert isinstance(header, bytes)

    def test_load_with_crop(self, test_nifti_path):
        volume, header = load(str(test_nifti_path), crop=2.0)
        assert volume.shape[0] <= 256
        assert volume.shape[1] <= 256
        assert volume.shape[2] <= 256


class TestSegment:
    def test_segment_returns_tensor(self, test_nifti_path):
        volume, header = load(str(test_nifti_path))
        result = segment(volume, "tissue_fast")
        assert isinstance(result, Tensor)
        assert result.shape == (256, 256, 256)

    def test_segment_with_header(self, test_nifti_path):
        volume, header = load(str(test_nifti_path))
        result = segment(volume, "tissue_fast", header)
        assert not isinstance(result, list)
        assert result.shape == (256, 256, 256)


class TestSave:
    def test_save_creates_file(self, test_nifti_path, tmp_path):
        volume, header = load(str(test_nifti_path))
        output_path = tmp_path / "output.nii.gz"
        save(volume, header, str(output_path))
        assert output_path.exists()


class TestEndToEnd:
    def test_full_pipeline(self, test_nifti_path, tmp_path):
        volume, header = load(str(test_nifti_path))
        result = segment(volume, "tissue_fast", header)
        assert not isinstance(result, list)
        output_path = tmp_path / "segmented.nii.gz"
        save(result, header, str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
