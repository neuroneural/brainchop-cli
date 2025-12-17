"""Tests for brainchop API."""

import pytest
from pathlib import Path
from tinygrad.helpers import fetch

from brainchop import Volume, load, save, segment, list_models


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
    def test_load_returns_volume(self, test_nifti_path):
        vol = load(str(test_nifti_path))
        assert isinstance(vol, Volume)
        assert vol.data.shape == (256, 256, 256)
        assert isinstance(vol.header, bytes)

    def test_load_with_crop(self, test_nifti_path):
        vol = load(str(test_nifti_path), crop=2.0)
        assert vol.data.shape[0] <= 256
        assert vol.data.shape[1] <= 256
        assert vol.data.shape[2] <= 256


class TestSegment:
    def test_segment_returns_volume(self, test_nifti_path):
        vol = load(str(test_nifti_path))
        result = segment(vol, "tissue_fast")
        assert isinstance(result, Volume)
        assert result.data.shape == (256, 256, 256)

    def test_segment_preserves_header(self, test_nifti_path):
        vol = load(str(test_nifti_path))
        result = segment(vol, "tissue_fast")
        assert isinstance(result, Volume)
        assert result.header == vol.header


class TestSave:
    def test_save_creates_file(self, test_nifti_path, tmp_path):
        vol = load(str(test_nifti_path))
        output_path = tmp_path / "output.nii.gz"
        save(vol, str(output_path))
        assert output_path.exists()


class TestEndToEnd:
    def test_full_pipeline(self, test_nifti_path, tmp_path):
        vol = load(str(test_nifti_path))
        result = segment(vol, "tissue_fast")
        assert isinstance(result, Volume)
        output_path = tmp_path / "segmented.nii.gz"
        save(result, str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
