import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from gains.utils.misc import _downscale_data


def make_input_h5(path: Path) -> None:
    """Helper to create sample data file for testing."""
    rng = np.random.default_rng()
    data = rng.random((4, 3), dtype=np.float64)
    
    with h5py.File(path, "w") as f:
        g = f.create_group("tasks")

        g.create_dataset(
            "a",
            data=data,
            chunks=(2, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )

        g.create_dataset(
            "b",
            data=np.array([1.5, 2.5, 3.5], dtype=np.float64),
        )


def test_downscale_data_structure(tmp_path: Path) -> None:
    """Test new file has the same data structure as the input file."""
    src = tmp_path / "input.h5"
    tmp = tmp_path / "temp.h5"

    make_input_h5(src)

    _downscale_data(src, tmp)

    with h5py.File(src, "r") as f:
        assert "tasks" in f
        assert set(f["tasks"].keys()) == {"a", "b"}


def test_downscale_data_dtype(tmp_path: Path) -> None:
    """Test the new file stores data as float32 format."""
    src = tmp_path / "input.h5"
    tmp = tmp_path / "temp.h5"

    make_input_h5(src)
    _downscale_data(src, tmp)

    with h5py.File(src, "r") as f:
        assert f["tasks/a"].dtype == np.float32
        assert f["tasks/b"].dtype == np.float32


def test_downscale_data_values(tmp_path: Path) -> None:
    """Test new data corresponds to the same numerical value as the original."""
    src = tmp_path / "input.h5"
    tmp = tmp_path / "temp.h5"

    make_input_h5(src)

    with h5py.File(src, "r") as f:
        a_orig = f["tasks/a"][:]
        b_orig = f["tasks/b"][:]

    _downscale_data(src, tmp)

    with h5py.File(src, "r") as f:
        np.testing.assert_allclose(f["tasks/a"][:], a_orig)
        np.testing.assert_allclose(f["tasks/b"][:], b_orig)


@pytest.fixture
def metadata() -> list[str]:
    """h5 metadata that should be preserved."""
    return [
        "chunks",
        "compression",
        "compression_opts",
        "shuffle",
        "fletcher32",
        "scaleoffset",
        "fillvalue",
    ]


def test_metadata(tmp_path: Path, metadata: list[str]) -> None:
    """Confirm metadata is preserved after downscaling the data."""
    src = tmp_path / "input.h5"
    tmp = tmp_path / "temp.h5"
    ref = tmp_path / "reference.h5"
    make_input_h5(src)
    shutil.copy(src, ref)

    _downscale_data(src, tmp)

    with h5py.File(ref, "r") as fref, h5py.File(src, "r") as ftest:
        for field in metadata:
            assert getattr(fref["tasks/a"], field) == getattr(ftest["tasks/a"], field)
            assert getattr(fref["tasks/b"], field) == getattr(ftest["tasks/b"], field)
