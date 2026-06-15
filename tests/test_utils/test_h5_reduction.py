from gains.utils.misc import _downscale_data
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import h5py

import h5py
import numpy as np


def make_input_h5(path):
    """Helper to create sample data file for testing."""
    with h5py.File(path, "w") as f:
        g = f.create_group("tasks")

        g.create_dataset(
            "a",
            data=np.random.rand(4, 3).astype(np.float64),
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


def test_downscale_data_structure(tmp_path):
    """Test new file has the same data structure as the input file."""
    src = tmp_path / "input.h5"
    tmp = tmp_path / "temp.h5"

    make_input_h5(src)

    _downscale_data(src, tmp)

    with h5py.File(src, "r") as f:
        assert "tasks" in f
        assert set(f["tasks"].keys()) == {"a", "b"}

def test_downscale_data_dtype(tmp_path):
    """Test the new file stores data as float32 format."""
    src = tmp_path / "input.h5"
    tmp = tmp_path / "temp.h5"

    make_input_h5(src)
    _downscale_data(src, tmp)

    with h5py.File(src, "r") as f:
        assert f["tasks/a"].dtype == np.float32
        assert f["tasks/b"].dtype == np.float32

def test_downscale_data_values(tmp_path):
    """Test new data corresponds to the same numerical value as the original"""
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
