from src.ktree.k_types import Vector, Rotation
import numpy as np
import pytest
from pydantic import ValidationError
import random
from scipy.spatial.transform import Rotation as R


def test_vector() -> None:
    test_list = [1, 2, 3]
    v = Vector(vector=test_list)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3

    # Set values
    v.x = 4
    assert v.x == 4

    # Test fail instantiation
    for my_fail_list in [[1, 2], [1, 2, 3, 4]]:
        with pytest.raises(ValidationError) as val_err:
            Vector(vector=my_fail_list)
            assert "List should have exactly 3 items after validation" in val_err.value

    # Test unit vector
    v = Vector(vector=[10, 0, 0])
    assert np.allclose(v.to_unit().vector, np.array([1, 0, 0]))


def test_rotation() -> None:
    """Test rotation conversion functions from euler angles to matrix and vice versa.
    Uses scipy.spatial.transform.Rotation as reference."""

    # Test with random angles except singularities
    list_rx = [random.uniform(-89, 89) for _ in range(1000)]
    list_ry = [random.uniform(-89, 89) for _ in range(1000)]
    list_rz = [random.uniform(-89, 89) for _ in range(1000)]

    for rx_i, ry_i, rz_i in zip(list_rx, list_ry, list_rz):
        rx = np.deg2rad(rx_i)
        ry = np.deg2rad(ry_i)
        rz = np.deg2rad(rz_i)

        baseR = R.from_euler("xyz", [rx, ry, rz], degrees=False)

        r = Rotation(rpy=[rx, ry, rz])

        assert np.allclose(r.matrix, baseR.as_matrix())
        r = Rotation()
        r.matrix = baseR.as_matrix()

        assert np.isclose(r.rx, rx)
        assert np.isclose(r.ry, ry)
        assert np.isclose(r.rz, rz)

def test_mult():
    #random 3x3 orthonormal matrix
    rotation1 = Rotation(rpy = [0.12, 0.23, 0.34])
    #random 3x3 orthonormal matrix
    rotation2 = Rotation(rpy = [0.1, 0.2, 0.3])
    #random 3x1 vector
    vector = Vector(vector = [1, 2, 3])
    
    assert np.allclose(rotation1.matrix @ rotation2.matrix @ vector.vector, ((rotation1 * rotation2) * vector).vector)
    
    