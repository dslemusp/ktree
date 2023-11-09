import numpy as np
import pytest
import random
from ktree.k_types import Rotation, Vector
from ktree.ktree import KinematicsTree
from ktree.models import KinematicsConfig
from loguru import logger
from pathlib import Path
from pydantic import ValidationError
from scipy.spatial.transform import Rotation as R


def test_vector() -> None:
    test_list = [1.0, 2.0, 3.0]
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
    assert Vector(vector=[1.234, 2.234, 3.234]) == Vector(vector=[1.234, 2.234, 3.234])


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

    assert Rotation(rpy=[1.2, 1.3, 1.4]) == Rotation(rpy=[1.2, 1.3, 1.4])


def test_mult() -> None:
    # random 3x3 orthonormal matrix
    rotation1 = Rotation(rpy=[0.12, 0.23, 0.34])
    # random 3x3 orthonormal matrix
    rotation2 = Rotation(rpy=[0.1, 0.2, 0.3])
    # random 3x1 vector
    vector = Vector(vector=[1, 2, 3])

    x_vector = Vector(vector=[1, 0, 0])
    y_vector = Vector(vector=[0, 1, 0])

    assert np.allclose(rotation1.matrix @ rotation2.matrix @ vector.vector, ((rotation1 * rotation2) * vector).vector)
    assert np.allclose((x_vector @ y_vector).vector, np.array([0, 0, 1]))


def test_config_load() -> None:
    k_tree_config = KinematicsConfig.parse(Path("./test/config.yaml"))
    assert k_tree_config.base == "yaskawa_base"
    assert k_tree_config.transformations[0].pose.translation.x == 0.001
    assert k_tree_config.transformations[0].pose.translation.y == 0.002
    assert k_tree_config.transformations[0].pose.translation.z == 0.003
    assert k_tree_config.transformations[0].pose.rotation.rx == np.pi


def test_k_chain() -> None:
    kc = KinematicsConfig.parse(Path("./test/config.yaml"))
    kt = KinematicsTree(config=kc)
    end_eff_in_base = kt.get_transformation(parent=kc.base, child=kc.end_effector)
    logger.info(f"End effector in base: {end_eff_in_base}")
    assert np.allclose(end_eff_in_base.pose.translation.vector, np.array([0.008, -0.006, -0.006]))
    # end_eff_in_base = k_tree.get_transformation(parent=k_config.base, child=k_config.end_effector)

    # assert np.allclose(end_eff_in_base.pose.translation.vector, np.array([0.022, 0.0022, 0.024]))
