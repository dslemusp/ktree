import numpy as np
import pytest
import random
import yaml
from ktree.k_types import Pose, Rotation, Transformation, Vector
from ktree.ktree import KinematicsTree
from ktree.models import KinematicsConfig
from loguru import logger
from pathlib import Path
from pydantic import ValidationError
from pytest_loguru.plugin import caplog  # noqa: F401
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


def homogeneous_translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])


def homogeneous_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    r = R.from_rotvec(angle * np.array(axis))
    return r.as_matrix()


def multiply_homogeneous_transformations(matrix1: np.ndarray, matrix2: np.ndarray) -> None:
    return np.dot(matrix1, matrix2)


def test_translation_multiplication() -> None:
    matrix1 = Transformation(parent="A", child="B", pose=[2, 3, 4, 0, 0, 0])
    matrix2 = Transformation(parent="B", child="C", pose=[5, 6, 7, 0, 0, 0])

    expected_result = homogeneous_translation_matrix(7, 9, 11)

    result = matrix1 * matrix2

    assert np.allclose(result.hmatrix, expected_result, atol=1e-5)


def test_rotation_multiplication() -> None:
    matrix1 = Transformation(parent="A", child="B", pose=[0, 0, 0, np.deg2rad(20), np.deg2rad(10), np.deg2rad(45)])
    matrix2 = Transformation(parent="B", child="C", pose=[0, 0, 0, np.deg2rad(30), 0, 0])

    expected_result = np.eye(4)
    expected_result[:3, :3] = (
        R.from_euler("xyz", [20, 10, 45], degrees=True) * R.from_euler("xyz", [30, 0, 0], degrees=True)
    ).as_matrix()

    result = matrix1 * matrix2
    # result = multiply_homogeneous_transformations(matrix1, matrix2)

    assert np.allclose(result.hmatrix, expected_result, rtol=1e-3)


def test_translation_and_rotation_multiplication() -> None:
    matrix1 = Transformation(parent="A", child="B", pose=[1, 2, 3, np.deg2rad(20), np.deg2rad(10), np.deg2rad(45)])
    matrix2 = Transformation(parent="B", child="C", pose=[4, 5, 6, np.deg2rad(30), 0, 0])

    expected_result = np.eye(4)
    expected_result[:3, :3] = (
        R.from_euler("xyz", [20, 10, 45], degrees=True) * R.from_euler("xyz", [30, 0, 0], degrees=True)
    ).as_matrix()
    expected_result[:3, 3:] = (
        matrix1.pose.translation.vector + matrix1.pose.rotation.matrix @ matrix2.pose.translation.vector
    ).reshape(3, 1)
    result = matrix1 * matrix2
    # result = multiply_homogeneous_transformations(matrix1, matrix2)

    assert np.allclose(result.hmatrix, expected_result, rtol=1e-3)


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


def test_yaml_dump() -> None:
    kc = KinematicsConfig.parse(Path("./test/config.yaml"))
    kt = KinematicsTree(config=kc)
    kt._remove_transformation("cam", "conn")
    with open("./test/config_dump.yaml", "w") as f:
        yaml.dump(kt.model_dump(), f)
    kc_dump = KinematicsConfig.parse(Path("./test/config_dump.yaml"))
    kt_dump = KinematicsTree(config=kc_dump)
    kt = KinematicsTree(config=kc)
    for transformation in kt_dump._get_all_transformations():
        assert transformation in kt._get_all_transformations()


def test_multiple_paths_warning(caplog) -> None:  # type: ignore # noqa: F811
    kc = KinematicsConfig.parse(Path("./test/config.yaml"))
    kt = KinematicsTree(config=kc)
    kt._add_transformation(Transformation(parent="yaskawa_base", child="cam", pose=Pose()))
    kt.get_transformation(parent="yaskawa_base", child="cam")

    assert "Multiple paths" in caplog.text
