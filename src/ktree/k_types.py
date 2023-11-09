import numpy as np
from enum import Enum
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

M_SUFFIX = "_m"
RAD_SUFFIX = "_rad"
MM_SUFFIX = "_mm"
DEG_SUFFIX = "_deg"
X = "x"
Y = "y"
Z = "z"
RX = "rx"
RY = "ry"
RZ = "rz"
POSE = [X, Y, Z, RX, RY, RZ]


def _validate_list(v: NDArray[np.float_] | list[float]) -> NDArray:
    if isinstance(v, list):
        return np.array(v)
    return v


class Vector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    vector: NDArray[np.float_] = Field(default=np.array([0.0, 0.0, 0.0]), min_length=3, max_length=3)

    # validators
    @field_validator("vector", mode="before")
    @classmethod
    def _vector_validation(cls, v: NDArray[np.float_] | list[float]) -> NDArray:
        return _validate_list(v)

    @computed_field  # type: ignore[misc]
    @property
    def x(self) -> float:
        return self.vector[0]

    @x.setter
    def x(self, value: float) -> None:
        self.vector[0] = value

    @computed_field  # type: ignore[misc]
    @property
    def y(self) -> float:
        return self.vector[1]

    @y.setter
    def y(self, value: float) -> None:
        self.vector[1] = value

    @computed_field  # type: ignore[misc]
    @property
    def z(self) -> float:
        return self.vector[2]

    @z.setter
    def z(self, value: float) -> None:
        self.vector[2] = value

    def to_unit(self) -> "Vector":
        return Vector(vector=self.vector / np.linalg.norm(self.vector))

    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(vector=self.vector + other.vector)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(vector=self.vector - other.vector)

    def __mul__(self, other: float | object) -> "Vector":
        """Pairwise multiplication of vectors or multiplication by scalar"""
        if isinstance(other, self.__class__):
            return Vector(vector=self.vector * other.vector)
        if isinstance(other, (int, float)):
            return Vector(vector=self.vector * other)
        raise ValueError(f"Cannot multiply Vector with {other}")

    def __matmul__(self, other: "Vector") -> "Vector":
        """Cross product of vectors"""
        return Vector(vector=np.cross(self.vector, other.vector))

    def __str__(self) -> str:
        return f"x: {self.x*1000:.3f} mm, y: {self.y*1000:.3f} mm, z: {self.z*1000:.3f} mm"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return np.allclose(self.vector, other.vector)
        else:
            raise NotImplementedError(f"Cannot compare Vector with {other}")


class JointType(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    SPATIAL = "spatial"


class Rotation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rpy: NDArray[np.float_] = Field(default=np.array([0.0, 0.0, 0.0]), min_length=3, max_length=3)

    # validators
    # _rpy_validator = field_validator("rpy", mode="before")(_validate_list)
    @field_validator("rpy", mode="before")
    @classmethod
    def _rpy_validator(cls, v: NDArray[np.float_] | list[float]) -> NDArray:
        return _validate_list(v)

    @staticmethod
    def rot_x(angle: float) -> NDArray:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def rot_y(angle: float) -> NDArray:
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    @staticmethod
    def rot_z(angle: float) -> NDArray:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    @computed_field  # type: ignore[misc]
    @property
    def rx(self) -> float:
        return self.rpy[0]

    @rx.setter
    def rx(self, value: float) -> None:
        self.rpy[0] = value

    @computed_field  # type: ignore[misc]
    @property
    def ry(self) -> float:
        return self.rpy[1]

    @ry.setter
    def ry(self, value: float) -> None:
        self.rpy[1] = value

    @computed_field  # type: ignore[misc]
    @property
    def rz(self) -> float:
        return self.rpy[2]

    @rz.setter
    def rz(self, value: float) -> None:
        self.rpy[2] = value

    @computed_field  # type: ignore[misc]
    @property
    def matrix(self) -> NDArray:
        return self.rot_z(self.rpy[2]) @ self.rot_y(self.rpy[1]) @ self.rot_x(self.rpy[0])

    @matrix.setter
    def matrix(self, matrix: NDArray) -> None:
        """
        Interprets the provided matrix as a 3x3 rotation matrix and returns Euler angles in radians to be applied as
        individual sequential rotations about axes x, y and then z (extrinsic).
        """
        # See: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

        # check orthogonality
        assert np.allclose(matrix @ matrix.T, np.eye(3))

        sy = -matrix[2, 0]
        if np.isclose(sy, 1.0):
            rx = np.arctan2(matrix[0, 1], matrix[0, 2])
            ry = 0.5 * np.pi
            rz = 0.0
        elif np.isclose(sy, -1.0):
            rx = np.arctan2(-matrix[0, 1], -matrix[0, 2])
            ry = -0.5 * np.pi
            rz = 0.0
        else:
            cy_inv = 1.0 / np.sqrt(1.0 - sy * sy)  # 1 / cos(ry)
            rx = np.arctan2(matrix[2, 1] * cy_inv, matrix[2, 2] * cy_inv)
            ry = np.arcsin(sy)
            rz = np.arctan2(matrix[1, 0] * cy_inv, matrix[0, 0] * cy_inv)

        self.rpy = np.array([rx, ry, rz])

    def __mul__(self, other: Self | Vector | NDArray[np.float_]) -> "Vector | Rotation":
        if isinstance(other, Vector):
            return Vector(vector=self.matrix @ other.vector)
        if isinstance(other, self.__class__):
            rot = Rotation()
            rot.matrix = self.matrix @ other.matrix
            return rot
        if isinstance(other, np.ndarray):
            if other.shape == (3,):
                return Vector(vector=self.matrix @ other)
            elif other.shape == (3, 3):
                rot = Rotation()
                rot.matrix = self.matrix @ other
                return rot

        raise NotImplementedError(f"Cannot multiply Rotation with {other}")

    def __str__(self) -> str:
        return (
            f"rx: {np.rad2deg(self.rx):.1f} deg, ry: {np.rad2deg(self.ry):.1f} deg, rz: {np.rad2deg(self.rz):.1f} deg"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return np.allclose(self.rpy, other.rpy)
        else:
            raise NotImplementedError(f"Cannot compare Rotation with {other}")


class Pose(BaseModel):
    translation: Vector = Field(default=Vector(), description="Translation vector in SI units")
    rotation: Rotation = Field(default=Rotation(), description="Rotation matrix in SI units or roll pitch yaw angles")

    @classmethod
    def from_list(cls, pose: NDArray[np.float_] | list[float]) -> "Pose":
        return cls(translation=Vector(vector=pose[:3]), rotation=Rotation(rpy=pose[3:]))  # type: ignore[arg-type]

    def to_list(self) -> list[float]:
        return list(self.translation.vector) + list(self.rotation.rpy)

    def __str__(self) -> str:
        return f"Pose({self.translation}, {self.rotation})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.translation == other.translation and self.rotation == other.rotation
        else:
            raise NotImplementedError(f"Cannot compare Pose with {other}")


class JointAxis(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"


class Joint(BaseModel):
    type: JointType = Field(default=JointType.FIXED, description="Degree of freedom type of the joint")
    axis: JointAxis | None = Field(
        default=None, description="If `type` is other than FIXED, axis of rotation or translation (x, y or z)"
    )

    @model_validator(mode="after")
    def _axis_validator(self) -> "Joint":
        match self.type:
            case JointType.FIXED | JointType.SPATIAL:
                self.axis = None
        return self

    @computed_field  # type: ignore[misc]
    @property
    def vector(self) -> Vector | None:
        match self.axis:
            case JointAxis.X:
                return Vector(vector=np.array([1.0, 0.0, 0.0]))
            case JointAxis.Y:
                return Vector(vector=np.array([0.0, 1.0, 0.0]))
            case JointAxis.Z:
                return Vector(vector=np.array([0.0, 0.0, 1.0]))
            case _:
                return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.type == other.type and self.axis == other.axis
        else:
            raise NotImplementedError(f"Cannot compare Joint with {other}")


class Transformation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    pose: Pose = Field(
        default=Pose(),
        description=(
            "List or ndarray in SI units or dict with values with units specified in each field as suffix of the type"
            "_mm, _m, _rad, _deg. Pose contains 3 values for position and 3 values for the euler angles following the"
            " roll-pitch-yaw convention."
        ),
    )
    parent: str = Field(..., description="Parent frame. The definition of the pose is wrt to this frame")
    child: str = Field(
        ...,
        description="Child frame. The pose defines the origin and orientation of the child wrt to the parent",
    )
    joint: Joint = Field(default=Joint(), description="Joint connecting parent and child")

    @field_validator("pose", mode="before")
    @classmethod
    def _pose_validator(cls, v: NDArray[np.float_] | list[float]) -> Pose:
        match v:
            case list() | np.ndarray():
                return Pose.from_list(v)
            case dict():
                pose_dict = dict()
                for key, value in v.items():
                    if key.endswith(MM_SUFFIX):
                        pose_dict[key.replace(MM_SUFFIX, M_SUFFIX)] = value / 1000
                    elif key.endswith(DEG_SUFFIX):
                        pose_dict[key.replace(DEG_SUFFIX, RAD_SUFFIX)] = np.deg2rad(value)
                    elif key.endswith(RAD_SUFFIX) | key.endswith(M_SUFFIX):
                        pose_dict[key] = value
                    else:
                        raise ValueError(f"Invalid key {key} in pose dict. Check config file.")
                return Pose.from_list(
                    [
                        pose_dict[X + M_SUFFIX],
                        pose_dict[Y + M_SUFFIX],
                        pose_dict[Z + M_SUFFIX],
                        pose_dict[RX + RAD_SUFFIX],
                        pose_dict[RY + RAD_SUFFIX],
                        pose_dict[RZ + RAD_SUFFIX],
                    ]
                )
        return v

    @computed_field  # type: ignore[misc]
    @property
    def hmatrix(self) -> NDArray:
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = self.pose.rotation.matrix
        homogeneous_matrix[:3, 3] = self.pose.translation.vector
        return homogeneous_matrix

    @hmatrix.setter
    def hmatrix(self, matrix: NDArray) -> None:
        new_rot = Rotation()
        new_rot.matrix = matrix[:3, :3]
        self.pose.rotation = new_rot
        self.pose.translation = Vector(vector=matrix[:3, 3])

    def inv(self) -> "Transformation":
        """
        Returns the inverse transformation.
        """
        new_rot = Rotation()
        new_rot.matrix = self.pose.rotation.matrix.T

        new_pos = -new_rot.matrix @ self.pose.translation.vector

        return Transformation(
            pose=Pose(translation=Vector(vector=new_pos), rotation=new_rot),
            parent=self.child,
            child=self.parent,
        )

    def reset_rotation(self) -> "Transformation":
        """
        Returns the transformation with zero rotation.
        """
        self.pose.rotation = Rotation()
        return self

    def reset_translation(self) -> "Transformation":
        """
        Returns the transformation with zero translation.
        """
        self.pose.translation = Vector()
        return self

    def __mul__(self, other: "Transformation") -> "Transformation":
        if self.child != other.parent:
            raise ValueError(
                "Cannot multiply transformations with non-matching child/parent."
                f" {self.parent.upper()}_T_{self.child.upper()} <-X->"
                f" {other.parent.upper()}_T_{other.child.upper()}"
            )

        mult_transformation = Transformation(parent=self.parent, child=other.child)
        mult_transformation.hmatrix = self.hmatrix @ other.hmatrix

        return mult_transformation

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.parent == other.parent and self.child == other.child and self.pose == other.pose
        else:
            return False

    def __str__(self) -> str:
        return f"{self.child.upper()} in {self.parent.upper()} Pose({self.pose.translation}, {self.pose.rotation})"

    def __hash__(self) -> int:
        return id(self)
