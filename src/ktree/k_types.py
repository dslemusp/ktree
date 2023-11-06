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


def _validate_list(v: NDArray[np.float64] | list[float]) -> NDArray:
    if isinstance(v, list):
        return np.array(v)
    return v



class Pose(BaseModel):
    x: float = Field(default=0.0, description="X translation of origin of child in meters in parent frame")
    y: float = Field(default=0.0, description="Y translation of origin of child in meters in parent frame")
    z: float = Field(default=0.0, description="Z translation of origin of child in meters in parent frame")
    rx: float = Field(default=0.0, description="Roll rotation in radians")
    ry: float = Field(default=0.0, description="Pitch rotation in radians")
    rz: float = Field(default=0.0, description="Yaw rotation in radians")
    
    def from_list(self, pose: list[float]) -> "Pose":
        return Pose(x=pose[0], y=pose[1], z=pose[2], rx=pose[3], ry=pose[4], rz=pose[5])


class Vector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    vector: NDArray[np.float64] = Field(default=np.array([0.0, 0.0, 0.0]), min_length=3, max_length=3)

    # validators
    _vector_validator = field_validator("vector", mode="before")(_validate_list)

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
        if isinstance(other, self.__class__):
            return Vector(vector=self.vector * other.vector)
        if isinstance(other, (int, float)):
            return Vector(vector=self.vector * other)
        raise ValueError(f"Cannot multiply Vector with {other}")

    def __str__(self) -> str:
        return f"x: {self.x*1000:.3f} mm, y: {self.y*1000:.3f} mm, z: {self.z*1000:.3f} mm"


class JointType(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    SPATIAL = "spatial"


class JointAxis(Enum):
    X = 0
    Y = 1
    Z = 2


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


class ConfigPose(BaseModel):
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float = Field(default=0.0, le=180, gt=-180)
    ry_deg: float = Field(default=0.0, le=180, gt=-180)
    rz_deg: float = Field(default=0.0, le=180, gt=-180)

    def to_pose(self) -> NDArray:
        pose = np.zeros(6)
        pose[:3] = np.array([self.x_mm, self.y_mm, self.z_mm]) / 1000
        pose[3:] = np.deg2rad([self.rx_deg, self.ry_deg, self.rz_deg])
        return pose


class Rotation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rpy: NDArray[np.float64] = Field(default=np.array([0.0, 0.0, 0.0]), min_length=3, max_length=3)

    # validators
    _rpy_validator = field_validator("rpy", mode="before")(_validate_list)

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

    def __mult__(self, other: "Rotation") -> NDArray:
        return self.matrix @ other.matrix

    def __str__(self) -> str:
        return (
            f"rx: {np.rad2deg(self.rx):.1f} deg, ry: {np.rad2deg(self.ry):.1f} deg, rz: {np.rad2deg(self.rz):.1f} deg"
        )


class Transformation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    pose: NDArray[np.float64] = Field(
        default=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        description=(
            "List or array in SI units or ConfigPose/dict with values in mm/deg. Pose contains 3 values for position"
            " and 3 values for the euler angles following the roll-pitch-yaw convention."
        ),
        min_length=6,
        max_length=6,
    )
    parent: str = Field(..., description="Parent frame. The definition of the pose is wrt to this frame")
    child: str = Field(
        ...,
        description="Child frame. The pose defines the origin and orientation of the child wrt to the parent",
    )
    joint: Joint = Field(default=Joint(), description="Joint connecting parent and child")

    @field_validator("pose", mode="before")
    def _pose_validator(cls, v: NDArray[np.float64] | list[float] | ConfigPose | Pose) -> NDArray:
        if isinstance(v, ConfigPose):
            return v.to_pose()
        if isinstance(v, dict):
            return ConfigPose(**v).to_pose()
        if isinstance(v, Pose):
            return np.array(v.list())
        if isinstance(v, list):
            return np.array(v)
        return v

    @computed_field  # type: ignore[misc]
    @property
    def translation(self) -> Vector:
        return Vector(vector=self.pose[:3])

    @translation.setter
    def translation(self, value: Vector) -> None:
        self.pose[:3] = value.vector

    @computed_field  # type: ignore[misc]
    @property
    def rotation(self) -> Rotation:
        return Rotation(rpy=self.pose[3:])

    @rotation.setter
    def rotation(self, value: Rotation) -> None:
        self.pose[3:] = value.rpy

    @computed_field  # type: ignore[misc]
    @property
    def hmatrix(self) -> NDArray:
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = self.rotation.matrix
        homogeneous_matrix[:3, 3] = self.translation.vector
        return homogeneous_matrix

    @hmatrix.setter
    def hmatrix(self, matrix: NDArray) -> None:
        new_rot = Rotation()
        new_rot.matrix = matrix[:3, :3]
        self.rotation = new_rot
        self.translation = Vector(vector=matrix[:3, 3])

    def to_pose(self) -> Pose:
        return Pose(x=self.translation.x, y=self.translation.y, z=self.translation.z, rx=self.rotation.rx, ry=self.rotation.ry, rz=self.rotation.rz)

    def inv(self) -> "Transformation":
        """
        Returns the inverse transformation.
        """
        new_rot = Rotation()
        new_rot.matrix = self.rotation.matrix.T

        new_pos = -new_rot.matrix @ self.translation.vector

        return Transformation(
            pose=np.concatenate([new_pos, new_rot.rpy]),
            parent=self.child,
            child=self.parent,
        )

    def reset_rotation(self) -> "Transformation":
        """
        Returns the transformation with zero rotation.
        """
        self.rotation = Rotation()
        return self

    def reset_translation(self) -> "Transformation":
        """
        Returns the transformation with zero translation.
        """
        self.translation = Vector()
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
            return self.parent == other.parent and self.child == other.child
        else:
            return False

    def __str__(self) -> str:
        return f"{self.child.upper()} in {self.parent.upper()} Pose({self.translation}, {self.rotation})"

    def __hash__(self) -> int:
        return id(self)
