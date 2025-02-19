import networkx as nx
import numpy as np
from ktree.k_types import JointType, Pose, Transformation, Vector
from ktree.models import KinematicsConfig
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_serializer, validate_call
from typing import Any, cast
from typing_extensions import Self


class KinematicsTree(BaseModel):
    config: KinematicsConfig

    def model_post_init(self, __context: Any) -> None:
        logger.info("Initializing kinematic chain")
        """Create kinematic chain based on parsed configuration"""
        self._k_chain = nx.DiGraph()
        self._joints = [
            t for t in self.config.transformations if t.joint.type != JointType.FIXED
        ]
        self._n_actuated_joints = len(self._joints)

        logger.debug("Kinematic chain nodes".upper())
        for transformation in self.config.transformations:
            logger.debug(transformation)
            self._add_transformation(transformation)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def update_transformation(
        self,
        transformation: Transformation | None = None,
        parent: str | None = None,
        child: str | None = None,
        pose: Pose | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        rx: float | None = None,
        ry: float | None = None,
        rz: float | None = None,
    ) -> Self:
        """Update transformation of child relative to parent. If a transformation is passed the rest of the arguments
        are ignored

        kwargs:
            pose: Pose
            x: float X translation of origin of child in meters in parent frame
            y: float Y translation of origin of child in meters in parent frame
            z: float Z translation of origin of child in meters in parent frame
            rx: float roll rotation in radians
            ry: float pitch rotation in radians
            rz: float yaw rotation in radians
        """

        if transformation is None:
            if parent is None or child is None:
                raise ValueError(
                    "In the absence of a transformation object, parent and child must be specified"
                )

            if not self._k_chain.has_node(parent) or not self._k_chain.has_node(child):
                raise ValueError(
                    f"Parent {parent.upper()} or child {child.upper()} not in kinematic chain ->"
                    f" {[frame.upper() for frame in self._k_chain.nodes]}"
                )

            # Get current transformation
            if not self._k_chain.has_edge(parent, child):
                # New transformation
                transformation = Transformation(parent=parent, child=child)
                # logger.success(f"New transformation {child.upper()} in {parent.upper()}")
            else:
                # Update existing transformation
                transformation = cast(
                    Transformation, self._k_chain.edges[parent, child]["T"]
                )

            # Update transformation
            if pose is not None:
                transformation.pose = pose
            else:
                if x is not None:
                    transformation.pose.translation.x = x
                if y is not None:
                    transformation.pose.translation.y = y
                if z is not None:
                    transformation.pose.translation.z = z
                if rx is not None:
                    transformation.pose.rotation.rx = rx
                if ry is not None:
                    transformation.pose.rotation.ry = ry
                if rz is not None:
                    transformation.pose.rotation.rz = rz

        remove_edges = False
        if not self._k_chain.has_edge(transformation.parent, transformation.child):
            remove_edges = True

        # Update transformation in kinematic chain
        self._add_transformation(transformation)

        # TODO: Add update of the kinematic chain via Jacobian
        # Remove from kinematic chain if the transformation was not in the chain in the first place
        # Prevents the addition of a transformation that is not in the kinematic chain
        if remove_edges:
            self._remove_transformation(
                parent=transformation.parent, child=transformation.child
            )

        return self

    @validate_call
    def get_transformation(self, parent: str, child: str) -> Transformation:
        """Get pose of child relative to parent"""

        sp = cast(list, nx.shortest_path(self._k_chain, source=parent, target=child))
        all_paths = list(
            nx.all_simple_paths(self._k_chain, source=parent, target=child)
        )

        if len(list(all_paths)) > 1:
            logger.warning(
                f"Multiple paths from {parent.upper()} to {child.upper()} -> {all_paths}. Returning shortest path {sp}"
            )

        total_transformation = cast(
            Transformation, self._k_chain.edges[sp[0], sp[1]]["T"]
        )

        for node in sp[1:-1]:
            total_transformation = total_transformation * cast(
                Transformation, self._k_chain.edges[node, sp[sp.index(node) + 1]]["T"]
            )

        return total_transformation

    def _get_jacobian(self) -> NDArray:
        jacobian = np.zeros((6, self._n_actuated_joints), dtype=float)

        # Parse active transformations between base frame and end effector from current kinematic chain
        # joints = [t for t in self.config.transformations if t.joint.type != JointType.FIXED]

        end_effector = self.get_transformation(
            parent=self.config.base, child=self.config.end_effector
        )

        for col_j, joint_j in enumerate(self._joints):
            joint_wt = self.get_transformation(
                parent=self.config.base, child=joint_j.parent
            )
            if joint_j.joint.vector is None:
                raise ValueError(
                    f"Actuated Joint transformation {joint_j.parent.upper()} - {joint_j.child.upper()} has no axis."
                    " Check configuration file"
                )
            match joint_j.joint.type:
                case JointType.PRISMATIC:
                    jacobian[:3, col_j] = joint_wt.pose.rotation * joint_j.joint.vector
                case JointType.REVOLUTE:
                    a_i = cast(Vector, joint_wt.pose.rotation * joint_j.joint.vector)
                    jacobian[:3, col_j] = (
                        a_i
                        @ (end_effector.pose.translation - joint_wt.pose.translation)
                    ).vector
                    jacobian[3:, col_j] = a_i.vector
                case JointType.FIXED:
                    ValueError(f"Joint type {joint_j.joint.type} not accepted")
                case _:
                    NotImplementedError(
                        f"Joint type {joint_j.joint.type} not implemented"
                    )

        # analitical_jacobian
        b_matrix = np.array(
            [
                [1, 0, np.sin(end_effector.pose.rotation.ry)],
                [
                    0,
                    np.cos(end_effector.pose.rotation.rx),
                    -np.cos(end_effector.pose.rotation.ry)
                    * np.sin(end_effector.pose.rotation.rx),
                ],
                [
                    0,
                    np.sin(end_effector.pose.rotation.rx),
                    np.cos(end_effector.pose.rotation.rx)
                    * np.cos(end_effector.pose.rotation.ry),
                ],
            ]
        )
        jacobian_rpy = np.eye(6)
        jacobian_rpy[3:, 3:] = np.linalg.inv(b_matrix)

        return jacobian_rpy @ jacobian

    def _add_transformation(self, transformation: Transformation) -> None:
        self._k_chain.add_edge(
            transformation.parent, transformation.child, T=transformation
        )
        self._k_chain.add_edge(
            transformation.child, transformation.parent, T=transformation.inv()
        )

    def _remove_transformation(self, parent: str, child: str) -> None:
        self._k_chain.remove_edge(parent, child)
        self._k_chain.remove_edge(child, parent)

    def _get_all_transformations(self) -> list[Transformation]:
        return [
            transformation["T"]
            for _, _, transformation in self._k_chain.to_undirected().edges(data=True)
        ]

    @model_serializer
    def serialize(self) -> dict:
        return dict(
            base_frame=self.config.base,
            end_effector_frame=self.config.end_effector,
            kinematics_chain=[
                dict(
                    parent=transformation.parent,
                    child=transformation.child,
                    pose=dict(
                        x_m=float(transformation.pose.translation.x),
                        y_m=float(transformation.pose.translation.y),
                        z_m=float(transformation.pose.translation.z),
                        rx_rad=float(transformation.pose.rotation.rx),
                        ry_rad=float(transformation.pose.rotation.ry),
                        rz_rad=float(transformation.pose.rotation.rz),
                    ),
                )
                for transformation in self._get_all_transformations()
            ],
        )
