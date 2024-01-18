import yaml
from ktree.k_types import Transformation
from pathlib import Path
from pydantic import BaseModel, Field, model_serializer


class KinematicsConfig(BaseModel):
    # pass
    base: str = Field(alias="base_frame")
    end_effector: str = Field(alias="end_effector_frame")
    transformations: list[Transformation] = Field(alias="kinematics_chain")

    @staticmethod
    def parse(config_file: Path) -> "KinematicsConfig":
        with open(config_file, "r") as f:
            contents = yaml.safe_load(f)
        return KinematicsConfig.model_validate(contents if contents is not None else {})

    @model_serializer
    def serialize(self) -> dict:
        return dict(
            base=self.base,
            end_effector=self.end_effector,
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
                for transformation in self.transformations
            ],
        )
