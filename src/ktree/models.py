import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from src.ktree.k_types import Transformation


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
