# Kinematic Tree

Moved kinematic tree implementation from Pebble into a stand alone library.

## Installation

### As a poetry [^1.6] depedency

```shell
poetry add git+ssh://git@github.com:rocsys/research-ktree.git#0.3.1  # 0.3.1 as example (modify this for different versions)
```

## Usage

The library allows the creation of a static (and soon dynamic) kinematic tree based on pose transformations (Set in a configuration yaml)

```yaml

base_frame: yaskawa_base
end_effector_frame: conn

kinematics_chain:
  - parent: yaskawa_base
    child: yaskawa_eff
    joint:
      axis: x
      type: prismatic
    pose:
      x_mm: 1.0
      y_m: 0.002
      z_mm: 3.0
      rx_deg: 180.0
      ry_rad: 0.0
      rz_deg: 0.0
  - parent: yaskawa_eff
    child: checkered_board
    pose: 
      x_mm: 4.0
      y_mm: 5.0
      z_mm: 6.0
      rx_deg: 0.0
      ry_deg: 0.0
      rz_deg: 0.0
  - parent: cam
    child: checkered_board
    pose: 
      x_mm: 7.0
      y_mm: 8.0
      z_mm: 9.0
      rx_deg: 0.0
      ry_deg: 0.0
      rz_deg: 0.0
  - parent: cam
    child: conn
    pose: 
      x_mm: 10.0
      y_mm: 11.0
      z_mm: 12.0
      rx_deg: 0.0
      ry_deg: 0.0
      rz_deg: 0.0
```

- Pose transformations must contain clear references to parent and child frames and optionally if the transformation involves an active joint (set via the joint axis-type object).
<!-- - The library is capable of computing inverse kinematics based on an updated end effector pose (using the robot's jacobian) **Still needs tests** -->

To use the library just import it and load the config file into the object

```python
from ktree import KinematicsConfig, KinematicsTree, Pose

kc = KinematicsConfig.parse(Path("./test/config.yaml"))
kt = KinematicsTree(config=kc)
end_eff_in_base = kt.get_transformation(parent=kc.base, child=kc.end_effector)
print(end_eff_in_base)
kt.update_transformation(parent="cam", child="conn", pose=Pose)
print(end_eff_in_base)
```
