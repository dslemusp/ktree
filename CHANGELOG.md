# Change Log

## 0.1.7
### Add
- Custom serialization of KinematicConfig object.

### Fix
- Lint in test file.

## 0.1.6
### Add
- Enable remove_transformations from kinematic chain

## 0.1.5
### Add
- Add support for mm_deg in Pose object. You can now create/export poses in mm_deg
- Add unit testing for homogeneous transformations.

## 0.1.4
### Fix
- Type checking errors
### Add
- Validation of parent,child frames when passed in update_transformations.

## 0.1.3

### Fix
- Type checking errors

### Add
- Support for `__eq__` in types. 

## 0.1.2

### Add
- Implement `__str__` function in type Pose for printing.

## 0.1.1

### Fix
- Import issue in testing.

## 0.1.0

### Added
- Migrated kinematic tree implementation from [roc@home](https://github.com/rocsys/research-roc_at_home)
- Updated types to allow for easy vector cross product and matrix-matrix matrix-vector multiplication.
- Refactor to allow stand alone usage.
- Added unit tests for types, configuration model and basic functionality of the kinematic tree.
- Added CI.