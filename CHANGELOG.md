# Change Log

## 0.4.0
### Feature
- Change to uv as dependency manager.

## 0.3.4
### Fix
- Fix bug accepting `rot_vec` argument in `Rotation` as list.

## 0.3.3
### Add
- Update Rotation type to include support for definition using either quaternions, angle axis or matrix.

## 0.3.2
### Fix
- Warning on multiple paths now shows the paths found in the tree.

### Add
- Unit testing on multiple path warning.
- Instructions on calling dependency directly from github.

## 0.3.1
### Add
- Test for `KinematicTree` model dump.
- Add check for transformation for its inverse as well for equality.

### Fix
- numpy diving with slicing bug (might return zeros divinding if the type is not explicitly set as float)
- Model dump typo (was exporting the bidirectional graph)

## 0.3.0
### Change
- Rename `_add_edge` -> `_add_transformation` [BREAKING]

### Add
- Move model dump to `KinematicTree` model
- Add warning on closed loops

## 0.2.0
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