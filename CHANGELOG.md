# Changelog

## [1.0.2] - (2026-03-20)

 ### Added
 - `--training-mode` / `--prediction-mode` flag to `create-t2pmhc-graphs` command
   - `--training-mode`: extracts labels from PDB filename suffixes (`_0`/`_1`) for training graphs
   - `--prediction-mode`: assigns dummy labels for prediction graphs

 ### Fixed
 - Improved README

## [1.0.1] - Ionic Hleb (2025-21-01)

### Added

### Fixed
- git support in Dockerfile


## [1.0.0] - Ionic Hleb (2025-21-01)

### Added

- Initial release of t2pmhc
- Creation of t2pmhc graphs
- Training of t2pmhc-GCN and t2pmhc-GAT
- Prediction of samples using both variants
- Dockerfile for v1.0.0
- Documentation

### Fixed