# Changelog

## [1.1.1] - (2026-07-17)

### Added
- `CITATION.cff` with software citation metadata

## [1.1.0] - (2026-07-15)

### Added
- `--threshold` option for `create-t2pmhc-graphs`: configurable CA-CA distance threshold in Angstroms for contact-map construction (default: 10.0)

### Fixed
- GPU inference device mismatch in `t2pmhc-predict-binding`: the model is now moved to the selected device, fixing `Expected all tensors to be on the same device` on CUDA systems (fixes #13)

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