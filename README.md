# CIE 1931 Color Science Toolkit

A Python/NumPy implementation of fundamental CIE colorimetry for display calibration and color space conversion. Built for learning and practical use — every function maps to a real operation in the display calibration workflow.


## Why I Built This

I started building this while exploring how display calibration systems handle color 
accuracy across different devices. The core problem — converting 
between device-dependent RGB and device-independent XYZ — turns out 
to be pure linear algebra, and implementing it from scratch made the 
math click in a way that using libraries never did. Every function 
in this toolkit maps directly to a real operation in a display 
calibration workflow: measuring a display's primaries, building the 
conversion matrix, computing color error, and deriving a correction 
matrix to compensate for hardware drift. The signal processing 
patterns here — transform, calibrate, correct — translate directly 
to other domains where you're reading raw sensor data and need to 
get it into a standardized, meaningful space.


## What it does

```python
from cie_toolkit import *

# Build a conversion matrix from display primaries
M = build_rgb_to_xyz_matrix(REC709_PRIMARIES)

# Convert RGB to XYZ (matrix multiply)
xyz = rgb_to_xyz([0.8, 0.6, 0.2], M)

# Compute chromaticity (strip brightness)
xy = xyz_to_chromaticity(xyz)

# Check if a color is displayable
in_gamut = is_in_gamut(xyz, M)

# Compute color difference
error = delta_E_76(lab_target, lab_measured)

# Derive a correction matrix for a miscalibrated display
C = correction_matrix_from_matrices(M_actual, M_target)
corrected_rgb = C @ original_rgb
```

## Features

| Operation | Function | Math |
|-----------|----------|------|
| SPD → XYZ | `spd_to_xyz()` | Σ SPD(λ) × CMF(λ) × Δλ |
| XYZ → chromaticity | `xyz_to_chromaticity()` | x = X/(X+Y+Z) |
| Build M from primaries | `build_rgb_to_xyz_matrix()` | Column stacking + white point scaling |
| RGB → XYZ | `rgb_to_xyz()` | M @ rgb |
| XYZ → RGB | `xyz_to_rgb()` | np.linalg.solve(M, xyz) |
| Space-to-space | `convert_rgb()` | M_target⁻¹ @ M_source @ rgb |
| Correction matrix | `correction_matrix_from_matrices()` | M_actual⁻¹ @ M_target |
| Correction (least squares) | `correction_matrix_from_measurements()` | np.linalg.lstsq(actual, target) |
| Gamut check | `is_in_gamut()` | All RGB in [0, 1]? |
| Color difference | `delta_E_76()` | √(ΔL² + Δa² + Δb²) |
| Gamma decode/encode | `gamma_decode()` / `gamma_encode()` | rgb^γ / rgb^(1/γ) |
| Linearity check | `check_linearity()` | R² and residual analysis |

## Installation

```bash
git clone https://github.com/ViaSr/CIE-Toolkit.git
cd CIE-Toolkit
pip install -e ".[dev]"
```

## Run tests

```bash
pytest tests/ -v
```

## Project structure

```
CIE-Toolkit/
├── cie_toolkit/
│   └── __init__.py          # All functions in one module
├── tests/
│   └── test_toolkit.py      # 40+ tests verifying the math
├── examples/
│   └── calibration_demo.py  # Full calibration workflow
├── pyproject.toml
└── README.md
```

## Key concepts

**CIE XYZ** is the device-independent color space. Every color has a unique (X, Y, Z) coordinate, where Y equals luminance (perceived brightness).

**The M matrix** converts between a display's RGB and XYZ. Each column represents one primary's XYZ contribution at full power. Different displays have different M matrices — calibration means measuring M for a specific unit.

**Correction matrix** C pre-transforms RGB so an imperfect display produces correct colors: `M_actual @ (C @ rgb) = M_target @ rgb`. Derived either from known matrices or from least-squares fitting of measurement data.

**Grassmann's laws** (additivity and proportionality) make all of this work — they guarantee that color mixing is linear, so matrix multiplication is the right tool.

## Built with

- **NumPy** — all linear algebra operations
- **pytest** — test suite
- Optional: matplotlib/plotly for visualization, colour-science for validation

## License

MIT
