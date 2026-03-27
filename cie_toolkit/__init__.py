"""
CIE 1931 Color Science Toolkit
==============================
A Python/NumPy implementation of fundamental CIE colorimetry operations
for display calibration and color space conversion.

Implements:
    - CIE 1931 color matching functions (Gaussian approximation)
    - SPD → XYZ tristimulus integration
    - Chromaticity computation and round-trip conversion
    - RGB ↔ XYZ matrix construction from display primaries
    - Color space conversion between arbitrary RGB spaces
    - Correction matrix derivation (exact and least-squares)
    - Gamut boundary checking
    - Color difference metrics (ΔE, Δxy, Δu'v')
    - Gamma / transfer function encoding and decoding
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════

#: Standard D65 illuminant chromaticity (daylight white point)
D65_xy = np.array([0.3127, 0.3290])

#: Rec. 709 / sRGB primary chromaticities
REC709_PRIMARIES = {
    "R": (0.640, 0.330),
    "G": (0.300, 0.600),
    "B": (0.150, 0.060),
}

#: DCI-P3 primary chromaticities
DCI_P3_PRIMARIES = {
    "R": (0.680, 0.320),
    "G": (0.265, 0.690),
    "B": (0.150, 0.060),
}

#: Rec. 2020 primary chromaticities
REC2020_PRIMARIES = {
    "R": (0.708, 0.292),
    "G": (0.170, 0.797),
    "B": (0.131, 0.046),
}

#: Default wavelength range for CIE 1931 (380–780 nm, 1 nm steps)
DEFAULT_WAVELENGTHS = np.arange(380, 781, 1, dtype=float)


# ════════════════════════════════════════════════════════════
# Color Matching Functions
# ════════════════════════════════════════════════════════════

def _gaussian(x: NDArray, mu: float, sigma1: float, sigma2: float) -> NDArray:
    """Piecewise Gaussian with different spread left/right of peak."""
    sigma = np.where(x < mu, sigma1, sigma2)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cmf_x(wavelength: NDArray | float) -> NDArray:
    """CIE 1931 x̄(λ) color matching function (Wyman et al. 2013 approximation).

    Parameters
    ----------
    wavelength : array-like
        Wavelength(s) in nanometers.

    Returns
    -------
    ndarray
        x̄ value(s) at the given wavelength(s).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    return (1.056 * _gaussian(wavelength, 599.8, 37.9, 31.0)
            + 0.362 * _gaussian(wavelength, 442.0, 16.0, 26.7)
            - 0.065 * _gaussian(wavelength, 501.1, 20.4, 26.2))


def cmf_y(wavelength: NDArray | float) -> NDArray:
    """CIE 1931 ȳ(λ) color matching function — equals the luminance function.

    Parameters
    ----------
    wavelength : array-like
        Wavelength(s) in nanometers.

    Returns
    -------
    ndarray
        ȳ value(s) at the given wavelength(s).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    return (0.821 * _gaussian(wavelength, 568.8, 46.9, 40.5)
            + 0.286 * _gaussian(wavelength, 530.9, 16.3, 31.1))


def cmf_z(wavelength: NDArray | float) -> NDArray:
    """CIE 1931 z̄(λ) color matching function — peaks in the blue/violet region.

    Parameters
    ----------
    wavelength : array-like
        Wavelength(s) in nanometers.

    Returns
    -------
    ndarray
        z̄ value(s) at the given wavelength(s).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    return (1.217 * _gaussian(wavelength, 437.0, 11.8, 36.0)
            + 0.681 * _gaussian(wavelength, 459.0, 26.0, 13.8))


def cmf_matrix(wavelengths: NDArray | None = None) -> NDArray:
    """Build the (N, 3) color matching function matrix.

    Each row is [x̄(λ), ȳ(λ), z̄(λ)] at one wavelength.

    Parameters
    ----------
    wavelengths : ndarray, optional
        Wavelengths in nm. Defaults to 380–780 nm in 1 nm steps.

    Returns
    -------
    ndarray of shape (N, 3)
        The standard observer matrix.
    """
    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS
    return np.column_stack([cmf_x(wavelengths),
                            cmf_y(wavelengths),
                            cmf_z(wavelengths)])


# ════════════════════════════════════════════════════════════
# SPD → XYZ Integration
# ════════════════════════════════════════════════════════════

def spd_to_xyz(
    spd: NDArray,
    wavelengths: NDArray | None = None,
    cmfs: NDArray | None = None,
    delta_lambda: float = 1.0,
) -> NDArray:
    """Convert a spectral power distribution to CIE XYZ tristimulus values.

    This is the core integration operation:
        X = Σ SPD(λ) × x̄(λ) × Δλ
        Y = Σ SPD(λ) × ȳ(λ) × Δλ
        Z = Σ SPD(λ) × z̄(λ) × Δλ

    Parameters
    ----------
    spd : ndarray of shape (N,)
        Spectral power at each wavelength.
    wavelengths : ndarray, optional
        Wavelengths corresponding to spd values.
    cmfs : ndarray of shape (N, 3), optional
        Color matching function matrix. Built from wavelengths if not given.
    delta_lambda : float
        Wavelength step size in nm (default 1.0).

    Returns
    -------
    ndarray of shape (3,)
        Tristimulus values [X, Y, Z].
    """
    if cmfs is None:
        cmfs = cmf_matrix(wavelengths)
    return (spd @ cmfs) * delta_lambda


def monochromatic_xyz(wavelength: float, power: float = 1.0) -> NDArray:
    """Compute XYZ for a monochromatic light source at a single wavelength.

    Parameters
    ----------
    wavelength : float
        Wavelength in nm.
    power : float
        Power of the source.

    Returns
    -------
    ndarray of shape (3,)
        Tristimulus values [X, Y, Z].
    """
    return power * np.array([cmf_x(wavelength),
                             cmf_y(wavelength),
                             cmf_z(wavelength)]).flatten()


# ════════════════════════════════════════════════════════════
# Chromaticity
# ════════════════════════════════════════════════════════════

def xyz_to_chromaticity(xyz: NDArray) -> NDArray:
    """Convert XYZ to (x, y) chromaticity coordinates.

    Strips brightness, keeping only color quality:
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

    Parameters
    ----------
    xyz : ndarray of shape (3,) or (N, 3)
        Tristimulus value(s).

    Returns
    -------
    ndarray of shape (2,) or (N, 2)
        Chromaticity coordinates [x, y].
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        total = np.sum(xyz)
        return np.array([xyz[0] / total, xyz[1] / total])
    else:
        total = np.sum(xyz, axis=1, keepdims=True)
        return xyz[:, :2] / total


def chromaticity_to_xyz(x: float, y: float, Y: float = 1.0) -> NDArray:
    """Recover full XYZ from chromaticity (x, y) and luminance Y.

    Parameters
    ----------
    x, y : float
        Chromaticity coordinates.
    Y : float
        Luminance (default 1.0).

    Returns
    -------
    ndarray of shape (3,)
        Tristimulus values [X, Y, Z].
    """
    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    return np.array([X, Y, Z])


# ════════════════════════════════════════════════════════════
# RGB ↔ XYZ Matrix Construction
# ════════════════════════════════════════════════════════════

def build_rgb_to_xyz_matrix(
    primaries: dict[str, tuple[float, float]],
    white_point: NDArray | tuple[float, float] = D65_xy,
    Y_white: float = 1.0,
) -> NDArray:
    """Build the 3×3 RGB→XYZ conversion matrix from display primaries.

    This is the core calibration operation. Each column of the resulting
    matrix represents one primary's XYZ contribution at full power.

    Parameters
    ----------
    primaries : dict
        Keys 'R', 'G', 'B' mapping to (x, y) chromaticity tuples.
    white_point : array-like
        (x, y) chromaticity of the white point (default D65).
    Y_white : float
        Luminance of white (default 1.0).

    Returns
    -------
    ndarray of shape (3, 3)
        Matrix M where ``M @ rgb = xyz``.
    """
    white_point = np.asarray(white_point, dtype=float)

    # Convert each primary chromaticity to XYZ with Y=1
    XYZ_r = chromaticity_to_xyz(*primaries["R"])
    XYZ_g = chromaticity_to_xyz(*primaries["G"])
    XYZ_b = chromaticity_to_xyz(*primaries["B"])

    # Stack as columns
    M_unscaled = np.column_stack([XYZ_r, XYZ_g, XYZ_b])

    # Solve for scaling factors: M_unscaled @ S = XYZ_white
    XYZ_white = chromaticity_to_xyz(*white_point, Y=Y_white)
    S = np.linalg.solve(M_unscaled, XYZ_white)

    # Scale each column by its factor (broadcasting)
    return M_unscaled * S


def rgb_to_xyz(rgb: NDArray, M: NDArray) -> NDArray:
    """Convert RGB to XYZ using a conversion matrix.

    Parameters
    ----------
    rgb : ndarray of shape (3,) or (N, 3)
        Linear RGB value(s).
    M : ndarray of shape (3, 3)
        RGB→XYZ conversion matrix.

    Returns
    -------
    ndarray
        XYZ value(s), same shape as input.
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim == 1:
        return M @ rgb
    return (M @ rgb.T).T


def xyz_to_rgb(xyz: NDArray, M: NDArray) -> NDArray:
    """Convert XYZ to RGB using a conversion matrix.

    Uses np.linalg.solve for numerical stability (avoids explicit inverse).

    Parameters
    ----------
    xyz : ndarray of shape (3,) or (N, 3)
        Tristimulus value(s).
    M : ndarray of shape (3, 3)
        RGB→XYZ conversion matrix (forward direction).

    Returns
    -------
    ndarray
        Linear RGB value(s), same shape as input.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        return np.linalg.solve(M, xyz)
    return np.linalg.solve(M, xyz.T).T


# ════════════════════════════════════════════════════════════
# Color Space Conversion
# ════════════════════════════════════════════════════════════

def conversion_matrix(M_source: NDArray, M_target: NDArray) -> NDArray:
    """Compute the 3×3 matrix that converts from one RGB space to another.

    The resulting matrix C satisfies: rgb_target = C @ rgb_source.

    Parameters
    ----------
    M_source : ndarray (3, 3)
        RGB→XYZ matrix of the source color space.
    M_target : ndarray (3, 3)
        RGB→XYZ matrix of the target color space.

    Returns
    -------
    ndarray (3, 3)
        Conversion matrix from source RGB to target RGB.
    """
    return np.linalg.inv(M_target) @ M_source


def convert_rgb(
    rgb: NDArray,
    M_source: NDArray,
    M_target: NDArray,
) -> NDArray:
    """Convert RGB values from one color space to another.

    Parameters
    ----------
    rgb : ndarray of shape (3,) or (N, 3)
        RGB value(s) in the source space.
    M_source, M_target : ndarray (3, 3)
        RGB→XYZ matrices for source and target spaces.

    Returns
    -------
    ndarray
        RGB value(s) in the target space.
    """
    C = conversion_matrix(M_source, M_target)
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim == 1:
        return C @ rgb
    return (C @ rgb.T).T


# ════════════════════════════════════════════════════════════
# Correction Matrix
# ════════════════════════════════════════════════════════════

def correction_matrix_from_matrices(
    M_actual: NDArray,
    M_target: NDArray,
) -> NDArray:
    """Compute a correction matrix from known actual and target matrices.

    The correction C pre-transforms RGB so the imperfect display
    produces the correct output: M_actual @ (C @ rgb) = M_target @ rgb.

    Parameters
    ----------
    M_actual : ndarray (3, 3)
        The display's actual (measured) RGB→XYZ matrix.
    M_target : ndarray (3, 3)
        The desired (ideal) RGB→XYZ matrix.

    Returns
    -------
    ndarray (3, 3)
        Correction matrix C.
    """
    return np.linalg.solve(M_actual, M_target)


def correction_matrix_from_measurements(
    actual_xyz: NDArray,
    target_xyz: NDArray,
) -> NDArray:
    """Compute a correction matrix from measured test patches (least squares).

    Finds C that minimizes ||actual @ C - target||² across all patches.

    Parameters
    ----------
    actual_xyz : ndarray of shape (N, 3)
        Measured XYZ values for N test patches.
    target_xyz : ndarray of shape (N, 3)
        Expected XYZ values for N test patches.

    Returns
    -------
    ndarray (3, 3)
        Correction matrix C such that corrected = actual @ C ≈ target.
    """
    C, _, _, _ = np.linalg.lstsq(actual_xyz, target_xyz, rcond=None)
    return C


# ════════════════════════════════════════════════════════════
# Gamut Checking
# ════════════════════════════════════════════════════════════

def is_in_gamut(
    xyz: NDArray,
    M: NDArray,
    tolerance: float = 1e-6,
) -> bool | NDArray:
    """Check if a color (in XYZ) is within a display's gamut.

    A color is in gamut if the required RGB values are all in [0, 1].

    Parameters
    ----------
    xyz : ndarray of shape (3,) or (N, 3)
        Tristimulus value(s) to check.
    M : ndarray (3, 3)
        The display's RGB→XYZ matrix.
    tolerance : float
        Numerical tolerance for boundary cases.

    Returns
    -------
    bool or ndarray of bool
        True if the color is displayable.
    """
    rgb = xyz_to_rgb(xyz, M)
    if rgb.ndim == 1:
        return bool(np.all(rgb >= -tolerance) and np.all(rgb <= 1.0 + tolerance))
    return np.all(rgb >= -tolerance, axis=1) & np.all(rgb <= 1.0 + tolerance, axis=1)


def gamut_clip(rgb: NDArray) -> NDArray:
    """Clip RGB values to the displayable [0, 1] range.

    This is the simplest gamut mapping — it distorts hue and saturation
    but guarantees displayable values.

    Parameters
    ----------
    rgb : ndarray
        RGB value(s), possibly out of range.

    Returns
    -------
    ndarray
        Clipped RGB value(s).
    """
    return np.clip(rgb, 0.0, 1.0)


# ════════════════════════════════════════════════════════════
# Color Difference Metrics
# ════════════════════════════════════════════════════════════

def delta_xy(
    xy1: NDArray,
    xy2: NDArray,
) -> float | NDArray:
    """Chromaticity distance in CIE xy space.

    Parameters
    ----------
    xy1, xy2 : ndarray of shape (2,) or (N, 2)
        Chromaticity coordinate(s).

    Returns
    -------
    float or ndarray
        Euclidean distance(s) in xy space.
    """
    xy1, xy2 = np.asarray(xy1), np.asarray(xy2)
    diff = xy1 - xy2
    if diff.ndim == 1:
        return float(np.sqrt(np.sum(diff ** 2)))
    return np.sqrt(np.sum(diff ** 2, axis=1))


def delta_E_76(
    Lab1: NDArray,
    Lab2: NDArray,
) -> float | NDArray:
    """CIE 1976 color difference (Euclidean distance in Lab space).

    Parameters
    ----------
    Lab1, Lab2 : ndarray of shape (3,) or (N, 3)
        CIELAB values [L, a, b].

    Returns
    -------
    float or ndarray
        ΔE value(s).

    Notes
    -----
    ΔE < 1:   imperceptible
    ΔE 1-2:   barely noticeable
    ΔE 2-10:  visible at a glance
    ΔE > 10:  completely different
    """
    Lab1, Lab2 = np.asarray(Lab1), np.asarray(Lab2)
    diff = Lab1 - Lab2
    if diff.ndim == 1:
        return float(np.sqrt(np.sum(diff ** 2)))
    return np.sqrt(np.sum(diff ** 2, axis=1))


def xyz_to_lab(
    xyz: NDArray,
    white_xyz: NDArray | None = None,
) -> NDArray:
    """Convert XYZ to CIELAB.

    Parameters
    ----------
    xyz : ndarray of shape (3,) or (N, 3)
        Tristimulus values.
    white_xyz : ndarray of shape (3,), optional
        Reference white XYZ. Defaults to D65.

    Returns
    -------
    ndarray
        Lab values [L, a, b].
    """
    if white_xyz is None:
        white_xyz = chromaticity_to_xyz(*D65_xy, Y=1.0)

    xyz = np.asarray(xyz, dtype=float)
    ratio = xyz / white_xyz

    # CIE nonlinear compression
    delta = 6.0 / 29.0
    mask = ratio > delta ** 3
    f = np.where(mask, np.cbrt(ratio), ratio / (3 * delta ** 2) + 4.0 / 29.0)

    if f.ndim == 1:
        L = 116.0 * f[1] - 16.0
        a = 500.0 * (f[0] - f[1])
        b = 200.0 * (f[1] - f[2])
        return np.array([L, a, b])
    else:
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return np.column_stack([L, a, b])


# ════════════════════════════════════════════════════════════
# Gamma / Transfer Function
# ════════════════════════════════════════════════════════════

def gamma_decode(rgb_encoded: NDArray, gamma: float = 2.2) -> NDArray:
    """Linearize gamma-encoded RGB values.

    Must be applied BEFORE matrix math.

    Parameters
    ----------
    rgb_encoded : ndarray
        Gamma-encoded (nonlinear) RGB values in [0, 1].
    gamma : float
        Gamma exponent (default 2.2 for sRGB approximate).

    Returns
    -------
    ndarray
        Linear RGB values.
    """
    return np.clip(rgb_encoded, 0.0, 1.0) ** gamma


def gamma_encode(rgb_linear: NDArray, gamma: float = 2.2) -> NDArray:
    """Apply gamma encoding to linear RGB values.

    Applied AFTER matrix math for display output.

    Parameters
    ----------
    rgb_linear : ndarray
        Linear RGB values in [0, 1].
    gamma : float
        Gamma exponent (default 2.2 for sRGB approximate).

    Returns
    -------
    ndarray
        Gamma-encoded RGB values.
    """
    return np.clip(rgb_linear, 0.0, 1.0) ** (1.0 / gamma)


# ════════════════════════════════════════════════════════════
# Linearity Verification
# ════════════════════════════════════════════════════════════

def check_linearity(
    inputs: NDArray,
    measured: NDArray,
) -> dict:
    """Verify linearity of a display response using R² and residual analysis.

    Parameters
    ----------
    inputs : ndarray of shape (N,)
        Input drive levels (e.g., 0 to 255).
    measured : ndarray of shape (N,)
        Measured output values (e.g., luminance in cd/m²).

    Returns
    -------
    dict with keys:
        'r_squared': float — coefficient of determination (1.0 = perfect)
        'slope': float — best-fit slope
        'intercept': float — best-fit intercept
        'residuals': ndarray — measured minus predicted
        'max_residual': float — worst-case deviation
        'is_linear': bool — True if R² > 0.999
    """
    coeffs = np.polyfit(inputs, measured, 1)
    predicted = np.polyval(coeffs, inputs)
    residuals = measured - predicted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "r_squared": float(r_squared),
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "residuals": residuals,
        "max_residual": float(np.max(np.abs(residuals))),
        "is_linear": r_squared > 0.999,
    }


# ════════════════════════════════════════════════════════════
# Spectral Locus (for plotting the chromaticity diagram)
# ════════════════════════════════════════════════════════════

def spectral_locus(
    wavelengths: NDArray | None = None,
) -> NDArray:
    """Compute the (x, y) chromaticity of each monochromatic wavelength.

    This traces the horseshoe boundary of the chromaticity diagram.

    Parameters
    ----------
    wavelengths : ndarray, optional
        Wavelengths to compute. Defaults to 380–780 nm in 2 nm steps.

    Returns
    -------
    ndarray of shape (N, 3)
        Columns are [wavelength, x, y].
    """
    if wavelengths is None:
        wavelengths = np.arange(380, 781, 2, dtype=float)

    result = []
    for wl in wavelengths:
        xyz = monochromatic_xyz(wl)
        total = np.sum(xyz)
        if total > 1e-6:
            result.append([wl, xyz[0] / total, xyz[1] / total])

    return np.array(result)
