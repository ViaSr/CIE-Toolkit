"""
Tests for cie_toolkit — verifies math against known values.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from cie_toolkit import (
    cmf_x, cmf_y, cmf_z, cmf_matrix,
    spd_to_xyz, monochromatic_xyz,
    xyz_to_chromaticity, chromaticity_to_xyz,
    build_rgb_to_xyz_matrix, rgb_to_xyz, xyz_to_rgb,
    conversion_matrix, convert_rgb,
    correction_matrix_from_matrices, correction_matrix_from_measurements,
    is_in_gamut, gamut_clip,
    delta_xy, delta_E_76, xyz_to_lab,
    gamma_decode, gamma_encode,
    check_linearity, spectral_locus,
    REC709_PRIMARIES, DCI_P3_PRIMARIES, D65_xy,
)


# ── Color Matching Functions ──────────────────────────────

class TestCMF:
    def test_x_peaks_near_600nm(self):
        wl = np.arange(380, 781, 1)
        values = cmf_x(wl)
        peak_wl = wl[np.argmax(values)]
        assert 590 <= peak_wl <= 610

    def test_y_peaks_near_555nm(self):
        wl = np.arange(380, 781, 1)
        values = cmf_y(wl)
        peak_wl = wl[np.argmax(values)]
        assert 550 <= peak_wl <= 575

    def test_z_peaks_near_445nm(self):
        wl = np.arange(380, 781, 1)
        values = cmf_z(wl)
        peak_wl = wl[np.argmax(values)]
        assert 435 <= peak_wl <= 455

    def test_cmf_matrix_shape(self):
        M = cmf_matrix()
        assert M.shape == (401, 3)

    def test_cmf_matrix_custom_wavelengths(self):
        wl = np.arange(400, 701, 5)
        M = cmf_matrix(wl)
        assert M.shape == (len(wl), 3)

    def test_cmf_non_negative_y(self):
        """ȳ(λ) should be non-negative everywhere."""
        values = cmf_y(np.arange(380, 781, 1))
        assert np.all(values >= -0.001)


# ── SPD → XYZ ────────────────────────────────────────────

class TestSPDtoXYZ:
    def test_monochromatic(self):
        """Monochromatic source: XYZ = power × CMF(λ)."""
        xyz = monochromatic_xyz(550, power=100)
        expected = 100 * np.array([cmf_x(550), cmf_y(550), cmf_z(550)]).flatten()
        np.testing.assert_allclose(xyz, expected, atol=1e-10)

    def test_zero_spd(self):
        """No light → XYZ = [0, 0, 0]."""
        spd = np.zeros(401)
        xyz = spd_to_xyz(spd)
        np.testing.assert_allclose(xyz, [0, 0, 0], atol=1e-10)

    def test_additive(self):
        """SPD_a + SPD_b should give XYZ_a + XYZ_b (Grassmann's law)."""
        wl = np.arange(380, 781, 1)
        spd_a = np.zeros(401)
        spd_a[wl == 500] = 50
        spd_b = np.zeros(401)
        spd_b[wl == 600] = 30

        xyz_a = spd_to_xyz(spd_a)
        xyz_b = spd_to_xyz(spd_b)
        xyz_combined = spd_to_xyz(spd_a + spd_b)

        np.testing.assert_allclose(xyz_combined, xyz_a + xyz_b, atol=1e-10)

    def test_proportionality(self):
        """Doubling SPD should double XYZ (Grassmann's law)."""
        spd = np.zeros(401)
        spd[100] = 60  # arbitrary wavelength

        xyz_1x = spd_to_xyz(spd)
        xyz_2x = spd_to_xyz(2 * spd)

        np.testing.assert_allclose(xyz_2x, 2 * xyz_1x, atol=1e-10)


# ── Chromaticity ──────────────────────────────────────────

class TestChromaticity:
    def test_sum_to_one(self):
        """x + y + z must always equal 1."""
        xyz = np.array([0.58, 0.61, 0.28])
        total = np.sum(xyz)
        x, y = xyz_to_chromaticity(xyz)
        z = xyz[2] / total
        assert abs(x + y + z - 1.0) < 1e-10

    def test_brightness_invariance(self):
        """Bright and dim versions of the same color → same chromaticity."""
        xyz_bright = np.array([68, 73, 0.06])
        xyz_dim = np.array([34, 36.5, 0.03])  # half brightness
        xy_bright = xyz_to_chromaticity(xyz_bright)
        xy_dim = xyz_to_chromaticity(xyz_dim)
        np.testing.assert_allclose(xy_bright, xy_dim, atol=1e-10)

    def test_round_trip(self):
        """XYZ → chromaticity + Y → XYZ should recover original."""
        xyz = np.array([0.58, 0.61, 0.28])
        x, y = xyz_to_chromaticity(xyz)
        recovered = chromaticity_to_xyz(x, y, Y=xyz[1])
        np.testing.assert_allclose(recovered, xyz, atol=1e-10)

    def test_batch(self):
        """Batch chromaticity conversion."""
        xyz_batch = np.array([[0.5, 0.5, 0.5],
                              [0.3, 0.6, 0.1]])
        xy = xyz_to_chromaticity(xyz_batch)
        assert xy.shape == (2, 2)
        np.testing.assert_allclose(xy[0], [1/3, 1/3], atol=1e-10)


# ── RGB ↔ XYZ Matrix ─────────────────────────────────────

class TestMatrixConversion:
    @pytest.fixture
    def M_709(self):
        return build_rgb_to_xyz_matrix(REC709_PRIMARIES)

    def test_white_point(self, M_709):
        """RGB [1,1,1] should map to D65 white chromaticity."""
        xyz_white = M_709 @ np.array([1.0, 1.0, 1.0])
        xy = xyz_to_chromaticity(xyz_white)
        np.testing.assert_allclose(xy, D65_xy, atol=1e-3)

    def test_red_primary(self, M_709):
        """RGB [1,0,0] should map to the red primary chromaticity."""
        xyz_red = M_709 @ np.array([1.0, 0.0, 0.0])
        xy = xyz_to_chromaticity(xyz_red)
        np.testing.assert_allclose(xy, [0.640, 0.330], atol=1e-3)

    def test_green_primary(self, M_709):
        """RGB [0,1,0] should map to the green primary chromaticity."""
        xyz_green = M_709 @ np.array([0.0, 1.0, 0.0])
        xy = xyz_to_chromaticity(xyz_green)
        np.testing.assert_allclose(xy, [0.300, 0.600], atol=1e-3)

    def test_blue_primary(self, M_709):
        """RGB [0,0,1] should map to the blue primary chromaticity."""
        xyz_blue = M_709 @ np.array([0.0, 0.0, 1.0])
        xy = xyz_to_chromaticity(xyz_blue)
        np.testing.assert_allclose(xy, [0.150, 0.060], atol=1e-3)

    def test_round_trip(self, M_709):
        """RGB → XYZ → RGB should recover the original."""
        rgb = np.array([0.8, 0.6, 0.2])
        xyz = rgb_to_xyz(rgb, M_709)
        recovered = xyz_to_rgb(xyz, M_709)
        np.testing.assert_allclose(recovered, rgb, atol=1e-10)

    def test_luminance_coefficients(self, M_709):
        """Y row should approximately equal standard luminance weights."""
        Y_row = M_709[1]
        np.testing.assert_allclose(Y_row[0], 0.2126, atol=0.001)
        np.testing.assert_allclose(Y_row[1], 0.7152, atol=0.001)
        np.testing.assert_allclose(Y_row[2], 0.0722, atol=0.001)

    def test_batch_conversion(self, M_709):
        """Batch RGB → XYZ for multiple colors."""
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        xyz_batch = rgb_to_xyz(colors, M_709)
        assert xyz_batch.shape == (3, 3)


# ── Color Space Conversion ───────────────────────────────

class TestColorSpaceConversion:
    def test_709_to_p3_white(self):
        """White should stay [1,1,1] between spaces sharing D65."""
        M_709 = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
        M_p3 = build_rgb_to_xyz_matrix(DCI_P3_PRIMARIES)
        white_p3 = convert_rgb(np.array([1.0, 1.0, 1.0]), M_709, M_p3)
        np.testing.assert_allclose(white_p3, [1.0, 1.0, 1.0], atol=1e-10)

    def test_709_inside_p3(self):
        """All Rec. 709 primaries should be inside the P3 gamut."""
        M_709 = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
        M_p3 = build_rgb_to_xyz_matrix(DCI_P3_PRIMARIES)
        for primary in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            p3_rgb = convert_rgb(np.array(primary, dtype=float), M_709, M_p3)
            assert np.all(p3_rgb >= -1e-6), f"709 primary {primary} outside P3"


# ── Correction Matrix ────────────────────────────────────

class TestCorrectionMatrix:
    def test_correction_exact(self):
        """Correction matrix should perfectly map actual → target."""
        M_target = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
        shifted = dict(REC709_PRIMARIES)
        shifted["R"] = (0.635, 0.335)
        M_actual = build_rgb_to_xyz_matrix(shifted)

        C = correction_matrix_from_matrices(M_actual, M_target)
        rgb = np.array([0.8, 0.6, 0.2])
        corrected = C @ rgb
        result = M_actual @ corrected
        expected = M_target @ rgb
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_correction_lstsq(self):
        """Least-squares correction from multiple test patches."""
        M_target = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
        shifted = dict(REC709_PRIMARIES)
        shifted["G"] = (0.295, 0.605)
        M_actual = build_rgb_to_xyz_matrix(shifted)

        patches = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [1, 1, 1], [0.5, 0.5, 0.5]], dtype=float)
        actual = rgb_to_xyz(patches, M_actual)
        target = rgb_to_xyz(patches, M_target)

        C = correction_matrix_from_measurements(actual, target)
        corrected = actual @ C
        np.testing.assert_allclose(corrected, target, atol=1e-8)


# ── Gamut ─────────────────────────────────────────────────

class TestGamut:
    def test_primary_in_gamut(self):
        M = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
        xyz_red = M @ np.array([1.0, 0.0, 0.0])
        assert is_in_gamut(xyz_red, M)

    def test_saturated_cyan_out_of_gamut(self):
        M = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
        xyz_cyan = chromaticity_to_xyz(0.1, 0.5, Y=0.5)
        assert not is_in_gamut(xyz_cyan, M)

    def test_gamut_clip(self):
        rgb = np.array([1.2, 0.5, -0.1])
        clipped = gamut_clip(rgb)
        assert np.all(clipped >= 0)
        assert np.all(clipped <= 1)
        np.testing.assert_allclose(clipped, [1.0, 0.5, 0.0])


# ── Color Difference ──────────────────────────────────────

class TestColorDifference:
    def test_delta_xy_identical(self):
        xy = np.array([0.31, 0.33])
        assert delta_xy(xy, xy) == pytest.approx(0.0)

    def test_delta_xy_known(self):
        d = delta_xy(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        assert d == pytest.approx(5.0)

    def test_delta_E_identical(self):
        lab = np.array([50, 0, 0])
        assert delta_E_76(lab, lab) == pytest.approx(0.0)

    def test_delta_E_known(self):
        lab1 = np.array([50, 0, 0])
        lab2 = np.array([50, 3, 4])
        assert delta_E_76(lab1, lab2) == pytest.approx(5.0)

    def test_xyz_to_lab_white(self):
        """D65 white in Lab should be approximately [100, 0, 0]."""
        white_xyz = chromaticity_to_xyz(*D65_xy, Y=1.0)
        lab = xyz_to_lab(white_xyz)
        assert lab[0] == pytest.approx(100.0, abs=0.1)
        assert abs(lab[1]) < 1.0
        assert abs(lab[2]) < 1.0


# ── Gamma ─────────────────────────────────────────────────

class TestGamma:
    def test_round_trip(self):
        rgb = np.array([0.2, 0.5, 0.8])
        encoded = gamma_encode(rgb)
        decoded = gamma_decode(encoded)
        np.testing.assert_allclose(decoded, rgb, atol=1e-10)

    def test_midpoint_not_half(self):
        """50% encoded should NOT produce 50% linear output."""
        linear = gamma_decode(np.array([0.5]), gamma=2.2)
        assert linear[0] < 0.25  # 0.5^2.2 ≈ 0.217

    def test_identity_at_gamma_1(self):
        rgb = np.array([0.3, 0.6, 0.9])
        np.testing.assert_allclose(gamma_decode(rgb, gamma=1.0), rgb)


# ── Linearity ─────────────────────────────────────────────

class TestLinearity:
    def test_perfect_line(self):
        x = np.arange(0, 101, 5, dtype=float)
        y = 2 * x + 3
        result = check_linearity(x, y)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-10)
        assert result["is_linear"]

    def test_curved_data(self):
        x = np.arange(0, 101, 5, dtype=float)
        y = x ** 2.2
        result = check_linearity(x, y)
        assert result["r_squared"] < 0.999
        assert not result["is_linear"]


# ── Spectral Locus ────────────────────────────────────────

class TestSpectralLocus:
    def test_shape(self):
        locus = spectral_locus()
        assert locus.ndim == 2
        assert locus.shape[1] == 3  # wavelength, x, y

    def test_chromaticity_in_range(self):
        locus = spectral_locus()
        assert np.all(locus[:, 1] >= 0)  # x >= 0
        assert np.all(locus[:, 2] >= 0)  # y >= 0
        assert np.all(locus[:, 1] <= 1)  # x <= 1
        assert np.all(locus[:, 2] <= 1)  # y <= 1
