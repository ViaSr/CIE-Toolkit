"""
Full Display Calibration Workflow Demo
======================================
Demonstrates the complete pipeline from measurement to correction.
This is what SmallHD-style calibration automation looks like.
"""

import numpy as np
from cie_toolkit import (
    build_rgb_to_xyz_matrix, rgb_to_xyz, xyz_to_rgb,
    xyz_to_chromaticity, chromaticity_to_xyz,
    correction_matrix_from_matrices, correction_matrix_from_measurements,
    is_in_gamut, delta_xy, delta_E_76, xyz_to_lab,
    check_linearity, gamma_decode,
    REC709_PRIMARIES, DCI_P3_PRIMARIES, D65_xy,
    spd_to_xyz, monochromatic_xyz,
)

np.set_printoptions(precision=4, suppress=True)


def main():
    print("=" * 60)
    print("  DISPLAY CALIBRATION WORKFLOW DEMO")
    print("=" * 60)

    # ── Step 1: Define the ideal (target) ──────────────────
    print("\n1. Build target matrix (Rec. 709 standard)")
    M_target = build_rgb_to_xyz_matrix(REC709_PRIMARIES)
    print(f"   M_target:\n{M_target}")

    # ── Step 2: Simulate a real (imperfect) display ────────
    print("\n2. Build actual matrix (measured from this specific monitor)")
    actual_primaries = {
        "R": (0.635, 0.335),   # red shifted slightly
        "G": (0.295, 0.605),   # green shifted slightly
        "B": (0.155, 0.062),   # blue shifted slightly
    }
    M_actual = build_rgb_to_xyz_matrix(actual_primaries)
    print(f"   M_actual:\n{M_actual}")

    # ── Step 3: Measure test patches ───────────────────────
    print("\n3. Send test patches and compare target vs actual")
    test_colors = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 1.0],  # white
        [0.8, 0.6, 0.2],  # warm orange
        [0.2, 0.4, 0.9],  # cool blue
    ])
    labels = ["Red", "Green", "Blue", "White", "Orange", "Cool blue"]

    target_xyz = rgb_to_xyz(test_colors, M_target)
    actual_xyz = rgb_to_xyz(test_colors, M_actual)

    target_lab = xyz_to_lab(target_xyz)
    actual_lab = xyz_to_lab(actual_xyz)
    errors = delta_E_76(target_lab, actual_lab)

    print(f"   {'Patch':<12} {'ΔE':>8}  {'Verdict'}")
    print(f"   {'-'*36}")
    for label, dE in zip(labels, errors):
        verdict = "OK" if dE < 1 else "noticeable" if dE < 3 else "NEEDS FIX"
        print(f"   {label:<12} {dE:>8.4f}  {verdict}")

    # ── Step 4: Compute correction ─────────────────────────
    print("\n4. Compute correction matrix")
    C = correction_matrix_from_matrices(M_actual, M_target)
    print(f"   C:\n{C}")
    print(f"   (Close to identity — small corrections needed)")

    # ── Step 5: Verify correction ──────────────────────────
    print("\n5. Verify: apply correction and re-measure")
    corrected_xyz = np.array([M_actual @ (C @ rgb) for rgb in test_colors])
    corrected_lab = xyz_to_lab(corrected_xyz)
    errors_after = delta_E_76(target_lab, corrected_lab)

    print(f"   {'Patch':<12} {'Before':>8} {'After':>8}")
    print(f"   {'-'*32}")
    for label, before, after in zip(labels, errors, errors_after):
        print(f"   {label:<12} {before:>8.4f} {after:>8.4f}")
    print(f"\n   Max ΔE before: {np.max(errors):.4f}")
    print(f"   Max ΔE after:  {np.max(errors_after):.4f}")

    # ── Step 6: Gamut comparison ───────────────────────────
    print("\n6. Gamut check — can this monitor display P3 colors?")
    M_p3 = build_rgb_to_xyz_matrix(DCI_P3_PRIMARIES)
    p3_test_colors = np.array([
        [1.0, 0.0, 0.0],  # P3 red (wider than 709)
        [0.0, 1.0, 0.0],  # P3 green (wider than 709)
        [0.5, 0.0, 0.0],  # P3 50% red
    ])
    for i, rgb in enumerate(p3_test_colors):
        xyz = rgb_to_xyz(rgb, M_p3)
        in_709 = is_in_gamut(xyz, M_target)
        rgb_709 = xyz_to_rgb(xyz, M_target)
        print(f"   P3 color {rgb} → in 709? {in_709}")
        if not in_709:
            print(f"     Would need RGB = {rgb_709} (negative = impossible)")

    # ── Step 7: Metamerism demo ────────────────────────────
    print("\n7. Metamerism — two different SPDs, same perceived color")
    xyz_mono = monochromatic_xyz(575, power=80)
    xy_mono = xyz_to_chromaticity(xyz_mono)
    print(f"   Sodium lamp (575nm, power=80): XYZ={xyz_mono}")
    print(f"   Chromaticity: ({xy_mono[0]:.4f}, {xy_mono[1]:.4f})")

    rgb_to_match = xyz_to_rgb(xyz_mono, M_target)
    xyz_monitor = rgb_to_xyz(rgb_to_match, M_target)
    xy_monitor = xyz_to_chromaticity(xyz_monitor)
    print(f"   Monitor RGB to match: {rgb_to_match}")
    print(f"   Monitor produces XYZ: {xyz_monitor}")
    print(f"   Chromaticity: ({xy_monitor[0]:.4f}, {xy_monitor[1]:.4f})")
    print(f"   Match? {np.allclose(xyz_mono, xyz_monitor)}")
    print(f"   Different SPDs, same XYZ — these are metamers.")

    print("\n" + "=" * 60)
    print("  DONE — All calibration steps demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
