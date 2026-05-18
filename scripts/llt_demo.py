"""Demonstrate the nonlinear lifting-line solver on the three canonical wings.

Four artifacts:

1. **Elliptic-wing identity figure** — ``CDi`` vs ``CL²`` for an AR=8 elliptic
   wing on top of the analytical line ``CDi = CL² / (π · AR)``. The solver
   trace must sit on top of the textbook line. This is the headline "the
   solver is right" demo.
2. **Lift-curve through stall** — ``CL(α)`` for a rectangular AR=10 wing
   using the Hoerner flat-plate sectional polar. Demonstrates the nonlinear
   solver gracefully handling stall and post-stall, which the linear LLT
   cannot.
3. **Span-efficiency vs aspect ratio** — comparison of elliptic, rectangular,
   and tapered (λ=0.5) planforms across AR ∈ [4, 20]. Reproduces the
   classical-LLT result that elliptic wings have ``e ≈ 1`` while rectangular
   and tapered wings sit a few percent below.
4. **NeuralFoil-coupled wing analysis** — ``CL(α)`` and the drag polar
   ``CD(CL)`` for a NACA 2412 wing (AR=8, rectangular) with NeuralFoil
   supplying the 2D viscous sectional polar. Generated only if ``neuralfoil``
   is installed; this is the headline demonstration that the lifting-line
   solver couples a data-driven 2D viscous model with classical 3D
   inviscid wing theory to produce a viscous-3D drag buildup.

Usage::

    python scripts/llt_demo.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerosurrogate.lifting_line import (
    FlatPlatePostStall,
    ThinAirfoilSection,
    Wing,
    alpha_sweep,
    solve_lifting_line,
)

try:
    from aerosurrogate.lifting_line import NeuralFoilSection
    _HAS_NEURALFOIL = True
except ImportError:  # pragma: no cover
    _HAS_NEURALFOIL = False

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def figure_elliptic_identity() -> None:
    """Plot CDi vs CL² for an elliptic wing on top of analytical line."""
    AR = 8.0
    span = 10.0
    c_root = 4.0 * span / (math.pi * AR)
    wing = Wing.elliptic(span=span, root_chord=c_root, n_sections=80)
    section = ThinAirfoilSection()

    alphas = np.linspace(-2, 10, 25)
    out = alpha_sweep(wing, alphas, section)

    cl = out["CL"]
    cdi = out["CDi"]
    theory = cl ** 2 / (math.pi * AR)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(cl ** 2, theory, "k--", lw=1.5, label=r"$C_L^2 / (\pi \cdot AR)$ (theory)")
    ax.plot(cl ** 2, cdi, "o", color="tab:red", ms=6, label="LLT solver")
    ax.set_xlabel(r"$C_L^2$")
    ax.set_ylabel(r"$C_{D_i}$")
    ax.set_title(f"Elliptic wing, AR={AR:.0f}: induced drag matches the analytical identity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "llt_elliptic_identity.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    rel_err = float(np.max(np.abs(cdi - theory) / np.maximum(theory, 1e-12)))
    print(f"  elliptic identity  →  {out_path}   (max rel. error vs theory: {rel_err:.2e})")


def figure_lift_curve_through_stall() -> None:
    """Plot CL(α) for rectangular AR=10 wing through and past stall."""
    wing = Wing.rectangular(span=10.0, chord=1.0, n_sections=80)
    section = FlatPlatePostStall()
    alphas = np.arange(0.0, 56.0, 2.0)
    out = alpha_sweep(wing, alphas, section, warm_start=True)

    # 2D sectional polar for comparison
    cl_2d = np.sin(2.0 * np.deg2rad(alphas))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(alphas, cl_2d, "k--", lw=1.5, label=r"$C_l$ (2D flat plate, $\sin 2\alpha$)")
    ax.plot(alphas, out["CL"], "o-", color="tab:blue", ms=4, lw=1.5, label=r"$C_L$ (3D wing, LLT)")
    ax.axvline(45, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"Lift coefficient")
    ax.set_title("Rectangular AR=10 wing through stall (Hoerner flat-plate polar)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "llt_lift_curve_stall.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  lift curve through stall  →  {out_path}")


def figure_span_efficiency_vs_ar() -> None:
    """Plot span efficiency vs AR for elliptic / rectangular / tapered planforms."""
    ARs = np.array([4, 6, 8, 10, 12, 16, 20], dtype=float)
    e_ellip = np.zeros_like(ARs)
    e_rect = np.zeros_like(ARs)
    e_taper = np.zeros_like(ARs)
    span = 10.0
    section = ThinAirfoilSection()

    for k, AR in enumerate(ARs):
        # elliptic
        c_root = 4.0 * span / (math.pi * AR)
        wing = Wing.elliptic(span=span, root_chord=c_root, n_sections=80)
        e_ellip[k] = solve_lifting_line(wing, 5.0, section).span_efficiency
        # rectangular
        chord = span / AR
        wing = Wing.rectangular(span=span, chord=chord, n_sections=80)
        e_rect[k] = solve_lifting_line(wing, 5.0, section).span_efficiency
        # tapered λ=0.5 (root chord chosen to match AR)
        # S = b · c_root · (1+λ)/2,  AR = b/S * b  =>  c_root = 2 · S/b · 1/(1+λ) → solve via S = b²/AR
        S = span ** 2 / AR
        c_root = 2.0 * S / (span * 1.5)
        wing = Wing.tapered(span=span, root_chord=c_root, taper_ratio=0.5, n_sections=80)
        e_taper[k] = solve_lifting_line(wing, 5.0, section).span_efficiency

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(ARs, e_ellip, "o-", color="tab:red", lw=1.5, label="Elliptic")
    ax.plot(ARs, e_taper, "s-", color="tab:green", lw=1.5, label=r"Tapered ($\lambda$=0.5)")
    ax.plot(ARs, e_rect, "^-", color="tab:blue", lw=1.5, label="Rectangular")
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Aspect ratio (AR)")
    ax.set_ylabel("Span efficiency e")
    ax.set_ylim(0.84, 1.02)
    ax.set_title("Span efficiency vs aspect ratio (LLT, ThinAirfoil section, α=5°)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = FIG_DIR / "llt_span_efficiency.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  span efficiency vs AR  →  {out_path}")
    print(f"    elliptic: {e_ellip.round(4).tolist()}")
    print(f"    tapered:  {e_taper.round(4).tolist()}")
    print(f"    rectang:  {e_rect.round(4).tolist()}")


def figure_neuralfoil_wing() -> None:
    """``CL(α)`` and ``CD(CL)`` for a NACA 2412 wing using NeuralFoil polars.

    Rectangular AR=8 wing at Re=3×10⁶. Three traces:

    * **2D NeuralFoil** — what NeuralFoil reports for the airfoil alone.
      This is what the project's 2D-only scope could see.
    * **3D LLT + NeuralFoil** — the same airfoil installed in a finite wing
      and solved with the LLT. Lift drops vs the 2D curve because of the
      reduced effective angle (downwash); drag rises sharply at higher CL
      because induced drag now appears.
    * **3D LLT (induced drag only)** — for the drag polar, also plot
      ``CDi`` alone, so the contributions split visibly.

    The point of this figure: it's the headline of the whole project, the
    one thing a 2D NeuralFoil pipeline absolutely cannot show.
    """
    if not _HAS_NEURALFOIL:  # pragma: no cover
        print("  neuralfoil not installed — skipping NeuralFoil-coupled figure.")
        return

    wing = Wing.rectangular(span=8.0, chord=1.0, n_sections=80)   # AR = 8
    section = NeuralFoilSection(airfoil_name="naca2412", model_size="medium")
    Re = 3.0e6

    alphas = np.arange(-2.0, 15.0, 1.0)
    out = alpha_sweep(wing, alphas, section, V_inf=30.0, Re_ref=Re, warm_start=True)

    # 2D NeuralFoil polar at the same α / Re (no LLT)
    cl_2d = section.cl(alphas, np.full_like(alphas, Re))
    cd_2d = section.cd(alphas, np.full_like(alphas, Re))

    # CL(α) plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(alphas, cl_2d, "k--", lw=1.5, label="2D NeuralFoil (airfoil alone)")
    ax1.plot(alphas, out["CL"], "o-", color="tab:red", ms=4, lw=1.5,
             label="3D LLT + NeuralFoil (finite wing)")
    ax1.set_xlabel(r"$\alpha$ (deg)")
    ax1.set_ylabel(r"Lift coefficient")
    ax1.set_title(f"NACA 2412, rectangular AR={wing.aspect_ratio:.0f}, Re={Re:.1e}")
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axvline(0, color="gray", lw=0.5)

    # CD(CL) drag polar
    ax2.plot(cd_2d, cl_2d, "k--", lw=1.5, label="2D NeuralFoil (profile drag only)")
    ax2.plot(out["CD"], out["CL"], "o-", color="tab:red", ms=4, lw=1.5,
             label="3D LLT total drag = profile + induced")
    ax2.plot(out["CDi"], out["CL"], "s-", color="tab:blue", ms=3, lw=1.0, alpha=0.7,
             label="Induced drag $C_{D_i}$ (3D only)")
    ax2.set_xlabel(r"Drag coefficient $C_D$")
    ax2.set_ylabel(r"Lift coefficient $C_L$")
    ax2.set_title("Drag polar: what 2D cannot tell you")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(left=0)

    fig.suptitle(
        "NeuralFoil-coupled 3D wing analysis (LLT + viscous 2D polar)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = FIG_DIR / "llt_neuralfoil_wing.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    # Report the headline number: induced drag share at CL = 0.5.
    cl_target = 0.5
    if (out["CL"][0] < cl_target < out["CL"][-1]):
        idx = int(np.argmin(np.abs(out["CL"] - cl_target)))
        cdi_share = out["CDi"][idx] / out["CD"][idx]
        print(f"  NeuralFoil-coupled wing  →  {out_path}")
        print(f"    at CL≈{out['CL'][idx]:.2f}: CDi = {out['CDi'][idx]:.5f}, "
              f"CD_total = {out['CD'][idx]:.5f}, CDi share = {cdi_share*100:.1f}%")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating LLT demo figures →", FIG_DIR)
    figure_elliptic_identity()
    figure_lift_curve_through_stall()
    figure_span_efficiency_vs_ar()
    figure_neuralfoil_wing()
    print("Done.")


if __name__ == "__main__":
    main()
