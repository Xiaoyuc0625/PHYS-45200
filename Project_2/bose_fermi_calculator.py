from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = PROJECT_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

HBAR = 1.0
PI = np.pi


@dataclass(frozen=True)
class MixtureParams:
    n_bosons: float
    n_fermions: float
    m_b: float
    m_f: float
    omega_b: float
    omega_f: float
    g_b: float
    g_bf: float
    r_max: float
    grid_points: int
    mix_weight: float
    max_iter: int
    tol: float


def harmonic_potential(mass: float, omega: float, r: np.ndarray) -> np.ndarray:
    return 0.5 * mass * omega**2 * r**2


def fermi_prefactor(m_f: float) -> float:
    return (HBAR**2 / (2.0 * m_f)) * (6.0 * PI**2) ** (2.0 / 3.0)


def trap_integral(r: np.ndarray, density: np.ndarray) -> float:
    return float(4.0 * PI * np.trapezoid(r**2 * density, r))


def solve_boson_density(
    mu_b: float,
    g_b: float,
    v_b: np.ndarray,
    n_f: np.ndarray,
    g_bf: float,
) -> np.ndarray:
    if g_b <= 0.0:
        raise ValueError("g_b must be positive for the Thomas-Fermi boson profile.")
    return np.maximum((mu_b - v_b - g_bf * n_f) / g_b, 0.0)


def solve_fermion_density(
    mu_f: float,
    alpha_f: float,
    v_f: np.ndarray,
    n_b: np.ndarray,
    g_bf: float,
) -> np.ndarray:
    arg = np.maximum(mu_f - v_f - g_bf * n_b, 0.0)
    return (arg / alpha_f) ** 1.5


def normalize_species(
    target_number: float,
    density_builder,
    r: np.ndarray,
    lower: float,
    upper: float,
    tol: float,
    max_iter: int,
) -> tuple[float, np.ndarray]:
    lo = lower
    hi = upper
    density_hi = density_builder(hi)
    number_hi = trap_integral(r, density_hi)

    expand_count = 0
    while number_hi < target_number and expand_count < 80:
        hi *= 2.0 if hi > 0 else 1.0
        if hi == 0:
            hi = 1.0
        density_hi = density_builder(hi)
        number_hi = trap_integral(r, density_hi)
        expand_count += 1

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        density_mid = density_builder(mid)
        number_mid = trap_integral(r, density_mid)
        if abs(number_mid - target_number) <= tol * max(target_number, 1.0):
            return mid, density_mid
        if number_mid < target_number:
            lo = mid
        else:
            hi = mid

    final_mu = 0.5 * (lo + hi)
    return final_mu, density_builder(final_mu)


def initial_guesses(params: MixtureParams, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_b = harmonic_potential(params.m_b, params.omega_b, r)
    v_f = harmonic_potential(params.m_f, params.omega_f, r)
    alpha_f = fermi_prefactor(params.m_f)

    mu_b, n_b = normalize_species(
        target_number=params.n_bosons,
        density_builder=lambda mu: np.maximum((mu - v_b) / params.g_b, 0.0),
        r=r,
        lower=0.0,
        upper=max(v_b[-1] + params.g_b, 1.0),
        tol=params.tol,
        max_iter=params.max_iter,
    )
    _ = mu_b
    mu_f, n_f = normalize_species(
        target_number=params.n_fermions,
        density_builder=lambda mu: (np.maximum(mu - v_f, 0.0) / alpha_f) ** 1.5,
        r=r,
        lower=0.0,
        upper=max(v_f[-1] + alpha_f, 1.0),
        tol=params.tol,
        max_iter=params.max_iter,
    )
    _ = mu_f
    return n_b, n_f


def solve_coupled_profiles(params: MixtureParams) -> dict[str, np.ndarray | float | int | bool]:
    r = np.linspace(0.0, params.r_max, params.grid_points)
    v_b = harmonic_potential(params.m_b, params.omega_b, r)
    v_f = harmonic_potential(params.m_f, params.omega_f, r)
    alpha_f = fermi_prefactor(params.m_f)
    n_b, n_f = initial_guesses(params, r)

    mu_b = 0.0
    mu_f = 0.0
    converged = False
    max_change = np.inf

    for iteration in range(1, params.max_iter + 1):
        mu_b, new_b = normalize_species(
            target_number=params.n_bosons,
            density_builder=lambda mu: solve_boson_density(mu, params.g_b, v_b, n_f, params.g_bf),
            r=r,
            lower=0.0,
            upper=max(v_b[-1] + params.g_b * params.n_bosons, 1.0),
            tol=params.tol,
            max_iter=params.max_iter,
        )
        mixed_b = params.mix_weight * new_b + (1.0 - params.mix_weight) * n_b

        mu_f, new_f = normalize_species(
            target_number=params.n_fermions,
            density_builder=lambda mu: solve_fermion_density(mu, alpha_f, v_f, mixed_b, params.g_bf),
            r=r,
            lower=0.0,
            upper=max(v_f[-1] + alpha_f * params.n_fermions ** (2.0 / 3.0), 1.0),
            tol=params.tol,
            max_iter=params.max_iter,
        )
        mixed_f = params.mix_weight * new_f + (1.0 - params.mix_weight) * n_f

        max_change = float(max(np.max(np.abs(mixed_b - n_b)), np.max(np.abs(mixed_f - n_f))))
        n_b, n_f = mixed_b, mixed_f

        if max_change < params.tol:
            converged = True
            break

    overlap = trap_integral(r, np.minimum(n_b, n_f))
    return {
        "r": r,
        "n_b": n_b,
        "n_f": n_f,
        "mu_b": mu_b,
        "mu_f": mu_f,
        "center_b": float(n_b[0]),
        "center_f": float(n_f[0]),
        "overlap": overlap,
        "iterations": iteration,
        "converged": converged,
        "tf_radius_b": estimate_radius(r, n_b),
        "tf_radius_f": estimate_radius(r, n_f),
    }


def estimate_radius(r: np.ndarray, density: np.ndarray, threshold: float = 1e-6) -> float:
    if density.max() <= 0.0:
        return 0.0
    cutoff = threshold * density.max()
    support = np.where(density > cutoff)[0]
    if len(support) == 0:
        return 0.0
    return float(r[support[-1]])


def choose_regime_values(base_gbf: float) -> dict[str, float]:
    scale = max(abs(base_gbf), 0.2)
    return {
        "negative": -scale,
        "zero": 0.0,
        "positive": scale,
    }


def render_profile_plot(results: dict[str, dict[str, np.ndarray | float | int | bool]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    colors = {"boson": "#1f77b4", "fermion": "#d62728"}
    for ax, (label, result) in zip(axes, results.items()):
        r = result["r"]
        ax.plot(r, result["n_b"], color=colors["boson"], linewidth=2.2, label="Bosons")
        ax.plot(r, result["n_f"], color=colors["fermion"], linewidth=2.2, linestyle="--", label="Fermions")
        ax.set_title(f"{label.capitalize()} g_BF")
        ax.set_xlabel("r")
        ax.grid(alpha=0.25)
        ax.text(
            0.04,
            0.96,
            "\n".join(
                [
                    f"g_BF = {result['g_bf']:.3f}",
                    f"R_B = {result['tf_radius_b']:.3f}",
                    f"R_F = {result['tf_radius_f']:.3f}",
                    f"n_B(0) = {result['center_b']:.3f}",
                    f"n_F(0) = {result['center_f']:.3f}",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right")
    fig.suptitle("Bose-Fermi mixture density profiles in Thomas-Fermi / LDA")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_summary_table(results: dict[str, dict[str, np.ndarray | float | int | bool]]) -> None:
    rows = []
    for label, result in results.items():
        rows.append(
            {
                "Regime": label,
                "g_BF": result["g_bf"],
                "mu_B": result["mu_b"],
                "mu_F": result["mu_f"],
                "R_B": result["tf_radius_b"],
                "R_F": result["tf_radius_f"],
                "n_B(0)": result["center_b"],
                "n_F(0)": result["center_f"],
                "Overlap": result["overlap"],
                "Iterations": result["iterations"],
                "Converged": result["converged"],
            }
        )
    st.dataframe(rows, use_container_width=True)


st.set_page_config(page_title="Project 2 Bose-Fermi Calculator", layout="wide")
st.title("Project 2 - Bose-Fermi Mixture Calculator")
st.markdown(
    """
This calculator uses a static Thomas-Fermi / local-density approximation for a
co-trapped Bose-Fermi mixture in a spherically symmetric harmonic trap.

- Bosons obey `mu_B = V_B(r) + g_B n_B(r) + g_BF n_F(r)`.
- Fermions obey `mu_F = alpha_F n_F(r)^(2/3) + V_F(r) + g_BF n_B(r)`, with
  `alpha_F = (hbar^2 / 2m_F) (6 pi^2)^(2/3)`.
- Units are dimensionless with `hbar = 1`.
"""
)

with st.sidebar:
    st.header("Inputs")
    n_bosons = st.number_input("N_B", min_value=1.0, value=5000.0, step=100.0)
    n_fermions = st.number_input("N_F", min_value=1.0, value=5000.0, step=100.0)
    m_b = st.number_input("m_B", min_value=0.01, value=1.0, step=0.05)
    m_f = st.number_input("m_F", min_value=0.01, value=1.0, step=0.05)
    omega_b = st.number_input("omega_B", min_value=0.01, value=1.0, step=0.05)
    omega_f = st.number_input("omega_F", min_value=0.01, value=1.0, step=0.05)
    g_b = st.number_input("g_B", min_value=0.001, value=0.08, step=0.01, format="%.4f")
    g_bf = st.number_input("reference g_BF", value=0.04, step=0.01, format="%.4f")
    r_max = st.number_input("r_max", min_value=0.5, value=10.0, step=0.5)
    # grid_points = int(st.number_input("grid points", min_value=200.0, value=1200.0, step=100.0))
    # mix_weight = st.slider("density mixing", min_value=0.05, max_value=1.0, value=0.35, step=0.05)
    # tol = st.number_input("convergence tolerance", min_value=1e-8, value=1e-5, format="%.8f")
    # max_iter = int(st.number_input("max iterations", min_value=20.0, value=120.0, step=10.0))

params = MixtureParams(
    n_bosons=n_bosons,
    n_fermions=n_fermions,
    m_b=m_b,
    m_f=m_f,
    omega_b=omega_b,
    omega_f=omega_f,
    g_b=g_b,
    g_bf=g_bf,
    r_max=r_max,
    grid_points=1200,
    mix_weight=0.35,
    max_iter=120,
    tol=1e-5,
)

regimes = choose_regime_values(params.g_bf)
results: dict[str, dict[str, np.ndarray | float | int | bool]] = {}
for label, gbf_value in regimes.items():
    result = solve_coupled_profiles(
        MixtureParams(
            n_bosons=params.n_bosons,
            n_fermions=params.n_fermions,
            m_b=params.m_b,
            m_f=params.m_f,
            omega_b=params.omega_b,
            omega_f=params.omega_f,
            g_b=params.g_b,
            g_bf=gbf_value,
            r_max=params.r_max,
            grid_points=params.grid_points,
            mix_weight=params.mix_weight,
            max_iter=params.max_iter,
            tol=params.tol,
        )
    )
    result["g_bf"] = gbf_value
    results[label] = result

left, right = st.columns([1.7, 1.0])
with left:
    render_profile_plot(results)

with right:
    st.subheader("Profile Summary")
    render_summary_table(results)
    st.markdown(
        """
- Negative `g_BF`: attraction tends to pull both species inward and raise the center density.
- Zero `g_BF`: the species decouple and recover their separate trap profiles.
- Positive `g_BF`: repulsion tends to broaden the clouds and reduce overlap, signaling phase separation.
"""
    )

st.subheader("Chosen Parameters")
st.write(
    {
        "N_B": params.n_bosons,
        "N_F": params.n_fermions,
        "m_B": params.m_b,
        "m_F": params.m_f,
        "omega_B": params.omega_b,
        "omega_F": params.omega_f,
        "g_B": params.g_b,
        "reference_g_BF": params.g_bf,
        "units": "dimensionless, with hbar = 1",
    }
)
