# Project 2 - Phase 1 and Phase 2

This folder now contains both project tasks:

- `heisenberg_square_ed.py`
  QuSpin-based exact diagonalization for the spin-1/2 Heisenberg model on a
  periodic square lattice.
- `bose_fermi_calculator.py`
  A Streamlit calculator for Bose-Fermi mixture density profiles in the
  Thomas-Fermi / local-density approximation.

## Phase 1: Heisenberg model with QuSpin

The code implements

\[
\hat H = J \sum_{\langle i,j\rangle}\hat{\mathbf S}_i\cdot \hat{\mathbf S}_j
      - h \sum_i \hat S_i^z
\]

with periodic boundary conditions on `2x2`, `4x4`, and an optional restricted
`6x6` scan.

What it does:

- validates `2x2` against the analytic zero-field spectrum
- computes the lowest few levels for `4x4`
- computes the magnetization curve `M_z/N` versus `h/J`
- uses `QuSpin` basis and Hamiltonian objects, as recommended in the guide

Run it from the repository root:

```bash
python3 Project_2/heisenberg_square_ed.py
python3 Project_2/heisenberg_square_ed.py --include-6x6
```

Outputs are written to `Project_2/results/`.

Note on `6x6`:
the full `N=36` exact problem is still too large without more aggressive
symmetry reduction, so the script keeps that part restricted to sectors near
full polarization. That is exact within the retained sectors and useful for the
high-field regime.

## Phase 2: Bose-Fermi mixture calculator

The calculator solves coupled static density equations:

- bosons:
  `mu_B = V_B(r) + g_B n_B(r) + g_BF n_F(r)`
- fermions:
  `mu_F = alpha_F n_F(r)^(2/3) + V_F(r) + g_BF n_B(r)`

with

`alpha_F = (hbar^2 / 2m_F) (6 pi^2)^(2/3)`.

Assumptions:

- spherically symmetric harmonic traps
- Thomas-Fermi treatment for bosons
- local-density approximation for fermions
- dimensionless default units with `hbar = 1`

Run the calculator:

```bash
streamlit run Project_2/bose_fermi_calculator.py
```

The app lets you input:

- `N_B`, `N_F`
- `m_B`, `m_F`
- `omega_B`, `omega_F`
- `g_B`
- a reference `g_BF`

and it automatically compares three representative regimes:

- negative `g_BF`
- zero `g_BF`
- positive `g_BF`

The plots and table show:

- density profiles
- Thomas-Fermi radii
- trap-center densities
- overlap between species

so you can comment on cloud size changes and central compression/depletion as
`g_BF` moves away from zero.
