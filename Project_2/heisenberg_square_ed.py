from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian


@dataclass(frozen=True)
class SectorSpectrum:
    n_up: int
    sz_total: float
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class SectorGroundEnergy:
    n_up: int
    sz_total: float
    exchange_ground_energy: float


def state_to_bitstring(state: int, n_sites: int) -> str:
    return format(state, f"0{n_sites}b")[::-1]


def square_lattice_bonds(length: int) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for y in range(length):
        for x in range(length):
            site = y * length + x
            bonds.append((site, y * length + ((x + 1) % length)))
            bonds.append((site, ((y + 1) % length) * length + x))
    return bonds


def build_quspin_hamiltonian(
    length: int,
    n_up: int,
    field: float,
    coupling: float = 1.0,
):
    n_sites = length * length
    basis = spin_basis_general(n_sites, Nup=n_up, pauli=False)
    bonds = square_lattice_bonds(length)

    zz_terms = [[coupling, i, j] for i, j in bonds]
    flip_terms = [[0.5 * coupling, i, j] for i, j in bonds]
    field_terms = [[field, i] for i in range(n_sites)]

    static = [["zz", zz_terms], ["+-", flip_terms], ["-+", flip_terms], ["z", field_terms]]
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    ham = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
    return basis, ham


def low_eigenvalues(ham, n_lowest: int) -> np.ndarray:
    dim = ham.Ns
    if dim == 1:
        return np.array([float(ham.toarray()[0, 0])], dtype=float)
    if dim <= 64:
        vals = np.linalg.eigvalsh(ham.toarray())
        return np.sort(np.real(vals))[:n_lowest]

    k = min(n_lowest, dim - 1)
    vals = ham.eigsh(k=k, which="SA", return_eigenvectors=False, tol=1e-10)
    return np.sort(np.real(vals))[:n_lowest]


def analytic_2x2_zero_field_spectrum() -> np.ndarray:
    values = [-4.0] + [-2.0] * 3 + [0.0] * 7 + [2.0] * 5
    return np.array(sorted(values), dtype=float)


def validate_2x2(results_dir: Path) -> None:
    length = 2
    n_sites = length * length
    all_vals: list[float] = []

    for n_up in range(n_sites + 1):
        _, ham = build_quspin_hamiltonian(length=length, n_up=n_up, field=0.0, coupling=1.0)
        vals = np.linalg.eigvalsh(ham.toarray())
        all_vals.extend(np.real_if_close(vals).tolist())

    numerical = np.array(sorted(all_vals), dtype=float)
    analytic = analytic_2x2_zero_field_spectrum()
    diff = float(np.max(np.abs(numerical - analytic)))

    lines = [
        "2x2 periodic square-lattice Heisenberg validation using QuSpin",
        "Hamiltonian: H = J sum_<ij> S_i.S_j + H sum_i S_i^z with J = 1 and H = 0",
        "The analytic spectrum uses the same 2x2 torus bond convention as the code.",
        "",
        f"Maximum absolute difference: {diff:.3e}",
        "",
        "Numerical eigenvalues:",
        " ".join(f"{x: .6f}" for x in numerical),
        "",
        "Analytic eigenvalues:",
        " ".join(f"{x: .6f}" for x in analytic),
    ]
    (results_dir / "validation_2x2.txt").write_text("\n".join(lines) + "\n")


def write_sz0_sector_report(length: int, field: float, results_dir: Path) -> tuple[float, Path]:
    n_sites = length * length
    if n_sites % 2 != 0:
        raise ValueError("The Sz=0 sector only exists for an even number of spins.")

    n_up = n_sites // 2
    basis, ham = build_quspin_hamiltonian(length=length, n_up=n_up, field=field, coupling=1.0)
    matrix = np.asarray(ham.toarray(), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    ground_energy = float(np.real(eigenvalues[0]))
    ground_state = np.real_if_close(eigenvectors[:, 0])

    basis_states = [state_to_bitstring(int(state), n_sites) for state in basis.states]
    matrix_str = np.array2string(matrix, precision=6, suppress_small=True, max_line_width=200)
    gs_str = np.array2string(ground_state, precision=6, suppress_small=True, max_line_width=200)

    lines = [
        f"Sz = 0 sector report for {length}x{length} square lattice (N={n_sites})",
        "Hamiltonian: H = J sum_<ij> S_i.S_j + H sum_i S_i^z with J = 1",
        f"Field H/J = {field:.6f}",
        f"Sector constraint: Nup = {n_up}, Sz_total = 0",
        f"Basis dimension: {basis.Ns}",
        "",
        "Basis states in QuSpin order (bit 1 = spin up, listed as q0 q1 ...):",
    ]
    lines.extend(f"{idx:3d}: |{bitstring}>" for idx, bitstring in enumerate(basis_states))
    lines.extend(
        [
            "",
            "Hamiltonian matrix in the Sz = 0 basis:",
            matrix_str,
            "",
            f"Ground-state energy: {ground_energy:.12f}",
            "Ground-state coefficients in the same basis order:",
            gs_str,
        ]
    )

    out_path = results_dir / f"sz0_sector_{length}x{length}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return ground_energy, out_path


def exact_sector_spectra(
    length: int,
    field: float,
    coupling: float,
    n_lowest: int,
    sector_filter: list[int] | None = None,
) -> list[SectorSpectrum]:
    n_sites = length * length
    allowed = set(sector_filter) if sector_filter is not None else None
    spectra: list[SectorSpectrum] = []

    for n_up in range(n_sites + 1):
        if allowed is not None and n_up not in allowed:
            continue
        _, ham = build_quspin_hamiltonian(length=length, n_up=n_up, field=field, coupling=coupling)
        eigvals = low_eigenvalues(ham, n_lowest=n_lowest)
        spectra.append(
            SectorSpectrum(
                n_up=n_up,
                sz_total=n_up - n_sites / 2,
                eigenvalues=eigvals,
            )
        )
    return spectra


def sector_ground_energies(
    length: int,
    sector_filter: list[int] | None = None,
) -> list[SectorGroundEnergy]:
    n_sites = length * length
    allowed = set(sector_filter) if sector_filter is not None else None
    sector_data: list[SectorGroundEnergy] = []

    for n_up in range(n_sites + 1):
        if allowed is not None and n_up not in allowed:
            continue
        _, ham = build_quspin_hamiltonian(length=length, n_up=n_up, field=0.0, coupling=1.0)
        energy = float(low_eigenvalues(ham, n_lowest=1)[0])
        sector_data.append(
            SectorGroundEnergy(
                n_up=n_up,
                sz_total=n_up - n_sites / 2,
                exchange_ground_energy=energy,
            )
        )
    return sector_data


def scan_fields(
    length: int,
    fields: np.ndarray,
    coupling: float,
    n_lowest: int,
    sector_filter: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_sites = length * length
    low_energies = np.zeros((len(fields), n_lowest), dtype=float)
    magnetization = np.zeros(len(fields), dtype=float)

    zero_field_sectors = sector_ground_energies(length=length, sector_filter=sector_filter)

    for idx, field in enumerate(fields):
        spectra = exact_sector_spectra(
            length=length,
            field=float(field),
            coupling=coupling,
            n_lowest=n_lowest,
            sector_filter=sector_filter,
        )
        all_vals: list[float] = []
        for spec in spectra:
            all_vals.extend(spec.eigenvalues.tolist())

        sorted_vals = np.sort(np.array(all_vals, dtype=float))
        low_energies[idx, :] = sorted_vals[:n_lowest]

        sector_ground = [
            (coupling * sector.exchange_ground_energy + float(field) * sector.sz_total, sector.sz_total)
            for sector in zero_field_sectors
        ]
        best_energy, best_sz = min(sector_ground, key=lambda item: item[0])
        _ = best_energy
        magnetization[idx] = best_sz / n_sites

    return low_energies, magnetization


def ground_state_energy_and_magnetization(
    sector_data: list[SectorGroundEnergy],
    coupling: float,
    field: float,
    n_sites: int,
) -> tuple[float, float, float]:
    best_energy = np.inf
    best_sz = 0.0
    for sector in sector_data:
        energy = coupling * sector.exchange_ground_energy + field * sector.sz_total
        if energy < best_energy:
            best_energy = energy
            best_sz = sector.sz_total
    return float(best_energy), float(best_sz), float(best_sz / n_sites)


def identify_magnetization_staircase(
    sector_data: list[SectorGroundEnergy],
    coupling: float,
    n_sites: int,
) -> list[dict[str, float | str]]:
    sorted_sectors = sorted(sector_data, key=lambda item: item.sz_total)
    jump_fields: set[float] = set()

    for i, left in enumerate(sorted_sectors):
        for right in sorted_sectors[i + 1 :]:
            delta_m = right.sz_total - left.sz_total
            if abs(delta_m) < 1e-12:
                continue
            h_cross = -coupling * (right.exchange_ground_energy - left.exchange_ground_energy) / delta_m
            if h_cross >= 0.0:
                jump_fields.add(float(h_cross))

    sorted_fields = [0.0] + sorted(jump_fields)
    plateaus: list[dict[str, float | str]] = []
    tol = 1e-8

    def active_sector(test_field: float) -> SectorGroundEnergy:
        return min(
            sorted_sectors,
                key=lambda sector: coupling * sector.exchange_ground_energy + test_field * sector.sz_total,
        )

    current_sector = active_sector(0.0)
    lower_edge = 0.0
    for edge in sorted(jump_fields):
        probe = edge + tol
        next_sector = active_sector(probe)
        if next_sector.sz_total != current_sector.sz_total:
            plateaus.append(
                {
                    "h_left": lower_edge,
                    "h_right": edge,
                    "sz_total": current_sector.sz_total,
                    "mz_per_site": current_sector.sz_total / n_sites,
                }
            )
            lower_edge = edge
            current_sector = next_sector

    plateaus.append(
        {
            "h_left": lower_edge,
            "h_right": np.inf,
            "sz_total": current_sector.sz_total,
            "mz_per_site": current_sector.sz_total / n_sites,
        }
    )
    return plateaus


def write_staircase_report(
    path: Path,
    length: int,
    coupling: float,
    sector_data: list[SectorGroundEnergy],
) -> None:
    n_sites = length * length
    plateaus = identify_magnetization_staircase(sector_data, coupling=coupling, n_sites=n_sites)
    lines = [
        f"Magnetization staircase for {length}x{length} square lattice",
        "General finite-field ground state is",
        "E_GS(H,J) = min_M [ J E0(M; J=1, H=0) + H M ]",
        f"with J = {coupling:.6f}",
        "Positive H favors negative Sz_total with this sign convention.",
        "",
        "Sector exchange-only ground energies E0(M; J=1, H=0):",
    ]
    for sector in sorted(sector_data, key=lambda item: item.sz_total):
        lines.append(
            f"Sz_total={sector.sz_total:5.1f}  n_up={sector.n_up:2d}  E0={sector.exchange_ground_energy: .12f}"
        )

    lines.extend(["", "Magnetization plateaus:"])
    for plateau in plateaus:
        right_text = "infinity" if not np.isfinite(float(plateau["h_right"])) else f"{float(plateau['h_right']):.12f}"
        lines.append(
            " ".join(
                [
                    f"H in [{float(plateau['h_left']):.12f}, {right_text})",
                    f"-> Sz_total = {float(plateau['sz_total']):.1f}",
                    f"-> Mz/N = {float(plateau['mz_per_site']):.6f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def plot_exact_staircase(
    path: Path,
    length: int,
    coupling: float,
    sector_data: list[SectorGroundEnergy],
    field_max: float,
) -> None:
    n_sites = length * length
    plateaus = identify_magnetization_staircase(sector_data, coupling=coupling, n_sites=n_sites)

    x_values: list[float] = []
    y_values: list[float] = []
    for plateau in plateaus:
        left = float(plateau["h_left"])
        right = float(plateau["h_right"]) if np.isfinite(float(plateau["h_right"])) else field_max
        right = min(right, field_max)
        if right <= left:
            continue
        x_values.extend([left, right])
        y_values.extend([float(plateau["mz_per_site"]), float(plateau["mz_per_site"])])
        if right >= field_max:
            break

    plt.figure(figsize=(7, 4.5))
    plt.step(x_values, y_values, where="post", linewidth=2.4)
    plt.xlabel("H")
    plt.ylabel(r"$M_z/N$")
    plt.title(f"Exact magnetization staircase for {length}x{length} (J={coupling:g}, +H Sz)")
    plt.xlim(0.0, field_max)
    plt.ylim(-0.52, 0.52)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_spectrum_csv(path: Path, fields: np.ndarray, low_energies: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["h_over_j"] + [f"E_{idx}" for idx in range(low_energies.shape[1])])
        for field, row in zip(fields, low_energies):
            writer.writerow([field, *row.tolist()])


def write_magnetization_csv(path: Path, fields: np.ndarray, magnetization: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["h_over_j", "mz_per_site"])
        for field, mz in zip(fields, magnetization):
            writer.writerow([field, mz])


def plot_spectrum(path: Path, fields: np.ndarray, low_energies: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for level in range(low_energies.shape[1]):
        plt.plot(fields, low_energies[:, level], linewidth=1.8)
    plt.xlabel("H / J")
    plt.ylabel("Energy / J")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_magnetization(path: Path, fields: np.ndarray, magnetization: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.step(fields, magnetization, where="post", linewidth=2.0)
    plt.xlabel("H / J")
    plt.ylabel(r"$\langle M_z \rangle = \langle S^z_{\mathrm{tot}} \rangle / N$")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_sector_report(path: Path, length: int) -> None:
    n_sites = length * length
    lines = [f"Sector dimensions for {length}x{length} (N={n_sites}) using QuSpin Nup blocks"]
    for n_up in range(n_sites + 1):
        lines.append(f"n_up={n_up:2d}  Sz_total={n_up - n_sites / 2:5.1f}  dim={math.comb(n_sites, n_up)}")
    path.write_text("\n".join(lines) + "\n")


def restricted_6x6_sectors(fields: np.ndarray, spinflip_cutoff: int) -> list[int]:
    n_sites = 36
    if np.all(fields >= 0.0):
        return list(range(0, spinflip_cutoff + 1))
    if np.all(fields <= 0.0):
        return list(range(n_sites - spinflip_cutoff, n_sites + 1))
    raise ValueError(
        "Restricted 6x6 scan cannot reliably cover both positive and negative H with a single edge-sector truncation."
    )


def run_scan(length: int, fields: np.ndarray, coupling: float, n_lowest: int, results_dir: Path) -> None:
    low_energies, magnetization = scan_fields(
        length=length,
        fields=fields,
        coupling=coupling,
        n_lowest=n_lowest,
    )
    prefix = f"{length}x{length}"
    write_spectrum_csv(results_dir / f"spectrum_{prefix}.csv", fields, low_energies)
    write_magnetization_csv(results_dir / f"magnetization_{prefix}.csv", fields, magnetization)
    plot_spectrum(
        results_dir / f"spectrum_{prefix}.png",
        fields,
        low_energies,
        title=f"Lowest 5 energy eigenalues in a {prefix} latttice",
    )
    plot_magnetization(
        results_dir / f"magnetization_{prefix}.png",
        fields,
        magnetization,
        title=f"Magnetization in a {prefix} lattice",
    )
    write_sector_report(results_dir / f"sectors_{prefix}.txt", length)
    staircase_data = sector_ground_energies(length=length)
    write_staircase_report(
        results_dir / f"staircase_{prefix}.txt",
        length=length,
        coupling=coupling,
        sector_data=staircase_data,
    )


def run_restricted_6x6(
    fields: np.ndarray,
    coupling: float,
    n_lowest: int,
    spinflip_cutoff: int,
    results_dir: Path,
) -> None:
    allowed = restricted_6x6_sectors(fields=fields, spinflip_cutoff=spinflip_cutoff)
    low_energies, magnetization = scan_fields(
        length=6,
        fields=fields,
        coupling=coupling,
        n_lowest=n_lowest,
        sector_filter=allowed,
    )

    note_lines = [
        "Restricted 6x6 QuSpin calculation",
        f"Allowed n_up sectors: {allowed}",
        "For the Hamiltonian H = J sum S_i.S_j + H sum S_i^z, positive H favors negative Sz_total.",
        "This is exact within the retained sectors only.",
        "The full 6x6 exact diagonalization remains too large without stronger symmetry reduction.",
    ]
    (results_dir / "sectors_6x6_restricted.txt").write_text("\n".join(note_lines) + "\n")
    write_spectrum_csv(results_dir / "spectrum_6x6_restricted.csv", fields, low_energies)
    write_magnetization_csv(results_dir / "magnetization_6x6_restricted.csv", fields, magnetization)
    plot_spectrum(
        results_dir / "spectrum_6x6_restricted.png",
        fields,
        low_energies,
        title=f"Lowest 5 energy eigenalues in a 6x6 latttice",
    )
    plot_magnetization(
        results_dir / "magnetization_6x6_restricted.png",
        fields,
        magnetization,
        title=f"Magnetization in a 6x6 lattice",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuSpin exact diagonalization for the square-lattice spin-1/2 Heisenberg model."
    )
    parser.add_argument("--field-min", type=float, default=0.0, help="Minimum h/J value.")
    parser.add_argument("--field-max", type=float, default=10.0, help="Maximum h/J value.")
    parser.add_argument("--field-points", type=int, default=51, help="Number of field values.")
    parser.add_argument("--lowest", type=int, default=5, help="Number of low-energy levels to save.")
    parser.add_argument("--coupling", type=float, default=1.0, help="Exchange coupling J.")
    parser.add_argument("--single-field", type=float, default=0.0, help="Field H for a single general-case ground-state report.")
    parser.add_argument(
        "--general-length",
        type=int,
        default=4,
        help="Lattice length L for the general finite-(H,J) ground-state and staircase report on an LxL lattice.",
    )
    parser.add_argument(
        "--include-6x6",
        action="store_true",
        help="Run the restricted high-field 6x6 scan.",
    )
    parser.add_argument(
        "--six-six-spinflip-cutoff",
        type=int,
        default=4,
        help="Keep sectors this many or fewer spin flips away from full polarization for 6x6.",
    )
    parser.add_argument(
        "--sz0-length",
        type=int,
        default=2,
        help="Lattice length L for the printed Sz=0 sector report on an LxL lattice.",
    )
    parser.add_argument(
        "--sz0-field",
        type=float,
        default=0.0,
        help="Field h/J used for the printed Sz=0 sector report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fields = np.linspace(args.field_min, args.field_max, args.field_points)

    validate_2x2(results_dir)
    sz0_ground_energy, sz0_report_path = write_sz0_sector_report(
        length=args.sz0_length,
        field=args.sz0_field,
        results_dir=results_dir,
    )
    run_scan(length=4, fields=fields, coupling=args.coupling, n_lowest=args.lowest, results_dir=results_dir)
    sector_data_general = sector_ground_energies(length=args.general_length)
    n_sites_general = args.general_length * args.general_length
    general_energy, general_sz, general_mz = ground_state_energy_and_magnetization(
        sector_data=sector_data_general,
        coupling=args.coupling,
        field=args.single_field,
        n_sites=n_sites_general,
    )
    staircase_path = results_dir / f"staircase_{args.general_length}x{args.general_length}.txt"
    write_staircase_report(
        staircase_path,
        length=args.general_length,
        coupling=args.coupling,
        sector_data=sector_data_general,
    )
    staircase_plot_path = results_dir / f"staircase_{args.general_length}x{args.general_length}.png"
    plot_exact_staircase(
        staircase_plot_path,
        length=args.general_length,
        coupling=args.coupling,
        sector_data=sector_data_general,
        field_max=args.field_max * max(args.coupling, 1.0),
    )

    summary_lines = [
        "Project 2 Heisenberg-model QuSpin run complete.",
        f"Results directory: {results_dir}",
        f"Saved Sz=0 sector report: {sz0_report_path}",
        f"Sz=0 ground-state energy: {sz0_ground_energy:.12f}",
        f"General {args.general_length}x{args.general_length} ground-state energy at J={args.coupling:.6f}, H={args.single_field:.6f}: {general_energy:.12f}",
        f"Corresponding Sz_total={general_sz:.1f}, Mz/N={general_mz:.6f}",
        f"Saved staircase report: {staircase_path}",
        f"Saved staircase plot: {staircase_plot_path}",
        "Completed: 2x2 validation and 4x4 full exact scan.",
    ]

    run_restricted_6x6(
        fields=fields,
        coupling=args.coupling,
        n_lowest=args.lowest,
        spinflip_cutoff=args.six_six_spinflip_cutoff,
        results_dir=results_dir,
    )
    summary_lines.append(
        f"Completed: restricted 6x6 scan with spin-flip cutoff {args.six_six_spinflip_cutoff}."
    )
    
    (results_dir / "run_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
