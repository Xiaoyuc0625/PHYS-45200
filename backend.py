from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

MAX_QUBITS = 10
DEFAULT_SHOTS = 1024
_SIM = AerSimulator()
_STATEVEC_SIM = AerSimulator(method="statevector")


class BackendError(ValueError):
    """Raised when an invalid circuit is submitted to the backend."""


@dataclass
class ProtocolStep:
    number: int
    title: str
    description: str
    state_circuit: QuantumCircuit
    display_circuit: QuantumCircuit
    statevector_expression: str
    statevector_latex: str
    measurement_counts: Dict[str, int]


@dataclass
class TeleportationResult:
    title: str
    input_state_latex: str
    circuit: QuantumCircuit
    full_counts: Dict[str, int]
    bob_counts: Dict[str, int]
    expected_bob_probabilities: Dict[str, float]


@dataclass
class LongDistanceCNOTResult:
    target_circuit: QuantumCircuit
    routed_swap_circuit: QuantumCircuit
    decomposed_circuit: QuantumCircuit
    swap_count: int
    cnot_count: int


@dataclass
class HubbardReport:
    trotter_circuit: QuantumCircuit
    noninteracting_taus: List[float]
    noninteracting_site2_probs: List[float]
    strong_taus: List[float]
    strong_remain_probs: List[float]
    strong_transition_probs: List[float]



def validate_qubit_count(circuit: QuantumCircuit, max_qubits: int = MAX_QUBITS) -> None:
    if circuit.num_qubits > max_qubits:
        raise BackendError(
            f"Circuit has {circuit.num_qubits} qubits, exceeding the {max_qubits}-qubit limit."
        )



def ensure_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of the circuit with measurements on all qubits."""
    measured = circuit.copy()
    if measured.num_clbits < measured.num_qubits:
        extra = measured.num_qubits - measured.num_clbits
        measured.add_register(ClassicalRegister(extra, "c_extra"))
    if not any(instr.operation.name == "measure" for instr in measured.data):
        measured.measure(range(measured.num_qubits), range(measured.num_qubits))
    return measured



def run_circuit(circuit: QuantumCircuit, shots: int = DEFAULT_SHOTS) -> Dict[str, int]:
    """Run a circuit on AerSimulator and return the measurement histogram."""
    validate_qubit_count(circuit)
    measured = ensure_measurements(circuit)
    compiled = transpile(measured, _SIM)
    result = _SIM.run(compiled, shots=shots).result()
    return result.get_counts()



def statevector_after(circuit: QuantumCircuit) -> Statevector:
    validate_qubit_count(circuit)
    clean = circuit.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(clean)



def nonzero_amplitudes(statevector: Statevector, tol: float = 1e-10) -> List[Tuple[str, complex]]:
    out: List[Tuple[str, complex]] = []
    n = int(np.log2(len(statevector.data)))
    for idx, amp in enumerate(statevector.data):
        if abs(amp) > tol:
            out.append((format(idx, f"0{n}b"), complex(amp)))
    return out


def display_bitstring(bitstring: str) -> str:
    """Convert Qiskit state/count labels to q0...qN display order."""
    return bitstring[::-1]


def statevector_expression(circuit: QuantumCircuit) -> str:
    terms = [
        f"|{display_bitstring(bitstring)}>"
        for bitstring, _ in nonzero_amplitudes(statevector_after(circuit))
    ]
    return "( " + " + ".join(terms) + " ) / sqrt(2)"


def statevector_latex(circuit: QuantumCircuit) -> str:
    terms = [
        rf"\left|{display_bitstring(bitstring)}\right\rangle"
        for bitstring, _ in nonzero_amplitudes(statevector_after(circuit))
    ]
    return r"\frac{1}{\sqrt{2}}\left(" + " + ".join(terms) + r"\right)"


def measurement_counts(circuit: QuantumCircuit, shots: int = DEFAULT_SHOTS) -> Dict[str, int]:
    counts = run_circuit(circuit, shots=shots)
    return {
        display_bitstring(bitstring): count
        for bitstring, count in sorted(counts.items())
    }


def measurement_counts_1024(circuit: QuantumCircuit) -> Dict[str, int]:
    return measurement_counts(circuit, shots=DEFAULT_SHOTS)



def bell_param_circuit(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc



def ghz10_circuit(measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(10, 10 if measure else 0)
    qc.h(0)
    for i in range(9):
        qc.cx(i, i + 1)
    if measure:
        qc.measure(range(10), range(10))
    return qc



def prepare_superposition_201_425() -> QuantumCircuit:
    """Prepare 2^-1/2 (|201> + |425>) on 10 qubits.

    Qiskit uses little-endian ordering in statevector labels. We prepare basis states
    corresponding to bitstrings 0011001001 (= 201) and 0110101001 (= 425).
    """
    qc = QuantumCircuit(10)
    qc.x([2, 3, 6, 9])
    qc.h(1)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.barrier(label="|201> + |425>")
    return qc



def entangling_chain() -> QuantumCircuit:
    qc = QuantumCircuit(10)
    for i in range(9):
        qc.cx(i, i + 1)
    return qc



def entangling_chain_inverse() -> QuantumCircuit:
    qc = QuantumCircuit(10)
    for i in reversed(range(9)):
        qc.cx(i, i + 1)
    return qc


def q14_display_circuit(include_chain: bool = False, include_inverse: bool = False) -> QuantumCircuit:
    """Return the circuit shown for Q1.4."""
    qc = QuantumCircuit(10, 10)
    qc.compose(prepare_superposition_201_425(), inplace=True)
    if include_chain:
        qc.barrier(label="9 CNOT chain")
        for i in range(9):
            qc.cx(i, i + 1)
    if include_inverse:
        qc.barrier(label="inverse chain")
        for i in reversed(range(9)):
            qc.cx(i, i + 1)
    qc.measure(range(10), range(10))
    return qc


def q14_protocol_steps() -> List[ProtocolStep]:
    prep = prepare_superposition_201_425()
    chain = entangling_chain()
    recover = entangling_chain_inverse()

    step1 = prep
    step2 = prep.compose(chain)
    step3 = prep.compose(chain).compose(recover)

    step_data = [
        (
            1,
            "Prepare the initial superposition",
            r"Initialize all 10 qubits in $2^{-1/2}(|0011001001\rangle + |0110101001\rangle)$.",
            step1,
            q14_display_circuit(),
        ),
        (
            2,
            "Apply the 9-gate CNOT chain",
            r"Apply $\mathrm{CNOT}(q_0 \rightarrow q_1), \mathrm{CNOT}(q_1 \rightarrow q_2), \ldots, \mathrm{CNOT}(q_8 \rightarrow q_9)$.",
            step2,
            q14_display_circuit(include_chain=True),
        ),
        (
            3,
            "Reverse the chain and recover the state",
            r"Apply the inverse sequence in reverse order, from $\mathrm{CNOT}(q_8 \rightarrow q_9)$ back to $\mathrm{CNOT}(q_0 \rightarrow q_1)$.",
            step3,
            q14_display_circuit(include_chain=True, include_inverse=True),
        ),
    ]

    return [
        ProtocolStep(
            number=number,
            title=title,
            description=description,
            state_circuit=state_circuit,
            display_circuit=display_circuit,
            statevector_expression=statevector_expression(state_circuit),
            statevector_latex=statevector_latex(state_circuit),
            measurement_counts=measurement_counts_1024(state_circuit),
        )
        for number, title, description, state_circuit, display_circuit in step_data
    ]



def long_distance_cnot_q0_q4_linear() -> QuantumCircuit:
    """Implements CNOT(q0 -> q4) on a linear chain 0-1-2-3-4 using SWAP routing."""
    qc = QuantumCircuit(5, 5)
    qc.swap(3, 4)
    qc.swap(2, 3)
    qc.swap(1, 2)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.swap(2, 3)
    qc.swap(3, 4)
    qc.measure(range(5), range(5))
    return qc


def q22_target_cnot_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(5)
    qc.cx(0, 4)
    return qc


def q22_routed_swap_circuit(measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(5, 5 if measure else 0)
    qc.swap(3, 4)
    qc.swap(2, 3)
    qc.swap(1, 2)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.swap(2, 3)
    qc.swap(3, 4)
    if measure:
        qc.measure(range(5), range(5))
    return qc


def append_swap_decomposed(qc: QuantumCircuit, q_left: int, q_right: int) -> None:
    qc.cx(q_left, q_right)
    qc.cx(q_right, q_left)
    qc.cx(q_left, q_right)


def q22_decomposed_cnot_circuit(measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(5, 5 if measure else 0)
    append_swap_decomposed(qc, 3, 4)
    append_swap_decomposed(qc, 2, 3)
    append_swap_decomposed(qc, 1, 2)
    qc.cx(0, 1)
    append_swap_decomposed(qc, 1, 2)
    append_swap_decomposed(qc, 2, 3)
    append_swap_decomposed(qc, 3, 4)
    if measure:
        qc.measure(range(5), range(5))
    return qc


def q22_long_distance_cnot_result() -> LongDistanceCNOTResult:
    return LongDistanceCNOTResult(
        target_circuit=q22_target_cnot_circuit(),
        routed_swap_circuit=q22_routed_swap_circuit(),
        decomposed_circuit=q22_decomposed_cnot_circuit(),
        swap_count=6,
        cnot_count=19,
    )



def teleportation_circuit(alpha: complex, beta: complex, measure_alice: bool = True) -> QuantumCircuit:
    if not np.isclose(abs(alpha) ** 2 + abs(beta) ** 2, 1.0):
        raise ValueError("Input amplitudes must be normalized.")

    qc = QuantumCircuit(3, 3)

    qc.barrier(label="Input state")
    qc.initialize([alpha, beta], 0)

    qc.barrier(label="Bell State Preparation")
    qc.h(1)
    qc.cx(1, 2)

    qc.barrier(label="Bell Measurement")
    qc.cx(0, 1)
    qc.h(0)

    if measure_alice:
        qc.measure(0, 0)
        qc.measure(1, 1)

        with qc.if_test((qc.clbits[1], 1)):
            qc.x(2)
        with qc.if_test((qc.clbits[0], 1)):
            qc.z(2)

        qc.measure(2, 2)
    return qc



def bob_measurement_distribution(alpha: complex, beta: complex, shots: int = DEFAULT_SHOTS) -> Dict[str, int]:
    qc = teleportation_circuit(alpha, beta, measure_alice=True)
    return run_circuit(qc, shots=shots)


def bob_marginal_counts(counts: Dict[str, int]) -> Dict[str, int]:
    bob_counts = {"0": 0, "1": 0}
    for bitstring, count in counts.items():
        compact = bitstring.replace(" ", "")
        bob_bit = compact[0]
        bob_counts[bob_bit] += count
    return bob_counts


def q21_teleportation_result() -> TeleportationResult:
    alpha = 2 / np.sqrt(5)
    beta = 1 / np.sqrt(5)
    circuit = teleportation_circuit(alpha, beta, measure_alice=True)
    full_counts = run_circuit(circuit, shots=DEFAULT_SHOTS)
    return TeleportationResult(
        title="Q2.1 Teleportation of (2|0> + |1>) / sqrt(5)",
        input_state_latex=r"\left|q_0\right\rangle = \frac{1}{\sqrt{5}}\left(2\left|0\right\rangle + \left|1\right\rangle\right)",
        circuit=circuit,
        full_counts=full_counts,
        bob_counts=bob_marginal_counts(full_counts),
        expected_bob_probabilities={"0": 4 / 5, "1": 1 / 5},
    )


def q23_teleportation_zero_result() -> TeleportationResult:
    alpha = 1.0
    beta = 0.0
    circuit = teleportation_circuit(alpha, beta, measure_alice=True)
    full_counts = run_circuit(circuit, shots=DEFAULT_SHOTS)
    return TeleportationResult(
        title="Q2.3 Teleportation statistics for input |0>",
        input_state_latex=r"\left|q_0\right\rangle = \left|0\right\rangle",
        circuit=circuit,
        full_counts=full_counts,
        bob_counts=bob_marginal_counts(full_counts),
        expected_bob_probabilities={"0": 1.0, "1": 0.0},
    )



def site_to_qubits(site: int) -> Tuple[int, int]:
    if site == 1:
        return 0, 1
    if site == 2:
        return 2, 3
    raise ValueError("Site must be 1 or 2.")



def interaction_term(qc: QuantumCircuit, q_up: int, q_dn: int, U: float, dt: float) -> None:
    # exp(-i * dt * U/4 * Z Z), up to global and single-qubit phases accounted below
    angle = U * dt / 2.0
    qc.rz(-U * dt / 2.0, q_up)
    qc.rz(-U * dt / 2.0, q_dn)
    qc.rzz(angle, q_up, q_dn)



def hopping_term(qc: QuantumCircuit, q_left: int, q_right: int, J: float, dt: float) -> None:
    # H_hop = (J/2)(XX + YY), so U = exp(-i dt H_hop) = RXX(J dt) RYY(J dt)
    theta = J * dt
    qc.rxx(theta, q_left, q_right)
    qc.ryy(theta, q_left, q_right)



def one_trotter_step_hubbard(U: float, J: float, dt: float, measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(4, 4 if measure else 0)

    # Interaction terms on each site: (q0,q1) and (q2,q3)
    qc.barrier(label="Interaction")
    interaction_term(qc, 0, 1, U, dt)
    interaction_term(qc, 2, 3, U, dt)

    # Hopping terms for spin-up (q0 <-> q2) and spin-down (q1 <-> q3).
    # For this compact educational circuit we expose the required Z-string explicitly.
    qc.barrier(label="Hopping")
    qc.cz(1, 2)
    hopping_term(qc, 0, 2, J, dt)
    qc.cz(1, 2)

    qc.cz(0, 3)
    hopping_term(qc, 1, 3, J, dt)
    qc.cz(0, 3)

    if measure:
        qc.measure(range(4), range(4))
    return qc


def symbolic_one_trotter_step_hubbard() -> QuantumCircuit:
    U = Parameter("U")
    J = Parameter("J")
    dt = Parameter("Δt")
    qc = QuantumCircuit(4)

    qc.barrier(label="Interaction")
    qc.rz(-U * dt / 2.0, 0)
    qc.rz(-U * dt / 2.0, 1)
    qc.rzz(U * dt / 2.0, 0, 1)

    qc.rz(-U * dt / 2.0, 2)
    qc.rz(-U * dt / 2.0, 3)
    qc.rzz(U * dt / 2.0, 2, 3)

    qc.barrier(label="Hopping")
    qc.append(Gate("exp(+i JΔt/2 X₀Z₁X₂)", 3, []), [0, 1, 2])
    qc.append(Gate("exp(+i JΔt/2 Y₀Z₁Y₂)", 3, []), [0, 1, 2])
    qc.append(Gate("exp(+i JΔt/2 X₁Z₂X₃)", 3, []), [1, 2, 3])
    qc.append(Gate("exp(+i JΔt/2 Y₁Z₂Y₃)", 3, []), [1, 2, 3])

    return qc



def hubbard_time_evolution(
    initial_bitstring: str,
    U: float,
    J: float,
    tau: float,
    n_steps: int,
) -> QuantumCircuit:
    if len(initial_bitstring) != 4 or any(b not in "01" for b in initial_bitstring):
        raise ValueError("Initial bitstring must be a 4-character binary string.")

    dt = tau / max(n_steps, 1)
    qc = QuantumCircuit(4)
    for i, bit in enumerate(reversed(initial_bitstring)):
        if bit == "1":
            qc.x(i)

    for _ in range(n_steps):
        qc.compose(one_trotter_step_hubbard(U, J, dt, measure=False), inplace=True)
    return qc



def basis_probability(statevector: Statevector, bitstring: str) -> float:
    index = int(bitstring, 2)
    return float(abs(statevector.data[index]) ** 2)



def sweep_hubbard_probability(
    initial_bitstring: str,
    target_bitstring: str,
    U: float,
    J: float,
    taus: Iterable[float],
    n_steps: int,
) -> List[float]:
    probs: List[float] = []
    for tau in taus:
        qc = hubbard_time_evolution(initial_bitstring, U, J, tau, n_steps)
        sv = statevector_after(qc)
        probs.append(basis_probability(sv, target_bitstring))
    return probs


def q3_hubbard_report() -> HubbardReport:
    noninteracting_taus = np.linspace(0, np.pi, 80)
    strong_taus = np.linspace(0, 2.0, 80)

    return HubbardReport(
        trotter_circuit=symbolic_one_trotter_step_hubbard(),
        noninteracting_taus=[float(tau) for tau in noninteracting_taus],
        noninteracting_site2_probs=sweep_hubbard_probability(
            "1000", "0010", U=0.0, J=1.0, taus=noninteracting_taus, n_steps=40
        ),
        strong_taus=[float(tau) for tau in strong_taus],
        strong_remain_probs=sweep_hubbard_probability(
            "1100", "1100", U=10.0, J=1.0, taus=strong_taus, n_steps=80
        ),
        strong_transition_probs=sweep_hubbard_probability(
            "1100", "0011", U=10.0, J=1.0, taus=strong_taus, n_steps=80
        ),
    )
