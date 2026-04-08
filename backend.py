from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

MAX_QUBITS = 10
DEFAULT_SHOTS = 1024
_SIM = AerSimulator()
_STATEVEC_SIM = AerSimulator(method="statevector")


class BackendError(ValueError):
    """Raised when an invalid circuit is submitted to the backend."""



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
    sv = np.zeros(2**10, dtype=complex)
    sv[201] = 1 / np.sqrt(2)
    sv[425] = 1 / np.sqrt(2)
    qc = QuantumCircuit(10)
    qc.initialize(sv, range(10))
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



def teleportation_circuit(alpha: complex, beta: complex, measure_alice: bool = True) -> QuantumCircuit:
    if not np.isclose(abs(alpha) ** 2 + abs(beta) ** 2, 1.0):
        raise ValueError("Input amplitudes must be normalized.")

    qc = QuantumCircuit(3, 3)

    qc.initialize([alpha, beta], 0)
    qc.barrier(label="Input state")

    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell State Preparation")

    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Bell Measurement")

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
    interaction_term(qc, 0, 1, U, dt)
    interaction_term(qc, 2, 3, U, dt)
    qc.barrier(label="Interaction")

    # Hopping terms for spin-up (q0 <-> q2) and spin-down (q1 <-> q3).
    # For this compact educational circuit we expose the required Z-string explicitly.
    qc.cz(1, 2)
    hopping_term(qc, 0, 2, J, dt)
    qc.cz(1, 2)

    qc.cz(0, 3)
    hopping_term(qc, 1, 3, J, dt)
    qc.cz(0, 3)
    qc.barrier(label="Hopping")

    if measure:
        qc.measure(range(4), range(4))
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
