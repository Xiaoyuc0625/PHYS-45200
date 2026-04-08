"""Notebook-style script for answering the written checkpoint questions.

Run sections sequentially in Jupyter or Colab.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from backend import (
    bell_param_circuit,
    bob_measurement_distribution,
    ghz10_circuit,
    hubbard_time_evolution,
    long_distance_cnot_q0_q4_linear,
    nonzero_amplitudes,
    one_trotter_step_hubbard,
    prepare_superposition_201_425,
    entangling_chain,
    entangling_chain_inverse,
    statevector_after,
    sweep_hubbard_probability,
    teleportation_circuit,
)

# Q1.2 Parameter control loop
qc_param = bell_param_circuit(np.pi)
counts_param = bob_measurement_distribution(0.0, 1.0)  # Equivalent theta=pi on q0 only for demo consistency
print("Q1.2 counts:", counts_param)
plot_histogram(counts_param)
plt.show()

# Q1.3 10-qubit GHZ visualization
qc_ghz = ghz10_circuit(measure=False)
fig = qc_ghz.draw(output="mpl", scale=0.55, idle_wires=False)
plt.show()

# Q1.4 Unitarity and state recovery
prep = prepare_superposition_201_425()
chain = entangling_chain()
recover = entangling_chain_inverse()
sv0 = statevector_after(prep)
sv1 = statevector_after(prep.compose(chain))
sv2 = statevector_after(prep.compose(chain).compose(recover))
print("Initial non-zero amplitudes:", nonzero_amplitudes(sv0))
print("After chain non-zero amplitudes:", nonzero_amplitudes(sv1))
print("Recovered non-zero amplitudes:", nonzero_amplitudes(sv2))

# Q2.1 Teleportation with |q0> = (2|0> + |1>) / sqrt(5)
alpha = 2 / np.sqrt(5)
beta = 1 / np.sqrt(5)
qc_tel = teleportation_circuit(alpha, beta)
fig = qc_tel.draw(output="mpl", scale=0.8, idle_wires=False)
plt.show()

# Q2.2 Long-distance CNOT circuit
qc_long = long_distance_cnot_q0_q4_linear()
fig = qc_long.draw(output="mpl", scale=0.8, idle_wires=False)
plt.show()
print("A SWAP decomposes into 3 CNOTs. Total CNOT count = 6 swaps * 3 + 1 = 19.")

# Q2.3 Teleportation fidelity for input |0>
counts_zero = bob_measurement_distribution(1.0, 0.0, shots=1024)
print("Teleporting |0> counts:", counts_zero)
plot_histogram(counts_zero)
plt.show()

# Q3.1 One Trotter step
qc_step = one_trotter_step_hubbard(U=2.0, J=1.0, dt=0.25, measure=False)
fig = qc_step.draw(output="mpl", scale=0.9, idle_wires=False)
plt.show()

# Q3.2 Non-interacting dynamics
T = np.linspace(0, np.pi, 120)
p_site2 = sweep_hubbard_probability("1000", "0010", U=0.0, J=1.0, taus=T, n_steps=40)
plt.figure(figsize=(7, 4))
plt.plot(T, p_site2)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$P(|0010\rangle)$")
plt.title("Non-interacting dynamics: transfer to site 2")
plt.grid(True, alpha=0.3)
plt.show()

# Q3.3 Strong interactions and doublon dynamics
T = np.linspace(0, 2.0, 120)
p_1100 = sweep_hubbard_probability("1100", "1100", U=10.0, J=1.0, taus=T, n_steps=80)
p_0011 = sweep_hubbard_probability("1100", "0011", U=10.0, J=1.0, taus=T, n_steps=80)
plt.figure(figsize=(7, 4))
plt.plot(T, p_1100, label="Remain in |1100>")
plt.plot(T, p_0011, label="Transition to |0011>")
plt.xlabel(r"$\tau$")
plt.ylabel("Probability")
plt.title("Strongly interacting doublon dynamics")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
