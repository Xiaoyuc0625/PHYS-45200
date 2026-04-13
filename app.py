from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from qiskit.visualization import plot_histogram

from backend import (
    DEFAULT_SHOTS,
    bell_param_circuit,
    bob_marginal_counts,
    ghz10_circuit,
    measurement_counts,
    q14_protocol_steps,
    q21_teleportation_result,
    q22_long_distance_cnot_result,
    q23_teleportation_zero_result,
    q3_hubbard_report,
    run_circuit,
)

st.set_page_config(page_title="P452 Quantum Simulator", layout="wide")
st.title("P452 Project 1 — Quantum Simulator")
st.caption("Teleportation and 2-site Fermi–Hubbard demos on a 10-qubit-capable Aer backend")


def render_circuit_matplotlib(qc, scale: float = 0.8):
    fig = qc.draw(output="mpl", scale=scale, idle_wires=False)
    st.pyplot(fig, clear_figure=True)


def render_q14_step(step, shots: int):
    st.markdown(f"### Step {step.number}. {step.title}")
    st.markdown(step.description)

    circuit_col, state_col, measurement_col = st.columns([1.35, 1, 1])
    with circuit_col:
        st.markdown("**Circuit diagram**")
        render_circuit_matplotlib(step.display_circuit, scale=0.42)
    with state_col:
        st.markdown("**State vector representation**")
        st.latex(step.statevector_latex)
    with measurement_col:
        st.markdown("**Measurement**")
        if st.button(f"Take measurement for Step {step.number}", key=f"q14_measure_{step.number}"):
            sampled_counts = measurement_counts(step.state_circuit, shots=shots)
            st.markdown(f"**New measurement after {shots} shots**")
            fig = plot_histogram(sampled_counts)
            st.pyplot(fig, clear_figure=True)
            st.table(
                [
                    {"Outcome": f"|{bitstring}⟩", "Counts": count}
                    for bitstring, count in sampled_counts.items()
                ]
            )


def render_teleportation_result(result, shots: int, show_circuit: bool = True):
    st.markdown(f"### {result.title}")
    st.latex(result.input_state_latex)

    if show_circuit:
        st.markdown("**Circuit diagram**")
        render_circuit_matplotlib(result.circuit, scale=0.78)

    st.markdown("**Measurement**")
    if st.button(f"Take measurement for {result.title}", key=f"measure_{result.title}"):
        full_counts = run_circuit(result.circuit, shots=shots)
        bob_counts = bob_marginal_counts(full_counts)

        count_col, bob_col = st.columns([1, 1])
        with count_col:
            st.markdown(f"**Full measurement data after {shots} shots**")
            fig = plot_histogram(full_counts)
            st.pyplot(fig, clear_figure=True)
            st.write(full_counts)
        with bob_col:
            st.markdown("**Bob's marginal result**")
            fig = plot_histogram(bob_counts)
            st.pyplot(fig, clear_figure=True)
            st.table(
                [
                    {
                        "Bob outcome": f"|{bit}⟩",
                        "Counts": count,
                        "Measured probability": f"{count / shots:.4f}",
                        "Expected probability": f"{result.expected_bob_probabilities[bit]:.4f}",
                    }
                    for bit, count in bob_counts.items()
                ]
            )


def render_hubbard_page():
    report = q3_hubbard_report()

    st.subheader("Q3 Fermi-Hubbard Model")
    st.markdown("Two lattice sites are encoded with four qubits: one spin-up and one spin-down mode per site.")
    st.table(
        [
            {"Site": "1", "Spin-up qubit": "q0", "Spin-down qubit": "q1", "Occupation bits": "|n1_up n1_down>"},
            {"Site": "2", "Spin-up qubit": "q2", "Spin-down qubit": "q3", "Occupation bits": "|n2_up n2_down>"},
        ]
    )
    st.latex(
        r"H=-J\sum_{\sigma}\left(c_{1\sigma}^{\dagger}c_{2\sigma}+c_{2\sigma}^{\dagger}c_{1\sigma}\right)"
        r"+U\sum_i n_{i\uparrow}n_{i\downarrow}"
    )

    st.markdown("### Q3.1 One Trotter step")
    st.markdown(
        "The circuit below implements one first-order Trotter step. "
        "The interaction barrier contains the on-site U terms. "
        "The hopping barrier includes the Jordan-Wigner anti-commutation strings in the operator labels."
    )
    render_circuit_matplotlib(report.trotter_circuit, scale=0.9)

    st.latex(
        r"\text{single-qubit interaction term: }\ "
        r"e^{+i\frac{U\Delta t}{4}Z_i}\ \text{(Site i)}"
    )

    st.latex(
        r"\text{ two-qubitinteraction term: }\ "
        r"e^{-i\frac{U\Delta t}{4}Z_0Z_1}\ \text{(Site 1)}"
        r"\quad "
        r"e^{-i\frac{U\Delta t}{4}Z_2Z_3}\ \text{(Site 2)}"
    )

    st.latex(
        r"\text{spin-up hopping: }\ "
        r"e^{+i\frac{J\Delta t}{2}X_0Z_1X_2}"
        r"e^{+i\frac{J\Delta t}{2}Y_0Z_1Y_2}"
    )
    st.latex(
        r"\text{spin-down hopping: }\ "
        r"e^{+i\frac{J\Delta t}{2}X_1Z_2X_3}"
        r"e^{+i\frac{J\Delta t}{2}Y_1Z_2Y_3}"
    )

    st.markdown("### Q3.2 Non-interacting dynamics")
    st.markdown(
        "Set `U = 0`, `J = 1`, and start from `|1000>`. "
        "With no on-site repulsion, the spin-up particle coherently tunnels from site 1 to site 2."
    )
    q32_df = pd.DataFrame(
        {
            "t": report.noninteracting_taus,
            "P(|0010>)": report.noninteracting_site2_probs,
        }
    )
    st.line_chart(q32_df, x="t", y="P(|0010>)")
    best_idx = int(np.argmax(report.noninteracting_site2_probs))
    st.table(
        [
            {"Quantity": "Initial state", "Value": "|1000>"},
            {"Quantity": "Target state", "Value": "|0010>"},
            {"Quantity": "Maximum plotted probability", "Value": f"{report.noninteracting_site2_probs[best_idx]:.4f}"},
            {"Quantity": "Time at maximum", "Value": f"{report.noninteracting_taus[best_idx]:.4f}"},
        ]
    )
    st.markdown(
        "This answers Q3.2: the probability rises to nearly one, showing full coherent transfer in the non-interacting limit."
    )

    st.markdown("### Q3.3 Strong interaction and doublon suppression")
    st.markdown(
        "Set `U = 10`, `J = 1`, and start from the doublon state `|1100>`. "
        "Compare remaining at site 1 with transitioning to the site-2 doublon state `|0011>`."
    )
    q33_df = pd.DataFrame(
        {
            "t": report.strong_taus,
            "P(|1100>) remain": report.strong_remain_probs,
            "P(|0011>) transition": report.strong_transition_probs,
        }
    )
    st.line_chart(q33_df, x="t", y=["P(|1100>) remain", "P(|0011>) transition"])
    transition_max_idx = int(np.argmax(report.strong_transition_probs))
    st.table(
        [
            {"Quantity": "Initial doublon", "Value": "|1100>"},
            {"Quantity": "Site-2 doublon", "Value": "|0011>"},
            {"Quantity": "Largest plotted transition probability", "Value": f"{report.strong_transition_probs[transition_max_idx]:.4f}"},
            {"Quantity": "Remain probability at same time", "Value": f"{report.strong_remain_probs[transition_max_idx]:.4f}"},
        ]
    )
    st.markdown(
        "This answers Q3.3: large `U` makes doublon motion energetically costly, so tunneling is suppressed. "
        "The system mostly remains in `|1100>`, which is the two-site signature of Mott-insulating behavior."
    )


preset = st.sidebar.selectbox(
    "Preset circuit",
    [
        "Q1.2 Parameter Control Loop",
        "Q1.3 10-qubit GHZ",
        "Q1.4 Unitarity and State Recovery",
        "Q2.1 and Q2.3 Teleportation",
        "Q2.2 Long-Distance CNOT",
        "Q3 Fermi-Hubbard Model",
    ],
)
shots = st.sidebar.number_input("Shots", min_value=128, max_value=8192, value=DEFAULT_SHOTS, step=128)

left, right = st.columns([1, 1])

if preset == "Q1.2 Parameter Control Loop":
    theta = st.sidebar.slider("Rotation angle θ", 0.0, float(np.pi), float(np.pi / 2), 0.01)
    qc = bell_param_circuit(theta)
    with left:
        st.subheader("Q1.2 Parameter Control Loop")
        st.markdown("**Circuit diagram**")
        render_circuit_matplotlib(qc, scale=0.9)
    with right:
        if st.button("Run parameter circuit"):
            counts = run_circuit(qc, shots=int(shots))
            st.subheader("Measurement histogram")
            fig = plot_histogram(counts)
            st.pyplot(fig, clear_figure=True)
            st.write(counts)

elif preset == "Q3 Fermi-Hubbard Model":
    render_hubbard_page()

elif preset == "Q1.3 10-qubit GHZ":
    qc = ghz10_circuit(measure=True)
    with left:
        st.subheader("Q1.3 10-qubit GHZ")
        st.markdown("**Circuit diagram**")
        render_circuit_matplotlib(qc, scale=0.55)
    with right:
        if st.button("Run GHZ"):
            counts = run_circuit(qc, shots=int(shots))
            st.subheader("Measurement histogram")
            fig = plot_histogram(counts)
            st.pyplot(fig, clear_figure=True)
            st.write(counts)

elif preset == "Q1.4 Unitarity and State Recovery":
    steps = q14_protocol_steps()

    st.subheader("Q1.4 Unitarity and State Recovery")
    st.markdown(
        "This protocol prepares the 10-qubit state "
        r"$2^{-1/2}(|201\rangle + |425\rangle)$, applies a reversible chain of nine "
        "two-qubit gates, and then applies the inverse chain to recover the original state."
    )
    st.markdown(
        r"Here $|201\rangle = |0011001001\rangle$ and "
        r"$|425\rangle = |0110101001\rangle$ in the displayed computational-basis labels."
    )

    for index, step in enumerate(steps):
        if index:
            st.divider()
        render_q14_step(step, shots=int(shots))

    st.markdown("### Analysis")
    st.markdown(
        "Each CNOT gate is unitary, so the nine-gate chain is unitary. "
        "Applying the same CNOTs in reverse order implements the inverse operation, "
        "which returns the non-zero amplitudes and measurement probabilities to the initial state."
    )

elif preset == "Q2.1 and Q2.3 Teleportation":
    q21 = q21_teleportation_result()
    q23 = q23_teleportation_zero_result()

    st.subheader("Phase 2: Teleportation")
    st.markdown(
        "The Bell State Preparation and Bell Measurement stages are labeled directly in the circuit barriers."
    )

    render_teleportation_result(q21, shots=int(shots), show_circuit=True)

    st.divider()
    render_teleportation_result(q23, shots=int(shots), show_circuit=True)
    st.markdown(
        "For Q2.3, Bob should find `|0⟩` with probability 1. "
        "In the ideal simulator, any nonzero `|1⟩` count would come from implementation error rather than hardware noise."
    )

elif preset == "Q2.2 Long-Distance CNOT":
    result = q22_long_distance_cnot_result()

    st.subheader("Q2.2 Long-Distance CNOT in a Linear Chain")
    st.markdown(
        "Goal: implement `CNOT(q0 -> q4)` when the hardware only permits adjacent gates on the chain `0-1-2-3-4`."
    )

    st.markdown("### Target operation")
    render_circuit_matplotlib(result.target_circuit, scale=0.85)

    st.markdown("### Neighbor-only routed circuit")
    st.markdown(
        "Route `q4` next to `q0` with three SWAPs, apply `CNOT(q0 -> q1)`, then undo the three SWAPs."
    )
    render_circuit_matplotlib(result.routed_swap_circuit, scale=0.85)

    st.markdown("### Fully decomposed into CNOT gates")
    st.markdown("Each SWAP is decomposed as three CNOT gates.")
    render_circuit_matplotlib(result.decomposed_circuit, scale=0.72)

    st.latex(r"6\ \mathrm{SWAPs}\times 3\ \mathrm{CNOTs/SWAP} + 1\ \mathrm{CNOT} = 19\ \mathrm{CNOTs}")
    st.table(
        [
            {"Quantity": "SWAP gates used for routing", "Value": result.swap_count},
            {"Quantity": "Total CNOT gates after SWAP decomposition", "Value": result.cnot_count},
        ]
    )
