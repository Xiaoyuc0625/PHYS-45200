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
    run_circuit,
    sweep_hubbard_probability,
    symbolic_one_trotter_step_hubbard,
)

st.set_page_config(page_title="P452 Quantum Simulator", layout="wide")
st.title("P452 Project 1 — Quantum Simulator")

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
    render_circuit_matplotlib(symbolic_one_trotter_step_hubbard(), scale=0.9)

    st.latex(
        r"\text{single-qubit interaction term: }\ "
        r"e^{+i\frac{U\Delta t}{4}Z_i}\ \text{(Qubit i)}"
    )

    st.latex(
        r"\text{ two-qubit interaction term: }\ "
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
    q32_u, q32_j = st.columns(2)
    with q32_u:
        q32_U = st.number_input("Q3.2 U", value=0.0, step=0.5, key="q32_u")
    with q32_j:
        q32_J = st.number_input("Q3.2 J", value=1.0, step=0.1, key="q32_j")
    st.markdown("**One Trotter-step block used in the sweep**")
    render_circuit_matplotlib(symbolic_one_trotter_step_hubbard(), scale=0.9)
    if st.button("Run Q3.2 circuit and plot", key="run_q32"):
        q32_t = np.linspace(0, np.pi, 80)
        q32_probs = sweep_hubbard_probability("1000", "0010", U=q32_U, J=q32_J, taus=q32_t, n_steps=40)
        q32_df = pd.DataFrame({"t": q32_t, "P(|0010>)": q32_probs})
        st.line_chart(q32_df, x="t", y="P(|0010>)")
        best_idx = int(np.argmax(q32_probs))
        st.table(
            [
                {"Quantity": "Initial state", "Value": "|1000>"},
                {"Quantity": "Target state", "Value": "|0010>"},
                {"Quantity": "Maximum plotted probability", "Value": f"{q32_probs[best_idx]:.4f}"},
                {"Quantity": "Time at maximum", "Value": f"{q32_t[best_idx]:.4f}"},
            ]
        )

    st.markdown("### Q3.3 Strong interaction and doublon suppression")
    q33_u, q33_j = st.columns(2)
    with q33_u:
        q33_U = st.number_input("Q3.3 U", value=10.0, step=0.5, key="q33_u")
    with q33_j:
        q33_J = st.number_input("Q3.3 J", value=1.0, step=0.1, key="q33_j")
    st.markdown("**One Trotter-step block used in the sweep**")
    render_circuit_matplotlib(symbolic_one_trotter_step_hubbard(), scale=0.9)
    if st.button("Run Q3.3 circuit and plot", key="run_q33"):
        q33_t = np.linspace(0, 2.0, 80)
        q33_remain = sweep_hubbard_probability("1100", "1100", U=q33_U, J=q33_J, taus=q33_t, n_steps=80)
        q33_transition = sweep_hubbard_probability("1100", "0011", U=q33_U, J=q33_J, taus=q33_t, n_steps=80)
        q33_df = pd.DataFrame(
            {
                "t": q33_t,
                "P(|1100>) remain": q33_remain,
                "P(|0011>) transition": q33_transition,
            }
        )
        st.line_chart(q33_df, x="t", y=["P(|1100>) remain", "P(|0011>) transition"])
        transition_max_idx = int(np.argmax(q33_transition))
        st.table(
            [
                {"Quantity": "Initial doublon", "Value": "|1100>"},
                {"Quantity": "Site-2 doublon", "Value": "|0011>"},
                {"Quantity": "Largest plotted transition probability", "Value": f"{q33_transition[transition_max_idx]:.4f}"},
                {"Quantity": "Remain probability at same time", "Value": f"{q33_remain[transition_max_idx]:.4f}"},
            ]
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

elif preset == "Q2.1 and Q2.3 Teleportation":
    q21 = q21_teleportation_result()
    q23 = q23_teleportation_zero_result()

    st.subheader("Phase 2: Teleportation")

    render_teleportation_result(q21, shots=int(shots), show_circuit=True)

    st.divider()
    render_teleportation_result(q23, shots=int(shots), show_circuit=True)

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
