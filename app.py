from __future__ import annotations

import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from qiskit.visualization import plot_histogram

from backend import (
    DEFAULT_SHOTS,
    bell_param_circuit,
    bob_measurement_distribution,
    ghz10_circuit,
    one_trotter_step_hubbard,
    run_circuit,
)

st.set_page_config(page_title="P452 Quantum Simulator", layout="wide")
st.title("P452 Project 1 — Quantum Simulator")
st.caption("Teleportation and 2-site Fermi–Hubbard demos on a 10-qubit-capable Aer backend")


def render_circuit_matplotlib(qc, scale: float = 0.8):
    fig = qc.draw(output="mpl", scale=scale, idle_wires=False)
    st.pyplot(fig, clear_figure=True)


preset = st.sidebar.selectbox(
    "Preset circuit",
    [
        "Parameter control loop",
        "Teleportation",
        "Hubbard (one Trotter step)",
        "10-qubit GHZ",
    ],
)
shots = st.sidebar.number_input("Shots", min_value=128, max_value=8192, value=DEFAULT_SHOTS, step=128)

left, right = st.columns([1, 1])

if preset == "Parameter control loop":
    theta = st.sidebar.slider("Rotation angle θ", 0.0, float(np.pi), float(np.pi / 2), 0.01)
    qc = bell_param_circuit(theta)
    with left:
        st.subheader("Circuit diagram")
        render_circuit_matplotlib(qc, scale=0.9)
    with right:
        if st.button("Run parameter circuit"):
            counts = run_circuit(qc, shots=int(shots))
            st.subheader("Measurement histogram")
            fig = plot_histogram(counts)
            st.pyplot(fig, clear_figure=True)
            st.write(counts)

elif preset == "Teleportation":
    theta = st.sidebar.slider("State-preparation angle θ for |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩", 0.0, float(np.pi), 1.107, 0.001)
    alpha = np.cos(theta / 2.0)
    beta = np.sin(theta / 2.0)
    from backend import teleportation_circuit

    qc = teleportation_circuit(alpha, beta, measure_alice=True)
    with left:
        st.subheader("Teleportation circuit")
        render_circuit_matplotlib(qc, scale=0.75)
    with right:
        if st.button("Run teleportation"):
            counts = bob_measurement_distribution(alpha, beta, shots=int(shots))
            st.subheader("Measurement histogram")
            fig = plot_histogram(counts)
            st.pyplot(fig, clear_figure=True)
            st.write(counts)

elif preset == "Hubbard (one Trotter step)":
    U = st.sidebar.slider("Interaction U/J", 0.0, 10.0, 2.0, 0.1)
    J = 1.0
    dt = st.sidebar.slider("Trotter time step Δτ", 0.05, 1.0, 0.25, 0.05)
    qc = one_trotter_step_hubbard(U=U, J=J, dt=dt, measure=True)
    with left:
        st.subheader("One Trotter step circuit")
        render_circuit_matplotlib(qc, scale=0.9)
        st.markdown("**Labels:** interaction blocks implement on-site terms; hopping blocks implement spin-conserving tunneling.")
    with right:
        if st.button("Run one Trotter step"):
            counts = run_circuit(qc, shots=int(shots))
            st.subheader("Measurement histogram")
            fig = plot_histogram(counts)
            st.pyplot(fig, clear_figure=True)
            st.write(counts)

elif preset == "10-qubit GHZ":
    qc = ghz10_circuit(measure=True)
    with left:
        st.subheader("10-qubit GHZ circuit")
        render_circuit_matplotlib(qc, scale=0.55)
    with right:
        if st.button("Run GHZ"):
            counts = run_circuit(qc, shots=int(shots))
            st.subheader("Measurement histogram")
            fig = plot_histogram(counts)
            st.pyplot(fig, clear_figure=True)
            st.write(counts)
