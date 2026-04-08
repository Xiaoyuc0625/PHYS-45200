# P452 Project 1 — Simulation of a Universal Quantum Computer

This package contains a full-stack starter solution for the assignment:

- `backend.py`: 10-qubit-capable Aer backend and reusable circuit utilities.
- `app.py`: Streamlit frontend with the required preset circuits.
- `analysis_notebook.py`: notebook-style script answering the written checkpoint questions.
- `requirements.txt`: dependencies for local execution or Streamlit Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files matched to assignment tasks

### Phase 1
- **Backend**: `run_circuit(circuit)` in `backend.py`
- **Frontend preset selection / slider / circuit view**: `app.py`
- **Q1.2 parameter loop**: `bell_param_circuit(theta)`
- **Q1.3 10-qubit GHZ visualization**: `ghz10_circuit()`
- **Q1.4 unitarity and state recovery**: `prepare_superposition_201_425()`, `entangling_chain()`, `entangling_chain_inverse()`

### Phase 2
- **Teleportation**: `teleportation_circuit(alpha, beta)`
- **Long-distance CNOT routing**: `long_distance_cnot_q0_q4_linear()`

### Phase 3
- **One Trotter step**: `one_trotter_step_hubbard(U, J, dt)`
- **Dynamics plots**: `sweep_hubbard_probability(...)`

## Deployment

1. Push these files to a public GitHub repository.
2. On Streamlit Cloud, deploy with `app.py` as the entry point.
3. Add `requirements.txt` so the cloud build installs Qiskit and Streamlit.

## Notes

- For the parameter-control screenshot, set `theta = pi`. The ideal output is all counts in `11`.
- For the GHZ screenshot, use the **10-qubit GHZ** preset and capture the circuit diagram.
- For the written report, use the figures generated in `analysis_notebook.py`.
