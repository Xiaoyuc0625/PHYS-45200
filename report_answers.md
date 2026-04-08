# P452 Project 1 — Written Answers

## Q1.1 Repository and Cloud Deployment
After you push this project to GitHub and deploy `app.py` on Streamlit Cloud, insert the two URLs here.

- GitHub repository: `...`
- Live Streamlit app: `...`

## Q1.2 The Parameter Control Loop
The circuit applies `Ry(theta)` to `q0` and then `CNOT(q0 -> q1)`. When `theta = pi`, the state evolves as

`|00> -> |10> -> |11>`.

Therefore, an ideal histogram with nearly 100% of shots in `11` shows that the slider value really reached the backend. If the slider failed to pass `theta = pi`, the output would not collapse to `|11>` with unit probability.

## Q1.3 10-Qubit Visualization
Use `ghz10_circuit()` and the Streamlit **10-qubit GHZ** preset. The circuit prepares

`(|0000000000> + |1111111111>) / sqrt(2)`.

The expected measurement histogram has dominant peaks at `0000000000` and `1111111111`, each near 50%.

## Q1.4 Unitarity and State Recovery
The initial state is prepared as

`( |201> + |425> ) / sqrt(2)`.

A chain of 9 CNOT gates maps the basis components reversibly. Because each gate is unitary, the full circuit is unitary, and applying the inverse chain in reverse order recovers the initial state exactly. Equality of the recovered statevector with the initial one confirms reversibility and hence unitarity.

## Q2.1 Teleportation of (2|0> + |1>) / sqrt(5)
The teleportation circuit initializes Alice's qubit in

`|psi> = (2|0> + |1>) / sqrt(5)`.

The **Bell State Preparation** stage entangles qubits 1 and 2. The **Bell Measurement** stage measures Alice's two qubits, and the conditional `X` and `Z` corrections reconstruct the input state on Bob's qubit.

## Q2.2 Long-Distance CNOT in a Linear Chain
To perform `CNOT(q0 -> q4)` on a chain `0-1-2-3-4`, route the target next to `q0` using 3 SWAPs, apply one nearest-neighbor CNOT, then undo the routing using 3 more SWAPs.

Since each SWAP decomposes into 3 CNOTs, the total number of CNOTs is

`6 * 3 + 1 = 19`.

## Q2.3 Teleportation Statistics for Input |0>
For input `|0>`, Bob should measure `|0>` with probability 1 in the ideal simulator. Any small deviation from 100% in a finite-shot histogram comes only from sampling noise because the circuit is simulated rather than run on noisy hardware.

## Q3.1 One Trotter Step
In the circuit produced by `one_trotter_step_hubbard(U, J, dt)`:

- the **interaction term** is implemented by the `RZZ`-based blocks on `(q0, q1)` and `(q2, q3)`;
- the **hopping term** is implemented by `RXX` and `RYY` rotations between `(q0, q2)` and `(q1, q3)`;
- the required Jordan–Wigner **Z-string** is shown explicitly using `CZ` gates surrounding the long-range hopping block.

## Q3.2 Non-Interacting Dynamics (U = 0)
With `U = 0`, `J = 1`, and initial state `|1000>`, the dynamics reduce to coherent hopping of a single spin-up fermion between sites 1 and 2. The probability of state `|0010>` oscillates sinusoidally and reaches 1 at the first full transfer time.

For the ideal two-level Hamiltonian with matrix element `J`, the transition probability is `sin^2(J tau)`, so complete transfer first occurs at

`tau = pi / 2`.

## Q3.3 Strong Interactions and Mott Physics
With `U = 10` and initial state `|1100>`, the large interaction energy strongly penalizes doublon motion. The probability of remaining in `|1100>` stays high, while the transfer to `|0011>` is suppressed and much slower than in the weakly interacting case.

This is the characteristic Mott-insulator picture: strong on-site repulsion blocks charge motion, reducing tunneling and localizing the particles.
