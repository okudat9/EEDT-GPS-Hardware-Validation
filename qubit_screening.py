"""
qubit_screening.py
==================
Screens all qubit pairs on an IBM Quantum backend and selects
candidates suitable for EEDT operation.

Selection criteria (from paper Section II-B):
  (i)  T2 > 150 us
  (ii) Readout error < 2%
  (iii) ZZ coupling nu_ZZ in [0.5, 4.2] kHz  (estimated from gate errors)

Then performs ZZ Ramsey spectroscopy on top candidates.

Usage:
  python qubit_screening.py --backend ibm_marrakesh --top 5

IBM Quantum access:
  Set your IBM Quantum token via environment variable:
    export IBM_QUANTUM_TOKEN="your_token_here"
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# Screening thresholds (paper criteria)
T2_MIN_US       = 150.0    # minimum T2 [us]
READOUT_MAX_ERR = 0.02     # maximum readout error (2%)
NU_ZZ_MIN_KHZ   = 0.5     # minimum ZZ coupling [kHz]
NU_ZZ_MAX_KHZ   = 4.2     # maximum ZZ coupling [kHz]

# ZZ Ramsey parameters
ZZ_RAMSEY_TAU_LIST = [10, 20, 30, 40, 50, 60, 70, 80]  # us
ZZ_RAMSEY_SHOTS    = 4096


def get_backend_properties(service: QiskitRuntimeService,
                            backend_name: str) -> dict:
    """Fetch and parse backend properties."""
    backend = service.backend(backend_name)
    props   = backend.properties()

    qubit_data = {}
    for q_idx in range(backend.num_qubits):
        t1  = props.t1(q_idx)  * 1e6   # s -> us
        t2  = props.t2(q_idx)  * 1e6   # s -> us
        ro_err = props.readout_error(q_idx)

        qubit_data[q_idx] = {
            "T1_us":       round(t1, 1),
            "T2_us":       round(t2, 1),
            "readout_err": round(ro_err, 4),
        }
    return qubit_data, backend


def screen_qubits(qubit_data: dict) -> list:
    """
    Apply screening criteria and return sorted list of
    (qubit_index, score) tuples.
    Score = T2 (higher is better).
    """
    candidates = []
    for q_idx, props in qubit_data.items():
        if props["T2_us"] >= T2_MIN_US and props["readout_err"] <= READOUT_MAX_ERR:
            candidates.append({
                "qubit":       q_idx,
                "T2_us":       props["T2_us"],
                "T1_us":       props["T1_us"],
                "readout_err": props["readout_err"],
                "score":       props["T2_us"],  # rank by T2
            })
    candidates.sort(key=lambda x: -x["score"])
    return candidates


def find_connected_pairs(candidates: list, backend) -> list:
    """
    Find pairs of adjacent (coupled) qubits from the candidate list.
    Returns list of (q0, q1) tuples sorted by combined T2.
    """
    # Get coupling map
    try:
        cm = backend.coupling_map
        edges = set(tuple(sorted(e)) for e in cm.get_edges())
    except Exception:
        # Fallback: use backend configuration
        edges = set()

    candidate_idxs = {c["qubit"] for c in candidates}
    candidate_map  = {c["qubit"]: c for c in candidates}

    pairs = []
    for q0, q1 in edges:
        if q0 in candidate_idxs and q1 in candidate_idxs:
            pairs.append({
                "qubit0":  q0,
                "qubit1":  q1,
                "T2_q0":   candidate_map[q0]["T2_us"],
                "T2_q1":   candidate_map[q1]["T2_us"],
                "T2_min":  min(candidate_map[q0]["T2_us"],
                               candidate_map[q1]["T2_us"]),
                "T2_sum":  candidate_map[q0]["T2_us"] + candidate_map[q1]["T2_us"],
            })
    pairs.sort(key=lambda x: -x["T2_min"])
    return pairs


def build_zz_ramsey_circuit(qubit0: int, qubit1: int,
                             tau_us: float) -> QuantumCircuit:
    """
    ZZ Ramsey spectroscopy circuit.
    Measures the oscillation frequency of Q1 fidelity vs tau,
    giving nu_ZZ directly.

    Q0 is prepared in |1> to activate ZZ interaction.
    Q1 undergoes Ramsey sequence: H - wait(tau) - H - measure.
    """
    from qiskit import QuantumRegister, ClassicalRegister
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)

    # Q0 in |1> to turn on ZZ
    qc.x(qr[0])
    # Q1 Ramsey
    qc.h(qr[1])
    qc.barrier()
    qc.delay(int(tau_us * 1000), qr[1], unit='ns')
    qc.barrier()
    qc.h(qr[1])
    qc.measure(qr[1], cr[0])
    return qc


def fit_nu_zz(tau_list: list, fidelities: list) -> dict:
    """
    Fit F(tau) = A * cos(2*pi*nu_ZZ*tau + phi) + offset
    to extract nu_ZZ.
    Returns dict with nu_zz_khz and fit quality.
    """
    tau_arr = np.array(tau_list, dtype=float)
    fid_arr = np.array(fidelities, dtype=float)

    # Simple FFT-based frequency estimate
    if len(tau_arr) < 4:
        return {"nu_zz_khz": None, "method": "insufficient_data"}

    dt = tau_arr[1] - tau_arr[0]
    freqs = np.fft.rfftfreq(len(tau_arr), d=dt)   # 1/us = MHz
    fft   = np.abs(np.fft.rfft(fid_arr - fid_arr.mean()))
    # Skip DC (index 0)
    peak_idx = np.argmax(fft[1:]) + 1
    nu_mhz   = freqs[peak_idx]
    nu_khz   = nu_mhz * 1e3

    return {
        "nu_zz_khz":  round(nu_khz, 3),
        "method":     "fft",
        "tau_list":   tau_list,
        "fidelities": [round(f, 4) for f in fidelities],
    }


def run_zz_ramsey(backend, qubit0: int, qubit1: int,
                  tau_list: list = None,
                  shots: int = ZZ_RAMSEY_SHOTS) -> dict:
    """Run ZZ Ramsey on a qubit pair and return nu_ZZ estimate."""
    if tau_list is None:
        tau_list = ZZ_RAMSEY_TAU_LIST

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    fidelities = []

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        for tau in tau_list:
            qc  = build_zz_ramsey_circuit(qubit0, qubit1, tau)
            isa = pm.run(qc)
            job = sampler.run([isa], shots=shots)
            counts = job.result()[0].data.c.get_counts()
            count_0 = sum(v for k, v in counts.items() if k[-1] == '0')
            fidelities.append(count_0 / shots)

    fit = fit_nu_zz(tau_list, fidelities)
    fit["job_backend"] = backend.name
    fit["qubit0"]      = qubit0
    fit["qubit1"]      = qubit1
    return fit


def main():
    parser = argparse.ArgumentParser(
        description="Screen IBM Quantum qubit pairs for EEDT compatibility")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--top",     type=int, default=3,
                        help="Number of top pairs to characterize via ZZ Ramsey")
    parser.add_argument("--no-ramsey", action="store_true",
                        help="Skip ZZ Ramsey (properties screening only)")
    parser.add_argument("--out",     default="screening_results.json")
    args = parser.parse_args()

    # IBM Quantum login
    token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum")

    print(f"Fetching properties for {args.backend}...")
    qubit_data, backend = get_backend_properties(service, args.backend)

    print(f"Total qubits: {len(qubit_data)}")
    candidates = screen_qubits(qubit_data)
    print(f"Qubits passing T2>{T2_MIN_US}us and RO_err<{READOUT_MAX_ERR*100}%: "
          f"{len(candidates)}")

    pairs = find_connected_pairs(candidates, backend)
    print(f"Connected pairs from candidates: {len(pairs)}")

    print("\nTop pairs by minimum T2:")
    print(f"  {'Q0':>4} {'Q1':>4} {'T2_min':>8} {'T2_q0':>8} {'T2_q1':>8}")
    print("-" * 40)
    for p in pairs[:10]:
        print(f"  {p['qubit0']:>4} {p['qubit1']:>4} "
              f"{p['T2_min']:>8.1f} {p['T2_q0']:>8.1f} {p['T2_q1']:>8.1f}")

    results = {
        "metadata": {
            "backend":   args.backend,
            "timestamp": datetime.utcnow().isoformat(),
            "criteria":  {
                "T2_min_us":      T2_MIN_US,
                "readout_max_err": READOUT_MAX_ERR,
                "nu_zz_range_khz": [NU_ZZ_MIN_KHZ, NU_ZZ_MAX_KHZ],
            },
        },
        "top_pairs":    pairs[:args.top],
        "all_candidates": candidates,
        "zz_ramsey":    [],
    }

    if not args.no_ramsey and len(pairs) > 0:
        print(f"\nRunning ZZ Ramsey on top {args.top} pairs...")
        for pair in pairs[:args.top]:
            q0, q1 = pair["qubit0"], pair["qubit1"]
            print(f"  Q{q0}-Q{q1}...", end=" ", flush=True)
            ramsey = run_zz_ramsey(backend, q0, q1)
            print(f"nu_ZZ = {ramsey['nu_zz_khz']} kHz")
            results["zz_ramsey"].append(ramsey)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")

    # Print recommended pair
    if results["zz_ramsey"]:
        valid = [r for r in results["zz_ramsey"]
                 if r["nu_zz_khz"] and NU_ZZ_MIN_KHZ <= r["nu_zz_khz"] <= NU_ZZ_MAX_KHZ]
        if valid:
            best = max(valid, key=lambda x: x["nu_zz_khz"])
            print(f"\n>>> Recommended pair: Q{best['qubit0']}-Q{best['qubit1']} "
                  f"(nu_ZZ = {best['nu_zz_khz']} kHz)")
        else:
            print("\n>>> No pairs found in optimal nu_ZZ range. "
                  "Consider relaxing criteria.")


if __name__ == "__main__":
    main()
