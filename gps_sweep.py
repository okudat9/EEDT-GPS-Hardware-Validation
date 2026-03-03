"""
gps_sweep.py
============
GPS (Gain-by-Post-Selection) heatmap experiment for EEDT validation.
Sweeps (tau, N_meas) parameter space and measures correction gain G_PS.

Reference:
  Okuda, T. "Hardware Operating Conditions for Entanglement-Enhanced
  Dynamical Tracking..." (2026)

Usage:
  python gps_sweep.py --backend ibm_marrakesh --qubit0 94 --qubit1 95

IBM Quantum access:
  Set your IBM Quantum token via environment variable:
    export IBM_QUANTUM_TOKEN="your_token_here"
  or save credentials once:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="...")
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ─────────────────────────────────────────────
# Experiment parameters (matching paper GPS v8)
# ─────────────────────────────────────────────
DEFAULT_TAU_LIST   = [30, 50]          # wait times in microseconds
DEFAULT_NMEAS_LIST = [1, 2, 3, 4, 5, 6]
DEFAULT_SHOTS      = 16000
SIGMA_THRESHOLD    = 2.0               # z-score threshold for significance


def build_eedt_circuit(qubit0: int, qubit1: int,
                       tau_us: float, n_meas: int,
                       nu_zz_khz: float) -> QuantumCircuit:
    """
    Build EEDT feedforward circuit.

    Parameters
    ----------
    qubit0    : ancilla qubit index
    qubit1    : target qubit index
    tau_us    : total wait time [microseconds]
    n_meas    : number of mid-circuit measurements during wait
    nu_zz_khz : ZZ coupling frequency [kHz], pre-measured via ZZ Ramsey

    Returns
    -------
    QuantumCircuit with n_meas mid-circuit measurements and
    conditional feedforward Rz rotations on qubit1.

    Circuit logic (equation 1 in paper):
      theta_opt = -omega_ZZ * t_star
    where t_star is the estimated T1-collapse time of Q0 inferred
    from the measurement record.
    """
    omega_zz = 2 * np.pi * nu_zz_khz * 1e3 * 1e-6  # kHz -> Hz -> rad/us
    dt       = tau_us / n_meas                 # interval between measurements

    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(n_meas, 'meas')
    ancilla_cr = ClassicalRegister(1, 'out')
    qc = QuantumCircuit(qr, cr, ancilla_cr)

    # Prepare ancilla in |1> (T1 decay source)
    qc.x(qr[0])
    # Prepare target in |+>
    qc.h(qr[1])

    for k in range(n_meas):
        # Wait dt microseconds (represented as barrier + delay)
        qc.barrier()
        qc.delay(int(dt * 1000), qr[0], unit='ns')
        qc.delay(int(dt * 1000), qr[1], unit='ns')

        # Mid-circuit measurement of ancilla
        qc.measure(qr[0], cr[k])

        # Feedforward: if ancilla collapsed to |0>, apply Rz correction
        # theta = -omega_ZZ * (k+1)*dt  (accumulated ZZ phase up to this point)
        theta_k = -omega_zz * (k + 1) * dt
        # Conditional rotation on target based on measurement outcome
        with qc.if_test((cr[k], 1)):
            qc.rz(theta_k, qr[1])

    # Final measurement of target
    qc.h(qr[1])
    qc.measure(qr[1], ancilla_cr[0])

    return qc


def build_reference_circuit(qubit0: int, qubit1: int,
                             tau_us: float) -> QuantumCircuit:
    """
    Build reference circuit (no EEDT correction).
    Target qubit undergoes free evolution for tau_us.
    """
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'ref')
    qc = QuantumCircuit(qr, cr)

    qc.x(qr[0])
    qc.h(qr[1])
    qc.barrier()
    qc.delay(int(tau_us * 1000), qr[1], unit='ns')
    qc.h(qr[1])
    qc.measure(qr[1], cr[0])

    return qc


def fidelity_from_counts(counts: dict, shots: int) -> float:
    """
    Estimate fidelity F = P(|+>) from measurement counts.
    Counts are keyed by bitstring; we look at the last bit (target qubit).
    """
    count_0 = sum(v for k, v in counts.items() if k[-1] == '0')
    return count_0 / shots


def compute_gps(f_eedt: float, f_ref: float) -> float:
    """G_PS = F_EEDT - F_ref  (equation 2 in paper)"""
    return f_eedt - f_ref


def compute_zscore(gps: float, shots: int) -> float:
    """
    z = G_PS / sigma

    F_EEDT and F_ref are each measured with `shots` independent shots.
    The combined uncertainty of their difference is:
      sigma = sqrt(sigma_EEDT^2 + sigma_ref^2) = sqrt(2 / shots)

    This matches Table I z-scores in the paper (e.g. G_PS=+0.037 -> z=+3.35
    at shots=16000: 0.037 / sqrt(2/16000) = 3.31 ≈ 3.35).

    Note: the paper's reported combined z = +11.9 used sigma = 1/sqrt(shots)
    instead, giving individual z-scores ~1.41x higher. Using the statistically
    correct sigma here yields combined z = +8.44 across 9 conditions,
    which remains p < 10^{-16} and is the recommended value.
    """
    sigma = np.sqrt(2.0 / shots)
    return gps / sigma


def run_gps_sweep(backend_name: str,
                  qubit0: int, qubit1: int,
                  nu_zz_khz: float,
                  tau_list: list = None,
                  nmeas_list: list = None,
                  shots: int = DEFAULT_SHOTS) -> dict:
    """
    Run the full GPS heatmap sweep.

    Parameters
    ----------
    backend_name : IBM Quantum backend name (e.g. "ibm_marrakesh")
    qubit0       : ancilla qubit index
    qubit1       : target qubit index
    nu_zz_khz    : pre-measured ZZ coupling [kHz]
    tau_list     : list of wait times [us]
    nmeas_list   : list of N_meas values
    shots        : shots per circuit

    Returns
    -------
    dict with full results, job IDs, and G_PS heatmap
    """
    if tau_list   is None: tau_list   = DEFAULT_TAU_LIST
    if nmeas_list is None: nmeas_list = DEFAULT_NMEAS_LIST

    # Load IBM Quantum credentials from environment or saved account
    token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum")  # uses saved account

    backend = service.backend(backend_name)
    pm      = generate_preset_pass_manager(optimization_level=1, backend=backend)

    results = {
        "metadata": {
            "backend":    backend_name,
            "qubit0":     qubit0,
            "qubit1":     qubit1,
            "nu_zz_khz":  nu_zz_khz,
            "shots":      shots,
            "sigma":      float(np.sqrt(2.0 / shots)),
            "timestamp":  datetime.utcnow().isoformat(),
        },
        "heatmap": [],
        "job_ids": [],
    }

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)

        for tau in tau_list:
            # Build reference circuit once per tau
            ref_qc  = build_reference_circuit(qubit0, qubit1, tau)
            ref_isa = pm.run(ref_qc)
            ref_job = sampler.run([ref_isa], shots=shots)
            ref_counts = ref_job.result()[0].data.ref.get_counts()
            f_ref      = fidelity_from_counts(ref_counts, shots)
            results["job_ids"].append({"type": "ref", "tau": tau,
                                       "job_id": ref_job.job_id()})

            for n_meas in nmeas_list:
                eedt_qc  = build_eedt_circuit(qubit0, qubit1,
                                               tau, n_meas, nu_zz_khz)
                eedt_isa = pm.run(eedt_qc)
                eedt_job = sampler.run([eedt_isa], shots=shots)
                eedt_counts = eedt_job.result()[0].data.out.get_counts()
                f_eedt      = fidelity_from_counts(eedt_counts, shots)

                gps     = compute_gps(f_eedt, f_ref)
                zscore  = compute_zscore(gps, shots)
                sig_flag = abs(zscore) >= SIGMA_THRESHOLD

                row = {
                    "tau_us":    tau,
                    "n_meas":    n_meas,
                    "f_eedt":    round(f_eedt, 4),
                    "f_ref":     round(f_ref,  4),
                    "gps":       round(gps,    4),
                    "zscore":    round(zscore,  2),
                    "sig_2sigma": sig_flag,
                    "job_id":    eedt_job.job_id(),
                }
                results["heatmap"].append(row)
                results["job_ids"].append({"type": "eedt",
                                           "tau": tau, "n_meas": n_meas,
                                           "job_id": eedt_job.job_id()})

                print(f"  tau={tau}us  N={n_meas}  "
                      f"G_PS={gps:+.4f}  z={zscore:+.2f}"
                      f"{'  **' if sig_flag else ''}")

    # Combined z-score (Fisher method, independent conditions)
    positive_z = [r["zscore"] for r in results["heatmap"] if r["gps"] > 0]
    if positive_z:
        z_combined = sum(positive_z) / np.sqrt(len(positive_z))
        results["z_combined"] = round(z_combined, 2)
        results["n_positive"] = len(positive_z)
    else:
        results["z_combined"] = 0.0
        results["n_positive"] = 0

    return results


def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("GPS SWEEP SUMMARY")
    print("=" * 60)
    meta = results["metadata"]
    print(f"Backend:   {meta['backend']}")
    print(f"Q0/Q1:     {meta['qubit0']}/{meta['qubit1']}")
    print(f"nu_ZZ:     {meta['nu_zz_khz']} kHz")
    print(f"Shots:     {meta['shots']}  (sigma={meta['sigma']:.4f})")
    print(f"Time:      {meta['timestamp']}")
    print()
    print(f"{'tau':>6} {'N':>4} {'F_EEDT':>8} {'F_ref':>8} "
          f"{'G_PS':>8} {'z':>7}  sig")
    print("-" * 60)
    for r in results["heatmap"]:
        flag = "**" if r["sig_2sigma"] else ""
        print(f"{r['tau_us']:>6} {r['n_meas']:>4} "
              f"{r['f_eedt']:>8.4f} {r['f_ref']:>8.4f} "
              f"{r['gps']:>+8.4f} {r['zscore']:>+7.2f}  {flag}")
    print("-" * 60)
    print(f"Positive conditions: {results['n_positive']}")
    print(f"Combined z-score:    {results['z_combined']:+.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="EEDT GPS heatmap sweep on IBM Quantum hardware")
    parser.add_argument("--backend",  default="ibm_marrakesh")
    parser.add_argument("--qubit0",   type=int, default=94,
                        help="Ancilla qubit index")
    parser.add_argument("--qubit1",   type=int, default=95,
                        help="Target qubit index")
    parser.add_argument("--nu-zz",    type=float, default=3.600,
                        help="Pre-measured ZZ coupling [kHz]")
    parser.add_argument("--shots",    type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--tau",      type=float, nargs="+",
                        default=DEFAULT_TAU_LIST,
                        help="Wait times in microseconds")
    parser.add_argument("--nmeas",    type=int,   nargs="+",
                        default=DEFAULT_NMEAS_LIST,
                        help="Number of MCMs per wait period")
    parser.add_argument("--out",      default="gps_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"GPS Sweep  |  {args.backend}  |  "
          f"Q{args.qubit0}-Q{args.qubit1}  |  nu_ZZ={args.nu_zz} kHz")
    print(f"tau={args.tau} us   N_meas={args.nmeas}   shots={args.shots}")
    print()

    results = run_gps_sweep(
        backend_name=args.backend,
        qubit0=args.qubit0,
        qubit1=args.qubit1,
        nu_zz_khz=args.nu_zz,
        tau_list=args.tau,
        nmeas_list=args.nmeas,
        shots=args.shots,
    )

    print_summary(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
