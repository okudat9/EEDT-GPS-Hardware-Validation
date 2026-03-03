"""
zz_ramsey.py
============
ZZ coupling characterization via Ramsey spectroscopy.
Also measures MCM (mid-circuit measurement) backaction on T2.

Corresponds to paper Sections III (ZZ drift) and IV (MCM backaction).

Usage:
  # Measure nu_ZZ for a qubit pair
  python zz_ramsey.py --backend ibm_marrakesh --qubit0 94 --qubit1 95

  # Track ZZ drift across multiple sessions
  python zz_ramsey.py --backend ibm_marrakesh --qubit0 94 --qubit1 95 --drift-track

  # Measure MCM backaction on T2
  python zz_ramsey.py --backend ibm_marrakesh --qubit0 94 --qubit1 95 --mcm-backaction

IBM Quantum access:
  Set your IBM Quantum token via environment variable:
    export IBM_QUANTUM_TOKEN="your_token_here"
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# Default parameters
ZZ_TAU_LIST_US = list(range(10, 130, 10)) + [150, 180, 210, 250, 300]  # us (extended for T2 > 200us)
T2_TAU_LIST_US = [10, 20, 30, 40, 60, 80, 100, 120, 150, 200]
SHOTS          = 4096


# ─────────────────────────────────────────────
# Circuit builders
# ─────────────────────────────────────────────

def zz_ramsey_circuit_q0_on(qubit0: int, qubit1: int, tau_us: float) -> QuantumCircuit:
    """ZZ Ramsey with Q0=|1> (ZZ interaction ON)."""
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.x(qr[0])
    qc.h(qr[1])
    qc.barrier()
    qc.delay(int(tau_us * 1000), qr[1], unit='ns')
    qc.barrier()
    qc.h(qr[1])
    qc.measure(qr[1], cr[0])
    return qc


def zz_ramsey_circuit_q0_off(qubit0: int, qubit1: int, tau_us: float) -> QuantumCircuit:
    """Ramsey with Q0=|0> (ZZ interaction OFF; measures bare T2 of Q1)."""
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    # Q0 stays in |0>
    qc.h(qr[1])
    qc.barrier()
    qc.delay(int(tau_us * 1000), qr[1], unit='ns')
    qc.barrier()
    qc.h(qr[1])
    qc.measure(qr[1], cr[0])
    return qc


def t2_with_mcm_circuit(qubit0: int, qubit1: int,
                         tau_us: float, n_mcm: int) -> QuantumCircuit:
    """
    T2 measurement with n_mcm mid-circuit measurements on Q0 during the wait.
    Used to quantify MCM backaction on Q1 coherence.
    """
    dt = tau_us / max(n_mcm, 1)
    qr = QuantumRegister(2, 'q')
    cr_mcm = ClassicalRegister(n_mcm, 'mcm')
    cr_out = ClassicalRegister(1, 'out')
    qc = QuantumCircuit(qr, cr_mcm, cr_out)

    qc.x(qr[0])      # Q0 = |1>
    qc.h(qr[1])      # Q1 = |+>
    qc.barrier()

    for k in range(n_mcm):
        qc.delay(int(dt * 1000), qr[1], unit='ns')
        qc.measure(qr[0], cr_mcm[k])
        qc.barrier()

    qc.h(qr[1])
    qc.measure(qr[1], cr_out[0])
    return qc


# ─────────────────────────────────────────────
# Fitting functions
# ─────────────────────────────────────────────

def zz_ramsey_model(tau, A, nu_zz, phi, offset, T2):
    """F(tau) = A * exp(-tau/T2) * cos(2*pi*nu_ZZ*tau + phi) + offset"""
    return A * np.exp(-tau / T2) * np.cos(2 * np.pi * nu_zz * tau + phi) + offset


def t2_decay_model(tau, A, T2, offset):
    """F(tau) = A * exp(-tau/T2) + offset"""
    return A * np.exp(-tau / T2) + offset


def fit_zz_ramsey(tau_list: list, fidelities: list) -> dict:
    """Fit ZZ Ramsey data to extract nu_ZZ and T2^ZZ."""
    tau_arr = np.array(tau_list, dtype=float)
    fid_arr = np.array(fidelities, dtype=float)

    # Initial guess via FFT
    dt    = tau_arr[1] - tau_arr[0] if len(tau_arr) > 1 else 10.0
    freqs = np.fft.rfftfreq(len(tau_arr), d=dt)
    fft   = np.abs(np.fft.rfft(fid_arr - fid_arr.mean()))
    peak  = np.argmax(fft[1:]) + 1
    nu0   = freqs[peak]    # 1/us = MHz

    p0  = [0.45, nu0, 0.0, 0.5, 100.0]
    try:
        popt, pcov = curve_fit(
            zz_ramsey_model, tau_arr, fid_arr,
            p0=p0,
            bounds=([-1, 0, -np.pi, 0, 1], [1, 10, np.pi, 1, 1000]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "nu_zz_khz":     round(popt[1] * 1e3, 3),  # MHz -> kHz
            "nu_zz_err_khz": round(perr[1] * 1e3, 3),
            "T2_zz_us":      round(popt[4], 1),
            "T2_zz_err_us":  round(perr[4], 1),
            "amplitude":     round(popt[0], 4),
            "offset":        round(popt[3], 4),
            "phi_rad":       round(popt[2], 4),
            "fit_method":    "curve_fit",
        }
    except RuntimeError:
        # Fallback to FFT estimate only
        return {
            "nu_zz_khz":  round(nu0 * 1e3, 3),
            "T2_zz_us":   None,
            "fit_method": "fft_fallback",
        }


def fit_t2_decay(tau_list: list, fidelities: list) -> dict:
    """Fit simple T2 decay curve."""
    tau_arr = np.array(tau_list, dtype=float)
    fid_arr = np.array(fidelities, dtype=float)
    try:
        popt, pcov = curve_fit(
            t2_decay_model, tau_arr, fid_arr,
            p0=[0.45, 80.0, 0.5],
            bounds=([0, 1, 0], [1, 2000, 1]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "T2_eff_us":     round(popt[1], 1),
            "T2_eff_err_us": round(perr[1], 1),
            "amplitude":     round(popt[0], 4),
            "offset":        round(popt[2], 4),
            "fit_method":    "curve_fit",
        }
    except RuntimeError:
        return {"T2_eff_us": None, "fit_method": "fit_failed"}


# ─────────────────────────────────────────────
# Main measurement functions
# ─────────────────────────────────────────────

def measure_nu_zz(backend, qubit0: int, qubit1: int,
                  tau_list: list = None,
                  shots: int = SHOTS) -> dict:
    """
    Measure nu_ZZ for a qubit pair.
    Returns nu_ZZ in kHz and T2^ZZ in us.
    """
    if tau_list is None:
        tau_list = ZZ_TAU_LIST_US

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    fid_on  = []
    fid_off = []

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        for tau in tau_list:
            # ZZ ON
            qc_on  = zz_ramsey_circuit_q0_on(qubit0, qubit1, tau)
            isa_on = pm.run(qc_on)
            job_on = sampler.run([isa_on], shots=shots)
            c_on   = job_on.result()[0].data.c.get_counts()
            fid_on.append(sum(v for k, v in c_on.items() if k[-1] == '0') / shots)

            # ZZ OFF (bare T2)
            qc_off  = zz_ramsey_circuit_q0_off(qubit0, qubit1, tau)
            isa_off = pm.run(qc_off)
            job_off = sampler.run([isa_off], shots=shots)
            c_off   = job_off.result()[0].data.c.get_counts()
            fid_off.append(sum(v for k, v in c_off.items() if k[-1] == '0') / shots)

    fit_on  = fit_zz_ramsey(tau_list, fid_on)
    fit_off = fit_t2_decay(tau_list, fid_off)

    return {
        "qubit0":      qubit0,
        "qubit1":      qubit1,
        "backend":     backend.name,
        "timestamp":   datetime.utcnow().isoformat(),
        "tau_list_us": tau_list,
        "fid_on":      [round(f, 4) for f in fid_on],
        "fid_off":     [round(f, 4) for f in fid_off],
        "zz_fit":      fit_on,
        "t2_bare_fit": fit_off,
    }


def measure_mcm_backaction(backend, qubit0: int, qubit1: int,
                            nmeas_list: list = None,
                            tau_us: float = 50.0,
                            shots: int = SHOTS) -> dict:
    """
    Measure T2 degradation due to MCM backaction.
    Runs T2 decay with 0, 1, 2, ... MCMs per wait period.
    alpha = T2_eff / T2_ZZ (paper reports alpha = 0.38 +/- 0.05)
    """
    if nmeas_list is None:
        nmeas_list = [0, 1, 2, 3, 4, 5, 6]

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    results = []

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        for n_mcm in nmeas_list:
            fidelities = []
            for tau in T2_TAU_LIST_US:
                if n_mcm == 0:
                    qc = zz_ramsey_circuit_q0_on(qubit0, qubit1, tau)
                else:
                    qc = t2_with_mcm_circuit(qubit0, qubit1, tau, n_mcm)
                isa = pm.run(qc)
                job = sampler.run([isa], shots=shots)
                res = job.result()[0].data
                cr_name = 'out' if n_mcm > 0 else 'c'
                counts = getattr(res, cr_name).get_counts()
                fid = sum(v for k, v in counts.items() if k[-1] == '0') / shots
                fidelities.append(fid)

            t2_fit = fit_t2_decay(T2_TAU_LIST_US, fidelities)
            results.append({
                "n_mcm":       n_mcm,
                "tau_us":      tau_us,
                "T2_eff_us":   t2_fit.get("T2_eff_us"),
                "fidelities":  [round(f, 4) for f in fidelities],
            })
            print(f"  N_MCM={n_mcm}  T2_eff={t2_fit.get('T2_eff_us')} us")

    # Compute alpha = T2_eff / T2_bare
    t2_bare = results[0]["T2_eff_us"] if results else None
    for r in results:
        if t2_bare and r["T2_eff_us"]:
            r["alpha"] = round(r["T2_eff_us"] / t2_bare, 3)
        else:
            r["alpha"] = None

    return {
        "qubit0":    qubit0,
        "qubit1":    qubit1,
        "backend":   backend.name,
        "timestamp": datetime.utcnow().isoformat(),
        "t2_bare_us": t2_bare,
        "mcm_backaction": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ZZ Ramsey characterization and MCM backaction measurement")
    parser.add_argument("--backend",      default="ibm_marrakesh")
    parser.add_argument("--qubit0",       type=int, default=94)
    parser.add_argument("--qubit1",       type=int, default=95)
    parser.add_argument("--shots",        type=int, default=SHOTS)
    parser.add_argument("--mcm-backaction", action="store_true",
                        help="Measure MCM backaction on T2")
    parser.add_argument("--drift-track",  action="store_true",
                        help="Print nu_ZZ for drift monitoring (run repeatedly)")
    parser.add_argument("--out",          default="zz_ramsey_results.json")
    args = parser.parse_args()

    token = os.environ.get("IBM_QUANTUM_TOKEN", None)
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum")

    backend = service.backend(args.backend)

    all_results = {}

    # Always measure nu_ZZ
    print(f"Measuring nu_ZZ for Q{args.qubit0}-Q{args.qubit1} "
          f"on {args.backend}...")
    zz_result = measure_nu_zz(backend, args.qubit0, args.qubit1,
                               shots=args.shots)
    all_results["zz_ramsey"] = zz_result

    nu_zz = zz_result["zz_fit"].get("nu_zz_khz")
    t2_zz = zz_result["zz_fit"].get("T2_zz_us")
    print(f"  nu_ZZ = {nu_zz} kHz")
    print(f"  T2^ZZ = {t2_zz} us")

    if args.drift_track:
        print("\nDrift tracking mode: run this script periodically "
              "to monitor nu_ZZ stability.")

    if args.mcm_backaction:
        print("\nMeasuring MCM backaction...")
        mcm_result = measure_mcm_backaction(
            backend, args.qubit0, args.qubit1, shots=args.shots)
        all_results["mcm_backaction"] = mcm_result

        t2_bare = mcm_result.get("t2_bare_us")
        print("\nMCM backaction summary:")
        print(f"  T2_bare = {t2_bare} us")
        for r in mcm_result["mcm_backaction"]:
            print(f"  N_MCM={r['n_mcm']}  T2_eff={r['T2_eff_us']} us  "
                  f"alpha={r['alpha']}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
