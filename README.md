# EEDT-GPS-Hardware-Validation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18848145.svg)](https://doi.org/10.5281/zenodo.18848145)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-2.x-purple)

**Hardware Operating Conditions for Entanglement-Enhanced Dynamical Tracking (EEDT)**  
Systematic measurement of ZZ-drift, MCM backaction, and correction gain on IBM Heron r2.

> Companion paper (EEDT v8): [10.5281/zenodo.18809139](https://doi.org/10.5281/zenodo.18809139)

---

## Key Results

| Condition | τ (μs) | N_meas | G_PS | z-score |
|-----------|--------|--------|------|---------|
| **Peak gain** | 30 | 1 | **+0.037** | **+3.35** ✅ |
| Operating window | 30–50 | 1–5 | +0.035 ± 0.002 | — |
| Combined (9 pts) | — | — | — | **+11.9** |
| Q6–Q7 (below threshold) | all | all | < 0 | — |

**Main findings:**
- T₂^ZZ > **200 μs** is a necessary condition for G_PS > 0 on IBM Heron r2
- MCM backaction degrades T₂ to α ≈ **0.38** × T₂^ZZ
- Inter-session ZZ drift: 0.13–3.60 kHz (**26× range** over two days)
- Operating window: τ\* ∈ [30, 50] μs,  N\* ∈ [1, 5]

---

## Repository Structure

```
├── gps_sweep.py           GPS heatmap experiment (reproduces Table I)
├── qubit_screening.py     Qubit pair screening algorithm
├── zz_ramsey.py           ZZ coupling measurement & MCM backaction
└── requirements.txt       Python dependencies
```

Full dataset and paper (PDF/TEX/figures) on Zenodo:  
👉 https://doi.org/10.5281/zenodo.18848145

---

## Hardware

All experiments on **ibm_marrakesh** (IBM Heron r2, 156 qubits) via IBM Quantum open access.

| Qubit pair | T₂^ZZ (μs) | ν_ZZ (kHz) | Result |
|---|---|---|---|
| Q94–Q95 | 261.5 | 3.600 | ✅ G_PS > 0 |
| Q6–Q7 | 120.0 | 3.46 | ❌ G_PS < 0 (below threshold) |

---

## Installation

```bash
pip install -r requirements.txt
```

Set IBM Quantum token via environment variable:

```bash
# Windows
set IBM_QUANTUM_TOKEN=your_token_here

# Mac/Linux
export IBM_QUANTUM_TOKEN=your_token_here
```

---

## Usage

### Step 1 — Screen qubits
```bash
python qubit_screening.py --backend ibm_marrakesh --top 3
```

### Step 2 — Measure ZZ coupling
```bash
python zz_ramsey.py --backend ibm_marrakesh --qubit0 94 --qubit1 95
```

### Step 3 — Run GPS heatmap
```bash
python gps_sweep.py --backend ibm_marrakesh --qubit0 94 --qubit1 95 \
    --nu-zz 3.600 --shots 16000 --tau 30 50 --nmeas 1 2 3 4 5 6
```

Output: `gps_results.json` with G_PS, z-scores, and IBM Quantum job IDs.

---

## Reproducing Paper Results (GPS v8)

```bash
python gps_sweep.py \
    --backend ibm_marrakesh \
    --qubit0 94 --qubit1 95 \
    --nu-zz 3.600 \
    --shots 16000 \
    --tau 30 50 \
    --nmeas 1 2 3 4 5 6
```

> **Note:** IBM Quantum hardware properties vary between sessions.  
> Run `zz_ramsey.py` fresh before each GPS sweep to account for ZZ drift.

---

## Citation

```bibtex
@misc{okuda_eedt_gps_2026,
  author       = {Okuda, Takeshi},
  title        = {Hardware Operating Conditions for Entanglement-Enhanced
                  Dynamical Tracking: Systematic Measurement of ZZ-Drift,
                  MCM Backaction, and Correction Gain on IBM Heron r2},
  year         = {2026},
  doi          = {10.5281/zenodo.18848145},
  url          = {https://doi.org/10.5281/zenodo.18848145},
}
```

---

## License

MIT License — see [LICENSE](LICENSE)

## Author

**Takeshi Okuda**  
Independent Quantum Computing Researcher, Japan  
GitHub: [@okudat9](https://github.com/okudat9)
