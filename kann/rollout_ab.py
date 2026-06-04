#!/usr/bin/env python3
"""
rollout_ab.py
Compiles and runs MLP and GRU at default settings, then plots rollout
divergence: MSE as a function of time t over [0, 3.0].

Training range is t in [0, 2.0].  The shaded region t > 2.0 is
out-of-distribution — this is where generalization breaks down.

Run from ~/kann:
    python3 rollout_ab.py
"""

import subprocess
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

MODELS = {
    'MLP (dense+tanh)': ('simple_mlp_ab.c', 'steelblue'),
    'GRU':              ('simple_rnn_ab.c',  'coral'),
}
SHARED   = ['kautodiff.c', 'kann.c']
N_T_EVAL = 25          # must match the value in the C files
T_MAX    = 3.0
T_TRAIN  = 2.0         # training range boundary

ROLLOUT_RE = re.compile(r'^ROLLOUT\s+([\d.\s]+)', re.MULTILINE)
MSE_RE     = re.compile(r'Validation MSE:\s*([\d.eE+\-]+)')

# ── Compile and run each model ────────────────────────────────────────────────
rollout_data = {}   # { label: np.array of shape (N_T_EVAL+1,) }
overall_mse  = {}   # { label: float }

for label, (src, color) in MODELS.items():
    binary = f'./rollout_tmp_{label.split()[0].lower()}'
    print(f'Compiling {label} ...', flush=True)
    cp = subprocess.run(
        ['gcc', src] + SHARED + ['-lm', '-o', binary],
        capture_output=True, text=True
    )
    if cp.returncode != 0:
        print(f'Compile error:\n{cp.stderr}')
        sys.exit(1)

    print(f'Running   {label} ...', flush=True)
    rp = subprocess.run([binary], capture_output=True, text=True)
    os.remove(binary)

    # overall validation MSE
    m = MSE_RE.search(rp.stdout)
    if m:
        overall_mse[label] = float(m.group(1))

    # rollout MSE values
    m = ROLLOUT_RE.search(rp.stdout)
    if not m:
        print(f'No ROLLOUT line found in output:\n{rp.stdout}')
        sys.exit(1)
    vals = list(map(float, m.group(1).split()))
    rollout_data[label] = np.array(vals)
    print(f'  overall MSE = {overall_mse.get(label, "?"):.6f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
t_vals = np.linspace(0, T_MAX, N_T_EVAL + 1)

fig, ax = plt.subplots(figsize=(10, 5))

# shade out-of-distribution region
ax.axvspan(T_TRAIN, T_MAX, alpha=0.08, color='gray', label='Out of training range (t > 2)')
ax.axvline(T_TRAIN, color='gray', linestyle='--', linewidth=1)

for label, (src, color) in MODELS.items():
    mse_curve = rollout_data[label]
    ax.plot(t_vals, mse_curve, color=color, linewidth=2,
            label=f'{label}  (overall MSE={overall_mse.get(label, 0):.4f})')
    ax.scatter(t_vals, mse_curve, color=color, s=20, zorder=3)

ax.set_xlabel('Time  t', fontsize=12)
ax.set_ylabel('MSE at time slice  (averaged over x)', fontsize=11)
ax.set_title(
    'Rollout Divergence — MLP vs GRU\n'
    'u(x,t) = tanh(x)·exp(−t)  |  trained on t ∈ [0, 2],  evaluated on t ∈ [0, 3]',
    fontsize=12
)
ax.legend(fontsize=10)
ax.set_xlim(0, T_MAX)
ax.set_ylim(bottom=0)

plt.tight_layout()
out = 'rollout_divergence_tan.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()
