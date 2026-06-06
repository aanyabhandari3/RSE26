#!/usr/bin/env python3
"""
error_norms_ab.py
Compiles and runs MLP and GRU at default settings, then plots L2 and
L-infinity error norms as a function of time t over [0, 3.0].

  L2(t)   = sqrt( integral_x |u_pred - u_exact|^2 dx )   -- average error
  Linf(t) = max_x |u_pred - u_exact|                      -- worst-case error

The shaded region t > 2.0 is outside the training range.

Run from ~/kann:
    python3 error_norms_ab.py
"""

import subprocess
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

MODELS = {
    'MLP (dense+cos)': ('simple_mlp_ab.c', 'steelblue'),
    'GRU':              ('simple_rnn_ab.c',  'coral'),
}
SHARED   = ['kautodiff.c', 'kann.c']
N_T_EVAL = 25
T_MAX    = 3.0
T_TRAIN  = 2.0

L2_RE   = re.compile(r'^L2NORM\s+([\d.\s]+)',   re.MULTILINE)
LINF_RE = re.compile(r'^LINF\s+([\d.\s]+)',     re.MULTILINE)
MSE_RE  = re.compile(r'Validation MSE:\s*([\d.eE+\-]+)')

# ── Compile and run ──────────────────────────────────────────────────────────
data = {}   # { label: { 'l2': array, 'linf': array, 'mse': float } }

for label, (src, color) in MODELS.items():
    binary = f'./norm_tmp_{label.split()[0].lower()}'
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

    m_mse  = MSE_RE.search(rp.stdout)
    m_l2   = L2_RE.search(rp.stdout)
    m_linf = LINF_RE.search(rp.stdout)

    if not m_l2 or not m_linf:
        print(f'Missing L2NORM or LINF line in output:\n{rp.stdout}')
        sys.exit(1)

    data[label] = {
        'l2':   np.array(list(map(float, m_l2.group(1).split()))),
        'linf': np.array(list(map(float, m_linf.group(1).split()))),
        'mse':  float(m_mse.group(1)) if m_mse else 0.0,
        'color': color,
    }
    print(f'  overall MSE = {data[label]["mse"]:.6f}')

# ── Plot ─────────────────────────────────────────────────────────────────────
t_vals = np.linspace(0, T_MAX, N_T_EVAL + 1)

fig, (ax_l2, ax_linf) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'MLP vs GRU — Spatial Error Norms over Time\n'
    'u(x,t) = cos(x)·exp(−t)  |  trained on t ∈ [0, 2],  evaluated on t ∈ [0, 3]',
    fontsize=12
)

for ax, norm_key, ylabel, title in [
    (ax_l2,   'l2',   'L2 norm  √(∫|error|² dx)',    'L2 Error vs Time'),
    (ax_linf, 'linf', 'L∞ norm  max|error|',          'L∞ Error vs Time'),
]:
    # shade out-of-distribution region
    ax.axvspan(T_TRAIN, T_MAX, alpha=0.08, color='gray')
    ax.axvline(T_TRAIN, color='gray', linestyle='--', linewidth=1,
               label='Training boundary (t=2)')

    for label, d in data.items():
        ax.plot(t_vals, d[norm_key], color=d['color'], linewidth=2,
                label=f'{label}  (MSE={d["mse"]:.4f})')
        ax.scatter(t_vals, d[norm_key], color=d['color'], s=18, zorder=3)

    ax.set_xlabel('Time  t', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

plt.tight_layout()
out = 'error_norms_cos.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()
