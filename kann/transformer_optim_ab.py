#!/usr/bin/env python3
"""
transformer_optim_ab.py
Grid search over WIDTH and DEPTH for the transformer on 6 target functions.
For each function, finds the best (lowest MSE) hyperparameter combo, then
plots one subplot per function showing:
  - L2 rollout curve for the best config
  - subplot title: function name
  - subplot header: best WIDTH, DEPTH, MSE

Hyperparameter grid:
    WIDTH : 32, 64, 128
    DEPTH : 1, 2, 4
    Total : 6 functions × 9 combos = 54 runs  (~35-45 min)

Run from ~/kann:
    python3 transformer_optim_ab.py
"""

import subprocess
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SRC    = 'transformer_optim_ab.c'
SHARED = ['kautodiff.c', 'kann.c']

# ── Hyperparameter grid ───────────────────────────────────────────────────────
WIDTHS = [32, 64, 128]
DEPTHS = [1, 2, 4]

# ── Target functions ──────────────────────────────────────────────────────────
FUNCTIONS = {
    'sin(x)·exp(-t)':       '-DFUNC_SIN',
    'cos(x)·exp(-t)':       '-DFUNC_COS',
    'tanh(x)·exp(-t)':      '-DFUNC_TANH',
    'sin(x)·exp(-2t)':      '-DFUNC_FAST_DECAY',
    'sin(x)·exp(-t/2)':     '-DFUNC_SLOW_DECAY',
    'sin(2x)·exp(-t)':      '-DFUNC_SIN2X',
}

N_T_EVAL = 25
T_MAX    = 3.0
T_TRAIN  = 2.0

MSE_RE  = re.compile(r'Validation MSE:\s*([\d.eE+\-]+)')
L2_RE   = re.compile(r'^L2NORM\s+([\d.\s]+)', re.MULTILINE)
LINF_RE = re.compile(r'^LINF\s+([\d.\s]+)',   re.MULTILINE)

# ── Grid search ───────────────────────────────────────────────────────────────
# results[fn_label] = list of dicts:
#   { 'width', 'depth', 'mse', 'l2': array, 'linf': array }
results = {fn: [] for fn in FUNCTIONS}

total  = len(FUNCTIONS) * len(WIDTHS) * len(DEPTHS)
run_n  = 0

for fn_label, fn_flag in FUNCTIONS.items():
    print(f'\n── {fn_label} ──')
    for width in WIDTHS:
        for depth in DEPTHS:
            run_n += 1
            tag    = f'optim_{fn_label.split("(")[0].strip()}_{width}_{depth}'
            binary = f'./optim_tmp_{tag}'

            compile_cmd = [
                'gcc',
                fn_flag,
                f'-DWIDTH={width}',
                f'-DDEPTH={depth}',
                '-Wno-macro-redefined',
                SRC, *SHARED, '-lm', '-o', binary
            ]
            print(f'  [{run_n:2d}/{total}]  W={width:3d}  D={depth}  ...', end='  ', flush=True)
            cp = subprocess.run(compile_cmd, capture_output=True, text=True)
            if cp.returncode != 0:
                print(f'\nCompile error:\n{cp.stderr}')
                sys.exit(1)

            rp = subprocess.run([binary], capture_output=True, text=True)
            os.remove(binary)

            m_mse  = MSE_RE.search(rp.stdout)
            m_l2   = L2_RE.search(rp.stdout)
            m_linf = LINF_RE.search(rp.stdout)

            if not m_mse or not m_l2 or not m_linf:
                print(f'\nMissing output lines:\n{rp.stdout}')
                sys.exit(1)

            mse = float(m_mse.group(1))
            print(f'MSE = {mse:.6f}')

            results[fn_label].append({
                'width': width,
                'depth': depth,
                'mse':   mse,
                'l2':    np.array(list(map(float, m_l2.group(1).split()))),
                'linf':  np.array(list(map(float, m_linf.group(1).split()))),
            })

# ── Find best config per function ─────────────────────────────────────────────
best = {}
for fn_label, runs in results.items():
    best[fn_label] = min(runs, key=lambda r: r['mse'])

print('\n── Best configurations ──')
for fn_label, b in best.items():
    print(f'  {fn_label:25s}  W={b["width"]:3d}  D={b["depth"]}  MSE={b["mse"]:.6f}')

# ── Plot: 2 rows × 3 cols, one subplot per function ──────────────────────────
t_vals   = np.linspace(0, T_MAX, N_T_EVAL + 1)
fn_list  = list(FUNCTIONS.keys())
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes_flat = axes.flatten()

# colour map for all hyperparameter combos shown as thin background lines
CMAP = plt.cm.Blues

for idx, fn_label in enumerate(fn_list):
    ax  = axes_flat[idx]
    b   = best[fn_label]
    all_runs = results[fn_label]

    # ── shade out-of-distribution region
    ax.axvspan(T_TRAIN, T_MAX, alpha=0.07, color='gray')
    ax.axvline(T_TRAIN, color='gray', linestyle='--', linewidth=0.8)

    # ── light background lines: all other configs (L2 only)
    for r in all_runs:
        if r is not b:
            ax.plot(t_vals, r['l2'], color='lightsteelblue',
                    linewidth=0.8, alpha=0.5, zorder=1)

    # ── best config: bold line
    ax.plot(t_vals, b['l2'], color='steelblue', linewidth=2.5,
            label='L2  best', zorder=3)
    ax.plot(t_vals, b['linf'], color='coral', linewidth=2.5,
            linestyle='--', label='L∞  best', zorder=3)
    ax.scatter(t_vals, b['l2'],   color='steelblue', s=16, zorder=4)
    ax.scatter(t_vals, b['linf'], color='coral',     s=16, zorder=4)

    # ── subplot title: function name + optimal hyperparameters
    ax.set_title(
        f'$u(x,t)$ = {fn_label}\n'
        f'Best: WIDTH={b["width"]}, DEPTH={b["depth"]}, MSE={b["mse"]:.5f}',
        fontsize=9
    )
    ax.set_xlabel('Time  t', fontsize=8)
    ax.set_ylabel('Error norm', fontsize=8)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(labelsize=7)

fig.suptitle(
    'Transformer Hyperparameter Optimisation — L2 & L∞ Rollout per Target Function\n'
    'Grid: WIDTH ∈ {32, 64, 128}  ×  DEPTH ∈ {1, 2, 4}  |  '
    'Light lines = all configs,  Bold = best config per function',
    fontsize=11
)
plt.tight_layout()
out = 'transformer_optim.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()
