#!/usr/bin/env python3
"""
sweep_functions.py
Runs MLP, GRU, and Transformer on selected target functions.

For each combination, collects L2 and L-infinity error norms vs time.
Produces a 2-row × N-column figure:
    row 1 — L2 norm vs t  (one subplot per function)
    row 2 — L∞ norm vs t  (one subplot per function)

Training range: t ∈ [0, 2].  Shaded region t > 2 is out-of-distribution.

Run from ~/kann:
    python3 sweep_functions.py
"""

import subprocess
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Function selection ─────────────────────────────────────────────────────────
def _to_c_float(expr):
    """Convert math function names to C single-precision variants (sinf, expf …)."""
    for src, dst in [
        ('sinh','sinhf'), ('cosh','coshf'), ('tanh','tanhf'),
        ('asin','asinf'), ('acos','acosf'), ('atan2','atan2f'), ('atan','atanf'),
        ('sin','sinf'),   ('cos','cosf'),   ('tan','tanf'),
        ('expm1','expm1f'), ('exp2','exp2f'), ('exp','expf'),
        ('log10','log10f'), ('log2','log2f'), ('log1p','log1pf'), ('log','logf'),
        ('sqrt','sqrtf'), ('cbrt','cbrtf'), ('pow','powf'),
        ('fabs','fabsf'), ('abs','fabsf'),
        ('ceil','ceilf'), ('floor','floorf'), ('round','roundf'),
    ]:
        expr = re.sub(rf'\b{src}\b(?=\s*\()', dst, expr)
    expr = re.sub(r'\bpi\b', '(float)M_PI', expr, flags=re.IGNORECASE)
    return expr

def _ask_functions():
    """Collect one or more function expressions from the user."""
    print('\nEnter target functions u(x,t) to compare, one per line.')
    print('Press Enter on an empty line when done (need at least 1).')
    print('Math functions auto-convert to C floats: sin→sinf, exp→expf, etc.')
    print('Examples:  sin(x)*exp(-t)    cos(2*x)*exp(-t/2)    tanh(x)*(1-t/3)')
    print()
    fns = {}   # label → compile flags list
    i = 1
    while True:
        raw = input(f'Function {i}: ').strip()
        if not raw:
            if fns:
                break
            print('  Enter at least one function.')
            continue
        c_expr = _to_c_float(raw)
        print(f'  → C: {c_expr}')
        fns[raw] = [f'-DTARGET_FN(x,t)=({c_expr})']
        i += 1
    print(f'\n→ Running {len(fns)} function(s)\n')
    return fns

# ── Configuration ─────────────────────────────────────────────────────────────
FUNCTIONS = _ask_functions()   # { label: [compile_flags] }
MODELS = {
    'MLP':         ('simple_mlp_ab.c',          'steelblue'),
    'GRU':         ('simple_rnn_ab.c',           'coral'),
    'Transformer': ('simple_transformer_ab.c',   'mediumseagreen'),
}
SHARED   = ['kautodiff.c', 'kann.c']
N_T_EVAL = 25
T_MAX    = 3.0
T_TRAIN  = 2.0

L2_RE   = re.compile(r'^L2NORM\s+([\d.\s]+)', re.MULTILINE)
LINF_RE = re.compile(r'^LINF\s+([\d.\s]+)',   re.MULTILINE)
MSE_RE  = re.compile(r'Validation MSE:\s*([\d.eE+\-]+)')

# ── Collect results ───────────────────────────────────────────────────────────
# results[fn_label][model_label] = { 'l2': array, 'linf': array, 'mse': float }
results = {fn: {} for fn in FUNCTIONS}

for fn_label, fn_flags in FUNCTIONS.items():
    print(f'\n── {fn_label} ──')
    for model_label, (src, color) in MODELS.items():
        tag    = f'fn_{fn_label.split("(")[0].strip()}_{model_label.lower()}'
        binary = f'./sweep_fn_tmp_{tag}'

        compile_cmd = ['gcc', '-Wno-macro-redefined'] + fn_flags + [src] + SHARED + ['-lm', '-o', binary]
        print(f'  compiling {model_label:12s} ...', end='  ', flush=True)
        cp = subprocess.run(compile_cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            print(f'\nCompile error:\n{cp.stderr}')
            sys.exit(1)

        rp = subprocess.run([binary], capture_output=True, text=True)
        os.remove(binary)

        m_mse  = MSE_RE.search(rp.stdout)
        m_l2   = L2_RE.search(rp.stdout)
        m_linf = LINF_RE.search(rp.stdout)

        if not m_l2 or not m_linf:
            print(f'\nMissing L2NORM or LINF in output:\n{rp.stdout}')
            sys.exit(1)

        mse = float(m_mse.group(1)) if m_mse else 0.0
        print(f'MSE = {mse:.6f}')

        results[fn_label][model_label] = {
            'l2':   np.array(list(map(float, m_l2.group(1).split()))),
            'linf': np.array(list(map(float, m_linf.group(1).split()))),
            'mse':  mse,
            'color': color,
        }

# ── Plot ──────────────────────────────────────────────────────────────────────
t_vals  = np.linspace(0, T_MAX, N_T_EVAL + 1)
fn_list = list(FUNCTIONS.keys())

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey='row')
fig.suptitle(
    'MLP vs GRU vs Transformer — Error Norms across Target Functions\n'
    'Defaults: NX=NT=50, WIDTH=256 (MLP/GRU) / 64 (Transformer), DEPTH=1',
    fontsize=12
)

norm_rows = [
    ('l2',   'L2 norm  √(∫|error|² dx)'),
    ('linf', 'L∞ norm  max|error|'),
]

for row_idx, (norm_key, ylabel) in enumerate(norm_rows):
    for col_idx, fn_label in enumerate(fn_list):
        ax = axes[row_idx][col_idx]

        # shade out-of-distribution region
        ax.axvspan(T_TRAIN, T_MAX, alpha=0.08, color='gray')
        ax.axvline(T_TRAIN, color='gray', linestyle='--', linewidth=0.8)

        for model_label, d in results[fn_label].items():
            mse_str = f'MSE={d["mse"]:.4f}'
            ax.plot(t_vals, d[norm_key], color=d['color'], linewidth=2,
                    label=f'{model_label} ({mse_str})')
            ax.scatter(t_vals, d[norm_key], color=d['color'], s=12, zorder=3)

        ax.set_title(fn_label, fontsize=10)
        ax.set_xlabel('Time  t', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)

plt.tight_layout()
out = 'sweep_functions.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()
