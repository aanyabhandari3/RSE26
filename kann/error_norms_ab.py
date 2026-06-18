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

# ── Function selection ────────────────────────────────────────────────────────
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

def _ask_function():
    """Prompt for a function expression; accept sys.argv[1] as shortcut."""
    if len(sys.argv) > 1:
        raw = ' '.join(sys.argv[1:]).strip()
        c_expr = _to_c_float(raw)
        print(f'Function: {raw}  →  C: {c_expr}')
        return raw, c_expr
    print('\nEnter the target function  u(x, t)  using variables x and t.')
    print('Math functions auto-convert to C floats: sin→sinf, exp→expf, etc.')
    print('Examples:')
    print('  sin(x)*exp(-t)')
    print('  cos(2*x)*exp(-t/2)')
    print('  tanh(x)*(1 - t/3)')
    print('  pow(x, 2)*exp(-t)')
    print()
    while True:
        raw = input('u(x,t) = ').strip()
        if raw:
            c_expr = _to_c_float(raw)
            print(f'→ C: {c_expr}\n')
            return raw, c_expr
        print('  Please enter a function expression.')

_fn_raw, _fn_c = _ask_function()
FN_LABEL      = _fn_raw
COMPILE_FLAGS = [f'-DTARGET_FN(x,t)=({_fn_c})']
FN_ARG        = re.sub(r'[^\w]', '_', _fn_raw)[:30].strip('_') or 'custom'
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'MLP':         ('simple_mlp_ab.c',          'steelblue'),
    'GRU':         ('simple_rnn_ab.c',           'coral'),
    'Transformer': ('simple_transformer_ab.c',   'mediumseagreen'),
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
        ['gcc'] + COMPILE_FLAGS + [src] + SHARED + ['-lm', '-o', binary],
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
    f'MLP vs GRU vs Transformer — Spatial Error Norms over Time\n'
    f'u(x,t) = {FN_LABEL}  |  trained on t ∈ [0, 2],  evaluated on t ∈ [0, 3]',
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
out = f'error_norms_{FN_ARG}.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()