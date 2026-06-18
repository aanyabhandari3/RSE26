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
N_T_EVAL = 25          # must match the value in the C files
T_MAX    = 3.0
T_TRAIN  = 2.0         # training range boundary

L2_RE  = re.compile(r'^L2NORM\s+([\d.\s]+)', re.MULTILINE)
MSE_RE = re.compile(r'Validation MSE:\s*([\d.eE+\-]+)')

# ── Compile and run each model ────────────────────────────────────────────────
rollout_data = {}   # { label: np.array of shape (N_T_EVAL+1,) }
overall_mse  = {}   # { label: float }

for label, (src, color) in MODELS.items():
    binary = f'./rollout_tmp_{label.split()[0].lower()}'
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

    # overall validation MSE
    m = MSE_RE.search(rp.stdout)
    if m:
        overall_mse[label] = float(m.group(1))

    # L2 norm per time slice (replaces old ROLLOUT/MSE line)
    m = L2_RE.search(rp.stdout)
    if not m:
        print(f'No L2NORM line found in output:\n{rp.stdout}')
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
ax.set_ylabel('L2 norm  √(∫|error|² dx)  per time slice', fontsize=11)
ax.set_title(
    f'Rollout Divergence — MLP vs GRU vs Transformer\n'
    f'u(x,t) = {FN_LABEL}  |  trained on t ∈ [0, 2],  evaluated on t ∈ [0, 3]',
    fontsize=12
)
ax.legend(fontsize=10)
ax.set_xlim(0, T_MAX)
ax.set_ylim(bottom=0)

plt.tight_layout()
out = f'rollout_divergence_{FN_ARG}.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()