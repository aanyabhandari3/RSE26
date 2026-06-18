#!/usr/bin/env python3
"""
sweep_width.py
Sweeps NX/NT (volume), WIDTH, and DEPTH for both MLP and GRU.
All other parameters are held at their defaults when one is varied.
Produces a figure with one subplot per parameter, MLP vs GRU side by side.

Defaults: NX=50, NT=50, WIDTH=256, DEPTH=1
Runs:     3 sweeps x 3 values x 2 models = 18 total
"""

import subprocess
import re
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

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
FN_LABEL        = _fn_raw
FN_COMPILE_FLAG = [f'-DTARGET_FN(x,t)=({_fn_c})']
FN_ARG          = re.sub(r'[^\w]', '_', _fn_raw)[:30].strip('_') or 'custom'
# ─────────────────────────────────────────────────────────────────────────────

# ── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    'MLP':         'simple_mlp_ab.c',
    'GRU':         'simple_rnn_ab.c',
    'Transformer': 'simple_transformer_ab.c',
}
SHARED = ['kautodiff.c', 'kann.c']
COLORS = ['steelblue', 'coral', 'mediumseagreen']

# ── Sweeps ───────────────────────────────────────────────────────────────────
# For each sweep, 'flags' returns the -D flags to pass for a given value.
# The other parameters are left at the defaults already in the source files.
SWEEPS = [
    {
        'title':  'Training Volume  (NX = NT)',
        'xlabel': 'NX = NT',
        'values': [20, 50, 100],
        'flags':  lambda v: [f'-DNX={v}', f'-DNT={v}'],
    },
    {
        'title':  'Hidden Width',
        'xlabel': 'Width (neurons per layer)',
        'values': [64, 256, 512],
        'flags':  lambda v: [f'-DWIDTH={v}'],
    },
    {
        'title':  'Depth  (hidden layers)',
        'xlabel': 'Depth',
        'values': [1, 2, 4],
        'flags':  lambda v: [f'-DDEPTH={v}'],
    },
]

MSE_RE = re.compile(r'Validation MSE:\s*([\d.eE+\-]+)')

# ── Runner ───────────────────────────────────────────────────────────────────
def run_one(src, extra_flags, tag):
    """Compile src with extra_flags, run, return MSE float."""
    binary = f'./sweep_tmp_{tag}'
    cmd = ['gcc', '-Wno-macro-redefined'] + FN_COMPILE_FLAG + extra_flags + [src] + SHARED + ['-lm', '-o', binary]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(f'\nCompile error ({tag}):\n{cp.stderr}')
        sys.exit(1)
    rp = subprocess.run([binary], capture_output=True, text=True)
    os.remove(binary)
    m = MSE_RE.search(rp.stdout)
    if not m:
        print(f'\nNo "Validation MSE:" found in output ({tag}):\n{rp.stdout}')
        sys.exit(1)
    return float(m.group(1))

# ── Collect results ───────────────────────────────────────────────────────────
# results[sweep_idx][model_label] = [mse_val0, mse_val1, mse_val2]
all_results = []

for sweep in SWEEPS:
    sweep_results = {label: [] for label in MODELS}
    print(f'\n── {sweep["title"]} ──')
    for v in sweep['values']:
        flags = sweep['flags'](v)
        for label, src in MODELS.items():
            tag = f'{label.split()[0].lower()}_{v}'
            print(f'  {label:20s}  {sweep["xlabel"]}={v} ...', end='  ', flush=True)
            mse = run_one(src, flags, tag)
            print(f'MSE = {mse:.6f}')
            sweep_results[label].append(mse)
    all_results.append(sweep_results)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f'MLP vs GRU vs Transformer — Validation MSE\n'
    f'u(x,t) = {FN_LABEL}  |  defaults: NX=NT=50, WIDTH=256, DEPTH=1, LR=0.001, epochs=50',
    fontsize=12
)

bw = 0.25   # narrower so 3 bars fit side by side without overlap
labels = list(MODELS.keys())
n_models = len(labels)

for ax, sweep, sweep_results in zip(axes, SWEEPS, all_results):
    x = np.arange(len(sweep['values']))
    offsets = np.linspace(-(n_models - 1) * bw / 2, (n_models - 1) * bw / 2, n_models)

    for label, color, offset in zip(labels, COLORS, offsets):
        mse_vals = sweep_results[label]
        bars = ax.bar(x + offset, mse_vals, bw, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.02,
                f'{h:.4f}',
                ha='center', va='bottom', fontsize=7
            )

    ax.set_title(sweep['title'], fontsize=11)
    ax.set_xlabel(sweep['xlabel'], fontsize=10)
    ax.set_ylabel('Validation MSE  (lower = better)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in sweep['values']])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

plt.tight_layout()
out = f'sweep_width_{FN_ARG}.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
plt.show()