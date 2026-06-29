#lang racket

;;; code_generator_rnn_validation.rkt
;;;
;;; Analog of validate-scalar-1d (from code_generator_core_validation.rkt)
;;; for validating a pre-trained GRU surrogate solver.
;;;
;;; The generated C is architecture-agnostic: it loads any .kann file produced
;;; by code_generator_rnn_training.rkt via kann_load and steps forward in time
;;; using kann_apply1 — no knowledge of the internal network structure needed.
;;;
;;; Usage (same as the MLP counterpart):
;;;
;;;   (require "code_generator_rnn_validation.rkt")
;;;
;;;   (define code
;;;     (validate-rnn-scalar-1d pde-linear-advection neural-net-shallow
;;;                             #:nx 400 #:t-final 0.5 ...))
;;;   (with-output-to-file "code/rnn_validate.c" #:exists 'replace
;;;     (lambda () (display code)))

(require "code_generator_core_training.rkt")

(provide validate-rnn-scalar-1d)


(define (validate-rnn-scalar-1d
         pde neural-net
         #:nx        [nx       200]
         #:x0        [x0       0.0]
         #:x1        [x1       2.0]
         #:t-final   [t-final  1.0]
         #:cfl       [cfl      0.95]
         #:init-func [init-func `(cond
                                   [(< x 1.0) 1.0]
                                   [else 0.0])])

  "Generate C code that validates a pre-trained GRU surrogate solver for the
   1D scalar PDE specified by `pde`.

   Drop-in replacement for validate-scalar-1d: identical `pde` and `neural-net`
   hash shapes, identical generated C (architecture is not reconstructed --
   the .kann file is loaded directly with kann_load).

   At each time step the network is queried via kann_apply1(ann, [t, x]) for
   every cell, replacing the PDE update with a direct neural-network prediction.
   The result is written to {name}_validation_{n}.csv at every step.

   pde hash keys:
     name, cons-expr, max-speed-expr, parameters
     (flux-expr is not needed for validation)

   neural-net hash keys: same shape as training -- only used for documentation"

  ;; ── Extract PDE description ───────────────────────────────────────────────
  (define name           (hash-ref pde 'name))
  (define cons-expr      (hash-ref pde 'cons-expr))
  (define max-speed-expr (hash-ref pde 'max-speed-expr))
  (define parameters     (hash-ref pde 'parameters))

  ;; ── Convert expressions to C strings ─────────────────────────────────────
  (define cons-code      (convert-expr cons-expr))
  (define max-speed-code (convert-expr max-speed-expr))
  (define init-func-code (convert-expr init-func))

  (define max-speed-local (flux-substitute max-speed-code cons-code "u[i]"))

  (define parameter-code
    (if (empty? parameters)
        ""
        (string-join (map (lambda (p)
                            (string-append "double " (convert-expr p) ";"))
                          parameters)
                     "\n")))

  ;; ── Generate C source ─────────────────────────────────────────────────────
  (define code
    (format "
// AUTO-GENERATED CODE FOR VALIDATING ON SCALAR PDE: ~a  [GRU ARCHITECTURE]
// Validate a pre-trained GRU surrogate solver for a scalar PDE in 1D.
// Loads {name}_neural_net.dat produced by code_generator_rnn_training.rkt
// and steps forward using kann_apply1 -- no network reconstruction needed.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include \"kann.h\"

// Additional PDE parameters (if any).
~a

int main() {
  // ── Spatial domain ──────────────────────────────────────────────────────
  const int nx     = ~a;
  const double x0  = ~a;
  const double x1  = ~a;
  const double L   = (x1 - x0);
  const double dx  = L / nx;

  // ── Time-stepper ────────────────────────────────────────────────────────
  const double cfl     = ~a;
  const double t_final = ~a;

  // ── Solution array ───────────────────────────────────────────────────────
  double *u = (double*) malloc((nx + 2) * sizeof(double));

  // ── Initial conditions ──────────────────────────────────────────────────
  for (int i = 0; i <= nx + 1; i++) {
    double x = x0 + (i - 0.5) * dx;
    u[i] = ~a;
  }

  // ── Load pre-trained GRU network ─────────────────────────────────────────
  kann_t *ann;
  const char *fmt = \"%s_neural_net.dat\";
  int sz = snprintf(0, 0, fmt, \"~a\");
  char file_nm[sz + 1];
  snprintf(file_nm, sizeof file_nm, fmt, \"~a\");

  FILE *fptr = fopen(file_nm, \"r\");
  if (fptr != NULL) {
    ann = kann_load(file_nm);
    fclose(fptr);
  }

  double t = 0.0;
  int n = 0;
  while (t < t_final) {
    // Global maximum wave-speed for CFL stability.
    double alpha = 0.0;
    for (int i = 1; i <= nx; i++) {
      double local_alpha = ~a;
      if (local_alpha > alpha) alpha = local_alpha;
    }
    if (alpha < 1e-14) alpha = 1e-14;

    double dt = cfl * dx / alpha;
    if (t + dt > t_final) dt = t_final - t;

    // Query GRU network for each cell: input = (t, x), output = u(t, x).
    for (int i = 1; i <= nx; i++) {
      double x = x0 + (i - 0.5) * dx;
      float input_data[2];
      input_data[0] = (float)t;
      input_data[1] = (float)x;
      const float *output_data = kann_apply1(ann, input_data);
      u[i] = output_data[0];
    }

    // Transmissive boundary conditions.
    u[0] = u[1];
    u[nx + 1] = u[nx];

    // Write solution snapshot to disk.
    const char *fmt2 = \"%s_validation_%d.csv\";
    int sz2 = snprintf(0, 0, fmt2, \"~a\", n);
    char file_nm2[sz2 + 1];
    snprintf(file_nm2, sizeof file_nm2, fmt2, \"~a\", n);
    FILE *fptr2 = fopen(file_nm2, \"w\");
    if (fptr2 != NULL) {
      for (int i = 1; i <= nx; i++) {
        double x = x0 + (i - 0.5) * dx;
        fprintf(fptr2, \"%f, %f\\n\", x, u[i]);
      }
      fclose(fptr2);
    }

    t += dt;
    n += 1;
  }

  free(u);
  kann_delete(ann);
  return 0;
}
"
           name            ;; ~1  PDE name (comment header)
           parameter-code  ;; ~2  global parameter declarations
           nx              ;; ~3  spatial cell count
           x0              ;; ~4  left domain boundary
           x1              ;; ~5  right domain boundary
           cfl             ;; ~6  CFL coefficient
           t-final         ;; ~7  final simulation time
           init-func-code  ;; ~8  initial condition
           name            ;; ~9  kann_load basename
           name            ;; ~10 kann_load snprintf argument
           max-speed-local ;; ~11 local wave-speed for CFL dt
           name            ;; ~12 CSV output basename
           name            ;; ~13 CSV output snprintf argument
           ))
  code)
