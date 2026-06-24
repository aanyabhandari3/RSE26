#lang racket

;;; code_generator_rnn_training.rkt
;;;
;;; Analog of train-lax-friedrichs-scalar-1d (from code_generator_core_training.rkt)
;;; using a GRU (RNN) neural-network architecture instead of an MLP.
;;;
;;; Everything in the generated C programs is identical to the MLP variant
;;; except the neural-network construction block inside main():
;;;
;;;   MLP (original):  kann_layer_dense x depth, each followed by kad_tanh
;;;   GRU:             kann_layer_gru   x depth
;;;
;;; Validation is architecture-agnostic: validate-scalar-1d from
;;; code_generator_core_validation.rkt calls kann_load then kann_apply1 and
;;; therefore works unchanged for any trained GRU surrogate.
;;;
;;; Usage (same as the MLP counterpart):
;;;
;;;   (require "code_generator_rnn_training.rkt")
;;;
;;;   (define code
;;;     (train-lax-friedrichs-rnn-scalar-1d
;;;       pde-linear-advection neural-net-config
;;;       #:nx 400 #:t-final 0.5 ...))
;;;   (with-output-to-file "rnn_train.c" #:exists 'replace
;;;     (lambda () (display code)))

(require "code_generator_core_training.rkt")

(provide train-lax-friedrichs-rnn-scalar-1d)


;; =============================================================================
;; GRU (RNN) variant of train-lax-friedrichs-scalar-1d
;; =============================================================================
;;
;; Replaces the MLP architecture block with stacked kann_layer_gru layers.
;; Everything else — Lax-Friedrichs PDE solver, data accumulation, training
;; call, file I/O — is identical to the MLP version.

(define (train-lax-friedrichs-rnn-scalar-1d
         pde neural-net
         #:nx        [nx       200]
         #:x0        [x0       0.0]
         #:x1        [x1       2.0]
         #:t-final   [t-final  1.0]
         #:cfl       [cfl      0.95]
         #:init-func [init-func `(cond
                                   [(< x 1.0) 1.0]
                                   [else 0.0])])

  "Generate C code that trains a GRU surrogate solver for the 1D scalar PDE
   specified by `pde` using the Lax-Friedrichs finite-difference method.

   Drop-in replacement for train-lax-friedrichs-scalar-1d: the `pde` and
   `neural-net` hash shapes are identical; only the NN construction block is
   replaced (dense+tanh layers -> stacked kann_layer_gru).

   neural-net keys:
     max-trains  -- maximum number of time-steps to accumulate as training data
     width       -- GRU hidden-state dimension (cells per layer)
     depth       -- number of stacked GRU layers
     num-threads -- threads to use during training
     mini-size   -- mini-batch size"

  ;; ── Extract PDE description ───────────────────────────────────────────────
  (define name           (hash-ref pde 'name))
  (define cons-expr      (hash-ref pde 'cons-expr))
  (define flux-expr      (hash-ref pde 'flux-expr))
  (define max-speed-expr (hash-ref pde 'max-speed-expr))
  (define parameters     (hash-ref pde 'parameters))

  ;; ── Extract neural-network hyperparameters ────────────────────────────────
  (define max-trains  (hash-ref neural-net 'max-trains))
  (define width       (hash-ref neural-net 'width))
  (define depth       (hash-ref neural-net 'depth))
  (define num-threads (hash-ref neural-net 'num-threads))
  (define mini-size   (hash-ref neural-net 'mini-size))

  ;; ── Convert symbolic expressions to C strings ─────────────────────────────
  (define cons-code      (convert-expr cons-expr))
  (define flux-code      (convert-expr flux-expr))
  (define max-speed-code (convert-expr max-speed-expr))
  (define init-func-code (convert-expr init-func))

  ;; ── Build flux substitutions for Lax-Friedrichs stencil ──────────────────
  ;; f(u_{i-1}), f(u_i), f(u_{i+1}) and the local CFL wave-speed
  (define flux-um         (flux-substitute flux-code      cons-code "um"))
  (define flux-ui         (flux-substitute flux-code      cons-code "ui"))
  (define flux-up         (flux-substitute flux-code      cons-code "up"))
  (define max-speed-local (flux-substitute max-speed-code cons-code "u[i]"))

  ;; ── Optional global parameter declarations (e.g. advection speed a) ──────
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
// AUTO-GENERATED CODE FOR TRAINING ON SCALAR PDE: ~a  [GRU ARCHITECTURE]
// Train a Lax-Friedrichs first-order finite-difference surrogate solver for a
// scalar PDE in 1D using a gated recurrent unit (GRU) neural network.
//
// Architecture: input (t, x) -> kann_layer_gru x nn_depth -> scalar output
// The GRU hidden state captures temporal correlations across the input sequence.

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

  // ── Neural network hyperparameters ──────────────────────────────────────
  const int num_trains  = ~a;
  const int nn_width    = ~a;
  const int nn_depth    = ~a;
  const int num_threads = ~a;
  const int mini_size   = ~a;

  // ── Solution arrays ─────────────────────────────────────────────────────
  double *u  = (double*) malloc((nx + 2) * sizeof(double));
  double *un = (double*) malloc((nx + 2) * sizeof(double));

  // ── Training data arrays ────────────────────────────────────────────────
  float **input_data  = (float**) malloc(nx * num_trains * sizeof(float*));
  float **output_data = (float**) malloc(nx * num_trains * sizeof(float*));

  // ── Initial conditions ──────────────────────────────────────────────────
  for (int i = 0; i <= nx + 1; i++) {
    double x = x0 + (i - 0.5) * dx;
    u[i]  = ~a;
    un[i] = ~a;
  }

  // ── GRU neural network architecture ─────────────────────────────────────
  //
  // Input:         (t, x) pair  -- 2-dimensional real vector
  // Hidden layers: nn_depth stacked GRU layers, each of hidden size nn_width
  //                kann_layer_gru(h, width, 0) adds one GRU layer;
  //                the trailing 0 disables the additional output projection.
  // Output:        scalar u(t, x)  via KANN_C_MSE cost layer
  //
  kad_node_t *t_net;
  kann_t *ann;
  t_net = kann_layer_input(2);

  for (int i = 0; i < nn_depth; i++) {
    t_net = kann_layer_gru(t_net, nn_width, 0);
  }

  t_net = kann_layer_cost(t_net, 1, KANN_C_MSE);
  ann = kann_new(t_net, 0);
  // ────────────────────────────────────────────────────────────────────────

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

    // Lax-Friedrichs flux update.
    for (int i = 1; i <= nx; i++) {
      double um = u[i - 1];
      double ui = u[i];
      double up = u[i + 1];

      double f_um = ~a;
      double f_ui = ~a;
      double f_up = ~a;

      // F_{i-1/2} = 0.5*(f(u_{i-1}) + f(u_i)) - 0.5*alpha*(u_i - u_{i-1})
      double fluxL = 0.5 * (f_um + f_ui) - 0.5 * alpha * (ui - um);
      // F_{i+1/2} = 0.5*(f(u_{i+1}) + f(u_i)) - 0.5*alpha*(u_{i+1} - u_i)
      double fluxR = 0.5 * (f_ui + f_up) - 0.5 * alpha * (up - ui);

      un[i] = ui - (dt / dx) * (fluxR - fluxL);
    }

    // Copy un -> u and apply transmissive boundary conditions.
    for (int i = 0; i <= nx + 1; i++) u[i] = un[i];
    u[0] = u[1];
    u[nx + 1] = u[nx];

    // Accumulate training data: input = (t, x), output = u(t, x).
    if (n < num_trains) {
      for (int i = 1; i <= nx; i++) {
        double x = x0 + (i - 0.5) * dx;
        input_data[(n * nx) + (i - 1)]  = (float*) malloc(2 * sizeof(float));
        output_data[(n * nx) + (i - 1)] = (float*) malloc(sizeof(float));
        input_data[(n * nx) + (i - 1)][0]  = t;
        input_data[(n * nx) + (i - 1)][1]  = x;
        output_data[(n * nx) + (i - 1)][0] = u[i];
      }
    }

    // Write solution snapshot to disk.
    const char *fmt = \"%s_output_%d.csv\";
    int sz = snprintf(0, 0, fmt, \"~a\", n);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, \"~a\", n);
    FILE *fptr = fopen(file_nm, \"w\");
    if (fptr != NULL) {
      for (int i = 1; i <= nx; i++) {
        double x = x0 + (i - 0.5) * dx;
        fprintf(fptr, \"%f, %f\\n\", x, u[i]);
      }
      fclose(fptr);
    }

    t += dt;
    n += 1;
  }

  // ── Train the GRU network on accumulated PDE solution data ───────────────
  kann_mt(ann, num_threads, mini_size);
  kann_train_fnn1(ann, 0.0001f, 64, 50, 10, 0.1f, n * nx, input_data, output_data);

  // ── Save trained network to disk ─────────────────────────────────────────
  const char *fmt2 = \"%s_neural_net.dat\";
  int sz2 = snprintf(0, 0, fmt2, \"~a\");
  char file_nm2[sz2 + 1];
  snprintf(file_nm2, sizeof file_nm2, fmt2, \"~a\");
  kann_save(file_nm2, ann);

  free(u);
  free(un);
  kann_delete(ann);
  for (int i = 0; i < n * nx; i++) {
    free(input_data[i]);
    free(output_data[i]);
  }
  free(input_data);
  free(output_data);
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
           max-trains      ;; ~8  num_trains: max training-data time-steps
           width           ;; ~9  nn_width: GRU hidden size
           depth           ;; ~10 nn_depth: number of stacked GRU layers
           num-threads     ;; ~11 parallelism for kann_mt
           mini-size       ;; ~12 mini-batch size
           init-func-code  ;; ~13 initial condition expression for u[i]
           init-func-code  ;; ~14 initial condition expression for un[i]
           max-speed-local ;; ~15 local wave-speed for CFL dt computation
           flux-um         ;; ~16 f(u_{i-1})
           flux-ui         ;; ~17 f(u_i)
           flux-up         ;; ~18 f(u_{i+1})
           name            ;; ~19 CSV output file basename
           name            ;; ~20 CSV output snprintf argument
           name            ;; ~21 .kann save file basename
           name            ;; ~22 .kann save snprintf argument
           ))
  code)
