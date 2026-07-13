#lang racket

;;; code_generator_transformer_pde_training.rkt
;;;
;;; Analog of train-lax-friedrichs-scalar-1d (from code_generator_core_training.rkt)
;;; using a Transformer neural-network architecture instead of an MLP.
;;;
;;; Identical to code_generator_rnn_training.rkt in structure -- the only
;;; difference is the NN construction block:
;;;
;;;   RNN:         kann_layer_gru x depth
;;;   Transformer: dense(width) -> tanh  [embedding]
;;;                -> transformer_block  x depth
;;;
;;; Usage (same as the MLP and RNN counterparts):
;;;
;;;   (require "code_generator_transformer_pde_training.rkt")
;;;
;;;   (define code
;;;     (train-lax-friedrichs-transformer-scalar-1d
;;;       pde-linear-advection neural-net-config
;;;       #:nx 400 #:t-final 0.5 ...))
;;;   (with-output-to-file "code/transformer_train.c" #:exists 'replace
;;;     (lambda () (display code)))

(require "code_generator_core_training.rkt")

(provide train-lax-friedrichs-transformer-scalar-1d)


(define (train-lax-friedrichs-transformer-scalar-1d
         pde neural-net
         #:nx        [nx       200]
         #:x0        [x0       0.0]
         #:x1        [x1       2.0]
         #:t-final   [t-final  1.0]
         #:cfl       [cfl      0.95]
         #:init-func [init-func `(cond
                                   [(< x 1.0) 1.0]
                                   [else 0.0])])

  "Generate C code that trains a Transformer surrogate solver for the 1D scalar
   PDE specified by `pde` using the Lax-Friedrichs finite-difference method.

   Drop-in replacement for train-lax-friedrichs-scalar-1d and
   train-lax-friedrichs-rnn-scalar-1d: identical `pde` and `neural-net` hash
   shapes, identical PDE numerics. Only the NN block differs:
     input(2) -> dense(width) -> tanh -> transformer_block x depth -> KANN_C_MSE

   neural-net keys:
     max-trains  -- maximum number of time-steps to accumulate as training data
     width       -- d_model (embedding / attention dimension)
     depth       -- number of stacked transformer blocks
     num-threads -- threads to use during training
     mini-size   -- mini-batch size"

  ;; ── Extract PDE description ───────────────────────────────────────────────
  (define name           (hash-ref pde 'name))
  (define cons-expr      (hash-ref pde 'cons-expr))
  (define flux-expr      (hash-ref pde 'flux-expr))
  (define max-speed-expr (hash-ref pde 'max-speed-expr))
  (define parameters     (hash-ref pde 'parameters))

  ;; ── Extract neural-net hyperparameters ───────────────────────────────────
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
  (define flux-um         (flux-substitute flux-code      cons-code "um"))
  (define flux-ui         (flux-substitute flux-code      cons-code "ui"))
  (define flux-up         (flux-substitute flux-code      cons-code "up"))
  (define max-speed-local (flux-substitute max-speed-code cons-code "u[i]"))

  ;; ── Optional global parameter declarations ────────────────────────────────
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
// AUTO-GENERATED CODE FOR TRAINING ON SCALAR PDE: ~a  [TRANSFORMER ARCHITECTURE]
// Train a Lax-Friedrichs first-order finite-difference surrogate solver for a
// scalar PDE in 1D using a Transformer neural network.
//
// Architecture: input (t, x) -> dense(d_model) -> tanh
//                            -> transformer_block x nn_depth
//                            -> scalar output (KANN_C_MSE)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include \"kann.h\"

// Additional PDE parameters (if any).
~a

// ── Transformer encoder block ─────────────────────────────────────────────────
static kad_node_t *transformer_block(kad_node_t *h, int d_model, int d_ff)
{
  kad_node_t *W_Q = kann_new_weight(d_model, d_model);
  kad_node_t *W_K = kann_new_weight(d_model, d_model);
  kad_node_t *W_V = kann_new_weight(d_model, d_model);
  kad_node_t *W_O = kann_new_weight(d_model, d_model);

  kad_node_t *Q     = kad_cmul(h, W_Q);
  kad_node_t *K     = kad_cmul(h, W_K);
  kad_node_t *V     = kad_cmul(h, W_V);
  kad_node_t *score = kad_mul(Q, K);
  kad_node_t *attn  = kad_softmax(score);
  kad_node_t *out   = kad_mul(attn, V);

  out = kad_cmul(out, W_O);
  h   = kad_stdnorm(kad_add(h, out));

  kad_node_t *ff = kann_layer_dense(h, d_ff);
  ff = kad_tanh(ff);
  ff = kann_layer_dense(ff, d_model);
  h  = kad_stdnorm(kad_add(h, ff));

  return h;
}
// ─────────────────────────────────────────────────────────────────────────────

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

  // ── Transformer architecture ──────────────────────────────────────────────
  //
  // Input embedding: R^2 -> R^{nn_width}  (dense + tanh)
  // Stacked transformer blocks: nn_depth x transformer_block
  // Output: scalar u(t, x) via KANN_C_MSE
  //
  kad_node_t *t_net;
  kann_t *ann;
  t_net = kann_layer_input(2);
  t_net = kann_layer_dense(t_net, nn_width);
  t_net = kad_tanh(t_net);
  for (int i = 0; i < nn_depth; i++)
    t_net = transformer_block(t_net, nn_width, nn_width * 2);
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

      double fluxL = 0.5 * (f_um + f_ui) - 0.5 * alpha * (ui - um);
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

  // ── Train the Transformer on accumulated PDE solution data ───────────────
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
           max-trains      ;; ~8  num_trains
           width           ;; ~9  nn_width
           depth           ;; ~10 nn_depth
           num-threads     ;; ~11 parallelism
           mini-size       ;; ~12 mini-batch size
           init-func-code  ;; ~13 initial condition for u[i]
           init-func-code  ;; ~14 initial condition for un[i]
           max-speed-local ;; ~15 local wave-speed for CFL dt
           flux-um         ;; ~16 f(u_{i-1})
           flux-ui         ;; ~17 f(u_i)
           flux-up         ;; ~18 f(u_{i+1})
           name            ;; ~19 CSV output basename
           name            ;; ~20 CSV output snprintf argument
           name            ;; ~21 .kann save basename
           name            ;; ~22 .kann save snprintf argument
           ))
  code)
