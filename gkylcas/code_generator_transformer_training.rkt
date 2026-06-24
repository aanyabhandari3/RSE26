#lang racket

;;; code_generator_transformer_training.rkt
;;;
;;; Generates C training code that approximates a target function u(x, t) using
;;; the Transformer architecture from transformer_optim_ab.c in the KANN directory:
;;;
;;;   input (x, t) -> dense(d_model) -> tanh
;;;                -> transformer_block x nn_depth
;;;                -> scalar output (KANN_C_MSE)
;;;
;;; The transformer_block helper (feature-wise self-attention + FFN with residual
;;; connections and layer norm) is embedded directly in the generated C file.
;;;
;;; Unlike the PDE-based generators (code_generator_core_training.rkt), this file
;;; does direct function approximation: it samples u(x, t) on a regular NX x NT
;;; grid and trains the network to fit those samples.
;;;
;;; Usage:
;;;
;;;   (require "code_generator_transformer_training.rkt")
;;;
;;;   (define fn-spec
;;;     (hash
;;;      'name        "sin_decay"
;;;      'target-expr `(* (sin x) (exp (- t)))
;;;      'x-min 0.0
;;;      'x-max 6.283185307179586   ; 2*pi
;;;      't-min 0.0
;;;      't-max 2.0))
;;;
;;;   (define neural-net
;;;     (hash 'nx 50 'nt 50 'width 64 'depth 1 'num-threads 4 'mini-size 64))
;;;
;;;   (define code (train-transformer-fn-1d fn-spec neural-net))
;;;   (with-output-to-file "code/transformer_sin_decay_train.c" #:exists 'replace
;;;     (lambda () (display code)))

(provide train-transformer-fn-1d)


;; ── Math expression → C float string ─────────────────────────────────────────
;;
;; Converts a Racket S-expression (using variables x and t) to a C string that
;; uses single-precision float math functions: sinf, cosf, expf, etc.
;;
;; Supported operators: + - * / (binary and unary -)
;; Supported functions: sin cos tan asin acos atan sinh cosh tanh
;;                      exp log sqrt abs expt/pow
;; Special symbol:      pi -> (float)M_PI
(define (convert-math-expr expr)
  (cond
    [(number? expr)
     (number->string (exact->inexact expr))]
    [(symbol? expr)
     (case expr
       [(pi)  "(float)M_PI"]
       [else  (symbol->string expr)])]
    [(pair? expr)
     (case (car expr)
       [(+)
        (string-append "(" (convert-math-expr (cadr expr))
                       " + " (convert-math-expr (caddr expr)) ")")]
       [(-)
        (if (null? (cddr expr))
            (string-append "(-" (convert-math-expr (cadr expr)) ")")
            (string-append "(" (convert-math-expr (cadr expr))
                           " - " (convert-math-expr (caddr expr)) ")"))]
       [(*)
        (string-append "(" (convert-math-expr (cadr expr))
                       " * " (convert-math-expr (caddr expr)) ")")]
       [(/)
        (string-append "(" (convert-math-expr (cadr expr))
                       " / " (convert-math-expr (caddr expr)) ")")]
       [(sin)  (string-append "sinf("  (convert-math-expr (cadr expr)) ")")]
       [(cos)  (string-append "cosf("  (convert-math-expr (cadr expr)) ")")]
       [(tan)  (string-append "tanf("  (convert-math-expr (cadr expr)) ")")]
       [(asin) (string-append "asinf(" (convert-math-expr (cadr expr)) ")")]
       [(acos) (string-append "acosf(" (convert-math-expr (cadr expr)) ")")]
       [(atan) (string-append "atanf(" (convert-math-expr (cadr expr)) ")")]
       [(sinh) (string-append "sinhf(" (convert-math-expr (cadr expr)) ")")]
       [(cosh) (string-append "coshf(" (convert-math-expr (cadr expr)) ")")]
       [(tanh) (string-append "tanhf(" (convert-math-expr (cadr expr)) ")")]
       [(exp)  (string-append "expf("  (convert-math-expr (cadr expr)) ")")]
       [(log)  (string-append "logf("  (convert-math-expr (cadr expr)) ")")]
       [(sqrt) (string-append "sqrtf(" (convert-math-expr (cadr expr)) ")")]
       [(abs)  (string-append "fabsf(" (convert-math-expr (cadr expr)) ")")]
       [(expt pow)
        (string-append "powf(" (convert-math-expr (cadr expr))
                       ", " (convert-math-expr (caddr expr)) ")")]
       [else
        (error (format "convert-math-expr: unknown operator '~a'" (car expr)))])]))


;; ── Main code generator ────────────────────────────────────────────────────────

(define (train-transformer-fn-1d
         fn neural-net
         #:t-eval-max [t-eval-max 3.0]
         #:n-x-eval   [n-x-eval  100]
         #:n-t-eval   [n-t-eval   25])

  "Generate C code that trains a Transformer to approximate u(x, t).

   Mirrors transformer_optim_ab.c: samples a regular NX x NT grid over
   [x-min, x-max] x [t-min, t-max], trains with kann_train_fnn1, then
   prints validation MSE and L2/LINF rollout norms over [0, t-eval-max].

   fn hash keys:
     name        -- string identifier, used in .kann filename
     target-expr -- S-expression for u(x, t) using variables x and t
     x-min, x-max -- spatial domain
     t-min, t-max -- training time domain

   neural-net hash keys:
     nx, nt      -- training grid dimensions
     width       -- d_model (embedding / attention dimension)
     depth       -- number of transformer blocks
     num-threads -- parallelism for kann_mt
     mini-size   -- mini-batch size"

  ;; ── Extract function spec ─────────────────────────────────────────────────
  (define name        (hash-ref fn 'name))
  (define target-expr (hash-ref fn 'target-expr))
  (define x-min       (hash-ref fn 'x-min))
  (define x-max       (hash-ref fn 'x-max))
  (define t-min       (hash-ref fn 't-min))
  (define t-max       (hash-ref fn 't-max))

  ;; ── Extract neural-net hyperparameters ───────────────────────────────────
  (define nx          (hash-ref neural-net 'nx))
  (define nt          (hash-ref neural-net 'nt))
  (define width       (hash-ref neural-net 'width))
  (define depth       (hash-ref neural-net 'depth))
  (define num-threads (hash-ref neural-net 'num-threads))
  (define mini-size   (hash-ref neural-net 'mini-size))

  ;; ── Convert target expression to C ───────────────────────────────────────
  (define target-c (convert-math-expr target-expr))

  ;; ── Generate C source ─────────────────────────────────────────────────────
  (format "
// AUTO-GENERATED CODE: TRANSFORMER FUNCTION APPROXIMATION  [~a]
// Target: u(x, t) = ~a
// Architecture: input(2) -> dense(nn_width) -> tanh -> transformer_block x nn_depth -> MSE

#include \"kann.h\"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Transformer encoder block ─────────────────────────────────────────────────
//
// Self-attention sublayer (feature-wise Q*K product -> softmax -> gate V)
// followed by a feedforward sublayer, each with residual connection + layer norm.
//
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

int main(void)
{
  int i, k;
  const int    NX          = ~a;
  const int    NT          = ~a;
  const int    nn_width    = ~a;
  const int    nn_depth    = ~a;
  const int    num_threads = ~a;
  const int    mini_size   = ~a;
  const double x_min       = ~a;
  const double x_max       = ~a;
  const double t_min       = ~a;
  const double t_max       = ~a;

  int n = NX * NT;
  float **input  = (float **)malloc(n * sizeof(float *));
  float **output = (float **)malloc(n * sizeof(float *));

  // Sample target function on NX x NT grid.
  k = 0;
  for (i = 0; i < NX; i++) {
    int j;
    for (j = 0; j < NT; j++) {
      input[k]  = (float *)malloc(2 * sizeof(float));
      output[k] = (float *)malloc(sizeof(float));
      float x = (float)(x_min + (double)i * (x_max - x_min) / (NX - 1));
      float t = (float)(t_min + (double)j * (t_max - t_min) / (NT - 1));
      input[k][0]  = x;
      input[k][1]  = t;
      output[k][0] = ~a;
      k++;
    }
  }

  // ── Transformer architecture ──────────────────────────────────────────────
  // input(2) -> dense(nn_width) -> tanh -> transformer_block x nn_depth -> MSE
  kad_node_t *t_net = kann_layer_input(2);
  t_net = kann_layer_dense(t_net, nn_width);
  t_net = kad_tanh(t_net);
  for (i = 0; i < nn_depth; i++)
    t_net = transformer_block(t_net, nn_width, nn_width * 2);
  t_net = kann_layer_cost(t_net, 1, KANN_C_MSE);
  kann_t *ann = kann_new(t_net, 0);
  // ─────────────────────────────────────────────────────────────────────────

  kann_mt(ann, num_threads, mini_size);
  kann_train_fnn1(ann, 0.001f, 64, 50, 10, 0.1f, n, input, output);

  kann_save(\"~a_transformer.kann\", ann);
  kann_delete(ann);
  ann = kann_load(\"~a_transformer.kann\");

  float val_cost = kann_cost_fnn1(ann, n, input, output);
  printf(\"Validation MSE: %.6f\\n\", val_cost);
  printf(\"(NX=%d, NT=%d, WIDTH=%d, DEPTH=%d)\\n\", NX, NT, nn_width, nn_depth);

  // ── L2 and L-infinity rollout norms ───────────────────────────────────────
  const int    n_t_eval   = ~a;
  const int    n_x_eval   = ~a;
  const double t_eval_max = ~a;
  const double dx_eval    = (x_max - x_min) / (n_x_eval - 1);

  printf(\"L2NORM\");
  for (int jt = 0; jt <= n_t_eval; jt++) {
    float t = (float)((double)jt * t_eval_max / n_t_eval);
    float l2 = 0.0f;
    for (int ix = 0; ix < n_x_eval; ix++) {
      float x = (float)(x_min + (double)ix * (x_max - x_min) / (n_x_eval - 1));
      float exact = ~a;
      float xv[2] = {x, t};
      const float *y = kann_apply1(ann, xv);
      float err = y[0] - exact;
      l2 += err * err * (float)dx_eval;
    }
    printf(\" %.6f\", sqrtf(l2));
  }
  printf(\"\\n\");

  printf(\"LINF\");
  for (int jt = 0; jt <= n_t_eval; jt++) {
    float t = (float)((double)jt * t_eval_max / n_t_eval);
    float linf = 0.0f;
    for (int ix = 0; ix < n_x_eval; ix++) {
      float x = (float)(x_min + (double)ix * (x_max - x_min) / (n_x_eval - 1));
      float exact = ~a;
      float xv[2] = {x, t};
      const float *y = kann_apply1(ann, xv);
      float err = fabsf(y[0] - exact);
      if (err > linf) linf = err;
    }
    printf(\" %.6f\", linf);
  }
  printf(\"\\n\");

  for (i = 0; i < n; i++) { free(input[i]); free(output[i]); }
  free(input);
  free(output);
  kann_delete(ann);
  return 0;
}
"
          name          ;; ~1  comment header identifier
          target-c      ;; ~2  target expression for comment
          nx            ;; ~3  NX
          nt            ;; ~4  NT
          width         ;; ~5  nn_width
          depth         ;; ~6  nn_depth
          num-threads   ;; ~7  num_threads
          mini-size     ;; ~8  mini_size
          x-min         ;; ~9  x_min
          x-max         ;; ~10 x_max
          t-min         ;; ~11 t_min
          t-max         ;; ~12 t_max
          target-c      ;; ~13 output[k][0] = ...
          name          ;; ~14 kann_save filename
          name          ;; ~15 kann_load filename
          n-t-eval      ;; ~16 n_t_eval
          n-x-eval      ;; ~17 n_x_eval
          t-eval-max    ;; ~18 t_eval_max
          target-c      ;; ~19 exact in L2 rollout
          target-c      ;; ~20 exact in LINF rollout
          ))
