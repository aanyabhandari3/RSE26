#lang racket

;;; code_generator_transformer_validation.rkt
;;;
;;; Generates C validation code for a pre-trained Transformer surrogate
;;; produced by code_generator_transformer_training.rkt.
;;;
;;; Unlike the PDE-based validators (code_generator_core_validation.rkt),
;;; this file does direct function evaluation: it loads the .kann file,
;;; queries it on a grid, and computes L2 and L-infinity error norms against
;;; the true target function -- no PDE time-stepping involved.
;;;
;;; Usage:
;;;
;;;   (require "code_generator_transformer_validation.rkt")
;;;
;;;   (define fn-spec
;;;     (hash
;;;      'name        "sin_decay"
;;;      'target-expr `(* (sin x) (exp (- t)))
;;;      'x-min 0.0
;;;      'x-max 6.283185307179586
;;;      't-min 0.0
;;;      't-max 2.0))
;;;
;;;   (define code (validate-transformer-fn-1d fn-spec
;;;                                            #:t-eval-max 3.0
;;;                                            #:n-x-eval 100
;;;                                            #:n-t-eval 25))
;;;   (with-output-to-file "code/transformer_sin_decay_validate.c" #:exists 'replace
;;;     (lambda () (display code)))

(provide validate-transformer-fn-1d)


;; ── Math expression → C float string ─────────────────────────────────────────
;; (same converter as in code_generator_transformer_training.rkt)
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

(define (validate-transformer-fn-1d
         fn
         #:t-eval-max [t-eval-max 3.0]
         #:n-x-eval   [n-x-eval  100]
         #:n-t-eval   [n-t-eval   25])

  "Generate C code that validates a pre-trained Transformer surrogate.

   Loads {name}_transformer.kann produced by code_generator_transformer_training.rkt,
   evaluates it on a grid of n-x-eval x n-t-eval points over
   [x-min, x-max] x [0, t-eval-max], and reports:
     - L2NORM row:  sqrt(integral |pred - exact|^2 dx) at each t step
     - LINF row:    max |pred - exact| at each t step
     - CSV files:   {name}_validation_{jt}.csv  (x, pred, exact columns)

   fn hash keys:
     name        -- string identifier matching the training run
     target-expr -- S-expression for u(x, t) using variables x and t
     x-min, x-max -- spatial domain (same as training)"

  ;; ── Extract function spec ─────────────────────────────────────────────────
  (define name        (hash-ref fn 'name))
  (define target-expr (hash-ref fn 'target-expr))
  (define x-min       (hash-ref fn 'x-min))
  (define x-max       (hash-ref fn 'x-max))

  ;; ── Convert target expression to C ───────────────────────────────────────
  (define target-c (convert-math-expr target-expr))

  ;; ── Generate C source ─────────────────────────────────────────────────────
  (format "
// AUTO-GENERATED CODE: TRANSFORMER VALIDATION  [~a]
// Target: u(x, t) = ~a
// Loads a pre-trained Transformer .kann file and evaluates it on a grid.
// Reports L2 and L-infinity error norms; writes CSV snapshots per time step.

#include \"kann.h\"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void)
{
  const double x_min     = ~a;
  const double x_max     = ~a;
  const double t_eval_max = ~a;
  const int    n_x_eval  = ~a;
  const int    n_t_eval  = ~a;
  const double dx_eval   = (x_max - x_min) / (n_x_eval - 1);

  // Load pre-trained Transformer network.
  kann_t *ann = kann_load(\"~a_transformer.kann\");
  if (!ann) {
    fprintf(stderr, \"Error: could not load ~a_transformer.kann\\n\");
    return 1;
  }

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

  // Write CSV snapshots: columns x, predicted, exact.
  for (int jt = 0; jt <= n_t_eval; jt++) {
    float t = (float)((double)jt * t_eval_max / n_t_eval);
    char fname[256];
    snprintf(fname, sizeof fname, \"~a_validation_%d.csv\", jt);
    FILE *f = fopen(fname, \"w\");
    if (f) {
      for (int ix = 0; ix < n_x_eval; ix++) {
        float x = (float)(x_min + (double)ix * (x_max - x_min) / (n_x_eval - 1));
        float exact = ~a;
        float xv[2] = {x, t};
        const float *y = kann_apply1(ann, xv);
        fprintf(f, \"%f, %f, %f\\n\", x, y[0], exact);
      }
      fclose(f);
    }
  }

  kann_delete(ann);
  return 0;
}
"
          name        ;; ~1  identifier for comment
          target-c    ;; ~2  target expression for comment
          x-min       ;; ~3  x_min
          x-max       ;; ~4  x_max
          t-eval-max  ;; ~5  t_eval_max
          n-x-eval    ;; ~6  n_x_eval
          n-t-eval    ;; ~7  n_t_eval
          name        ;; ~8  kann_load filename
          name        ;; ~9  error message filename
          target-c    ;; ~10 exact in L2 loop
          target-c    ;; ~11 exact in LINF loop
          name        ;; ~12 CSV output basename
          target-c    ;; ~13 exact in CSV loop
          ))
