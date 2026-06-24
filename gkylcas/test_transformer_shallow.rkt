#lang racket

(require "code_generator_transformer_training.rkt")

;; Target function: u(x, t) = sin(x) * exp(-t)
;; x in [0, 2*pi],  t in [0, 2]  (training)
(define fn-sin-decay
  (hash
   'name        "sin_decay"
   'target-expr `(* (sin x) (exp (- t)))
   'x-min 0.0
   'x-max 6.283185307179586   ; 2*pi
   't-min 0.0
   't-max 2.0))

(define neural-net-shallow
  (hash
   'nx 50            ; x grid points
   'nt 50            ; t grid points
   'width 64         ; d_model
   'depth 1          ; transformer blocks
   'num-threads 4
   'mini-size 64))

(define code
  (train-transformer-fn-1d fn-sin-decay neural-net-shallow
                           #:t-eval-max 3.0
                           #:n-x-eval   100
                           #:n-t-eval   25))

(with-output-to-file "code/transformer_sin_decay_train.c"
  #:exists 'replace
  (lambda () (display code)))

(display "Written to code/transformer_sin_decay_train.c\n")
