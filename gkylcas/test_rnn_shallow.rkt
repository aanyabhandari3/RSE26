#lang racket

(require "code_generator_rnn_training.rkt")

;; 1D linear advection: du/dt + d(au)/dx = 0
(define pde-linear-advection
  (hash
   'name "linear-advection"
   'cons-expr `u
   'flux-expr `(* a u)
   'max-speed-expr `(abs a)
   'parameters (list `(define a 1.0))))

(define neural-net-shallow
  (hash
   'max-trains 10000
   'width 64
   'depth 6
   'num-threads 12
   'mini-size 100))

(define code
  (train-lax-friedrichs-rnn-scalar-1d pde-linear-advection neural-net-shallow
                                      #:nx 400
                                      #:x0 0.0
                                      #:x1 2.0
                                      #:t-final 0.5
                                      #:cfl 0.95
                                      #:init-func `(cond
                                                     [(< x 1.0) 1.0]
                                                     [else 0.0])))

(with-output-to-file "code/rnn_linear_advection_lax_train.c"
  #:exists 'replace
  (lambda () (display code)))

(display "Written to code/rnn_linear_advection_lax_train.c\n")
