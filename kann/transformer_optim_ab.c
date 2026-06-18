#include "kann.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =====================================================================
   VARIATION 1: Training data volume  (default: 50x50)
   ===================================================================== */
#ifndef NX
#define NX 50
#endif
#ifndef NT
#define NT 50
#endif

/* =====================================================================
   VARIATION 2: Model dimension d_model  (default: 64)
   ===================================================================== */
#ifndef WIDTH
#define WIDTH 64
#endif

/* =====================================================================
   VARIATION 3: Number of transformer blocks  (default: 1)
   ===================================================================== */
#ifndef DEPTH
#define DEPTH 1
#endif

/* =====================================================================
   VARIATION 4: Target function  (default: cos)
   Pass one of these flags to transformer_optim_ab.py automatically:
     -DFUNC_SIN          sin(x)*exp(-t)         standard sine decay
     -DFUNC_COS          cos(x)*exp(-t)         standard cosine decay
     -DFUNC_TANH         tanh(x)*exp(-t)        tanh decay
     -DFUNC_FAST_DECAY   sin(x)*exp(-2t)        steeper time drop
     -DFUNC_SLOW_DECAY   sin(x)*exp(-t/2)       gentler time drop
     -DFUNC_SIN2X        sin(2x)*exp(-t)        higher spatial frequency
     -DFUNC_GAUSS        exp(-x²/π²)*exp(-t)    Gaussian in space
   ===================================================================== */
#if defined(FUNC_SIN)
  #define TARGET_FN(x, t) (sinf(x) * expf(-(t)))
  #define FN_LABEL "sin(x)*exp(-t)"
#elif defined(FUNC_TANH)
  #define TARGET_FN(x, t) (tanhf(x) * expf(-(t)))
  #define FN_LABEL "tanh(x)*exp(-t)"
#elif defined(FUNC_FAST_DECAY)
  #define TARGET_FN(x, t) (sinf(x) * expf(-2.0f*(t)))
  #define FN_LABEL "sin(x)*exp(-2t)"
#elif defined(FUNC_SLOW_DECAY)
  #define TARGET_FN(x, t) (sinf(x) * expf(-0.5f*(t)))
  #define FN_LABEL "sin(x)*exp(-t/2)"
#elif defined(FUNC_SIN2X)
  #define TARGET_FN(x, t) (sinf(2.0f*(x)) * expf(-(t)))
  #define FN_LABEL "sin(2x)*exp(-t)"
#elif defined(FUNC_GAUSS)
  #define TARGET_FN(x, t) (expf(-(x)*(x)/(float)(M_PI*M_PI)) * expf(-(t)))
  #define FN_LABEL "exp(-x^2/pi^2)*exp(-t)"
#else
  #define TARGET_FN(x, t) (cosf(x) * expf(-(t)))
  #define FN_LABEL "cos(x)*exp(-t)"
#endif

/* Feedforward sublayer inner dimension */
#define D_FF (WIDTH * 2)

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

int main(int argc, char **argv)
{
    int i, k;
    int n = NX * NT;
    float **input  = (float **)malloc(n * sizeof(float *));
    float **output = (float **)malloc(n * sizeof(float *));

    k = 0;
    for (i = 0; i < NX; i++) {
        int j;
        for (j = 0; j < NT; j++) {
            input[k]  = (float *)malloc(2 * sizeof(float));
            output[k] = (float *)malloc(sizeof(float));
            float x = (float)(i * 2.0 * M_PI / (NX - 1));
            float t = (float)(j * 2.0 / (NT - 1));
            input[k][0]  = x;
            input[k][1]  = t;
            output[k][0] = TARGET_FN(x, t);
            k++;
        }
    }

    kad_node_t *t_net = kann_layer_input(2);
    t_net = kann_layer_dense(t_net, WIDTH);
    t_net = kad_tanh(t_net);
    for (i = 0; i < DEPTH; i++)
        t_net = transformer_block(t_net, WIDTH, D_FF);
    t_net = kann_layer_cost(t_net, 1, KANN_C_MSE);
    kann_t *ann = kann_new(t_net, 0);

    kann_train_fnn1(ann, 0.001f, 64, 50, 10, 0.1f, n, input, output);

    kann_save("transformer_optim.kann", ann);
    kann_delete(ann);
    ann = kann_load("transformer_optim.kann");

    float val_cost = kann_cost_fnn1(ann, n, input, output);
    printf("Validation MSE: %.6f\n", val_cost);
    printf("(NX=%d, NT=%d, WIDTH=%d, DEPTH=%d, FN=%s)\n", NX, NT, WIDTH, DEPTH, FN_LABEL);

    /* L2 and L-infinity rollout norms — parsed by transformer_optim_ab.py */
    int n_t_eval = 25, n_x_eval = 100;
    float dx = (float)(2.0 * M_PI / (n_x_eval - 1));
    printf("L2NORM");
    for (int jt = 0; jt <= n_t_eval; jt++) {
        float t = (float)(jt * 3.0 / n_t_eval);
        float l2 = 0.0f;
        for (int ix = 0; ix < n_x_eval; ix++) {
            float x = (float)(ix * 2.0 * M_PI / (n_x_eval - 1));
            float exact = TARGET_FN(x, t);
            float xv[2] = {x, t};
            const float *y = kann_apply1(ann, xv);
            float err = y[0] - exact;
            l2 += err * err * dx;
        }
        printf(" %.6f", sqrtf(l2));
    }
    printf("\n");
    printf("LINF");
    for (int jt = 0; jt <= n_t_eval; jt++) {
        float t = (float)(jt * 3.0 / n_t_eval);
        float linf = 0.0f;
        for (int ix = 0; ix < n_x_eval; ix++) {
            float x = (float)(ix * 2.0 * M_PI / (n_x_eval - 1));
            float exact = TARGET_FN(x, t);
            float xv[2] = {x, t};
            const float *y = kann_apply1(ann, xv);
            float err = fabsf(y[0] - exact);
            if (err > linf) linf = err;
        }
        printf(" %.6f", linf);
    }
    printf("\n");

    for (i = 0; i < n; i++) { free(input[i]); free(output[i]); }
    free(input);
    free(output);
    kann_delete(ann);
    return 0;
}
