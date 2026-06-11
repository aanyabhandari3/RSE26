#include "kann.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
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
   WIDTH controls d_model here — kept consistent with MLP/GRU files
   so sweep_width.py works on all three without changes.
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
   Pass -DFUNC_SIN, -DFUNC_COS, or -DFUNC_TANH to sweep_functions.py,
   or change the #define below to switch manually.
   ===================================================================== */
#if defined(FUNC_SIN)
  #define TARGET_FN(x, t) (sinf(x) * expf(-(t)))
  #define FN_LABEL "sin"
#elif defined(FUNC_TANH)
  #define TARGET_FN(x, t) (tanhf(x) * expf(-(t)))
  #define FN_LABEL "tanh"
#else
  #define TARGET_FN(x, t) (cosf(x) * expf(-(t)))
  #define FN_LABEL "cos"
#endif

/* Feedforward sublayer inner dimension — standard transformer uses 2-4x d_model */
#define D_FF (WIDTH * 2)

/* =====================================================================
   transformer_block: one encoder block
   Input/output shape: (batch, d_model)

   Self-attention sublayer:
     Q = h * W_Q^T          projection to query space
     K = h * W_K^T          projection to key space
     V = h * W_V^T          projection to value space
     score = Q ⊙ K          element-wise product (feature-wise attention)
     attn  = softmax(score) attention weights over feature dimension
     out   = attn ⊙ V       weighted values
     out   = out * W_O^T    output projection
     h     = LayerNorm(h + out)  residual + norm

   Feedforward sublayer:
     ff = tanh(dense(h, D_FF))
     ff = dense(ff, d_model)
     h  = LayerNorm(h + ff)  residual + norm
   ===================================================================== */
static kad_node_t *transformer_block(kad_node_t *h, int d_model, int d_ff)
{
    /* Q, K, V weight matrices: each (d_model x d_model) */
    kad_node_t *W_Q = kann_new_weight(d_model, d_model);
    kad_node_t *W_K = kann_new_weight(d_model, d_model);
    kad_node_t *W_V = kann_new_weight(d_model, d_model);
    kad_node_t *W_O = kann_new_weight(d_model, d_model);

    /* Project to query, key, value spaces
       kad_cmul(x, W) = x * W^T  so (B, d_model) * (d_model, d_model) = (B, d_model) */
    kad_node_t *Q = kad_cmul(h, W_Q);
    kad_node_t *K = kad_cmul(h, W_K);
    kad_node_t *V = kad_cmul(h, W_V);

    /* Feature-wise attention:
       score = Q * K  element-wise  (B, d_model)
       attn  = softmax over feature dimension  (B, d_model)
       out   = attn * V  element-wise gated values  (B, d_model) */
    kad_node_t *score = kad_mul(Q, K);
    kad_node_t *attn  = kad_softmax(score);
    kad_node_t *out   = kad_mul(attn, V);

    /* Output projection, residual connection, layer norm */
    out = kad_cmul(out, W_O);
    h   = kad_stdnorm(kad_add(h, out));

    /* Feedforward sublayer: dense -> tanh -> dense, with residual + norm */
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

    /* Build transformer network:
       (B, 2) -> input embedding -> DEPTH x transformer block -> output */
    kad_node_t *t_net = kann_layer_input(2);

    /* Input embedding: project 2D input into d_model space */
    t_net = kann_layer_dense(t_net, WIDTH);
    t_net = kad_tanh(t_net);

    /* Stack transformer blocks */
    for (i = 0; i < DEPTH; i++)
        t_net = transformer_block(t_net, WIDTH, D_FF);

    /* Output: project d_model -> 1 scalar with MSE cost */
    t_net = kann_layer_cost(t_net, 1, KANN_C_MSE);
    kann_t *ann = kann_new(t_net, 0);

    kann_train_fnn1(ann, 0.001f, 64, 50, 10, 0.1f, n, input, output);

    kann_save("simple_transformer.kann", ann);
    kann_delete(ann);
    ann = kann_load("simple_transformer.kann");

    /* Prediction table — same grid as MLP and GRU for direct comparison */
    printf("x\t\tt\t\tu(x,t) exact\tprediction\n");
    for (i = 0; i <= 4; i++) {
        int j;
        for (j = 0; j <= 4; j++) {
            float x = (float)(i * 2.0 * M_PI / 4.0);
            float t = (float)(j * 3.0 / 4.0);
            float exact = TARGET_FN(x, t);
            float xv[2] = {x, t};
            const float *y = kann_apply1(ann, xv);
            printf("%.4f\t\t%.4f\t\t%.6f\t%.6f\n", x, t, exact, y[0]);
        }
        printf("\n");
    }

    float val_cost = kann_cost_fnn1(ann, n, input, output);
    printf("Validation MSE: %.6f\n", val_cost);
    printf("(NX=%d, NT=%d, WIDTH=%d, DEPTH=%d, FN=%s)\n", NX, NT, WIDTH, DEPTH, FN_LABEL);

    /* Error norms — same format as MLP and GRU so error_norms_ab.py can read them */
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
