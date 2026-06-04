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
   VARIATION 2: Width (neurons per hidden layer)  (default: 256)
   ===================================================================== */
#ifndef WIDTH
#define WIDTH 256
#endif

/* =====================================================================
   VARIATION 3: Depth (number of hidden layers)  (default: 1)
   ===================================================================== */
#ifndef DEPTH
#define DEPTH 1
#endif

int main(int argc, char **argv){

    int n = NX * NT;
    float **input  = (float **)malloc(n * sizeof(float *));
    float **output = (float **)malloc(n * sizeof(float *));

    int k = 0;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NT; j++) {
            input[k]  = (float *)malloc(2 * sizeof(float));
            output[k] = (float *)malloc(sizeof(float));
            float x = (float)(i * 2.0 * M_PI / (NX - 1));
            float t = (float)(j * 2.0 / (NT - 1));
            input[k][0]  = x;
            input[k][1]  = t;
            output[k][0] = tanhf(x) * expf(-t);
            k++;
        }
    }

    kad_node_t *t_net = kann_layer_input(2);
    for (int i = 0; i < DEPTH; i++) {
        t_net = kann_layer_dense(t_net, WIDTH);
        t_net = kad_tanh(t_net);
    }
    t_net = kann_layer_cost(t_net, 1, KANN_C_MSE);
    kann_t *ann = kann_new(t_net, 0);

    kann_train_fnn1(ann, 0.001f, 64, 50, 10, 0.1f, n, input, output);

    kann_save("simple_mlp.kann", ann);
    kann_delete(ann);
    ann = kann_load("simple_mlp.kann");

    printf("x\t\tt\t\tu(x,t) exact\tprediction\n");
    for (int i = 0; i <= 4; i++) {
        for (int j = 0; j <= 4; j++) {
            float x = (float)(i * 2.0 * M_PI / 4.0);
            float t = (float)(j * 3.0 / 4.0);
            float exact = tanh(x) * expf(-t);
            float xv[2] = {x, t};
            const float *y = kann_apply1(ann, xv);
            printf("%.4f\t\t%.4f\t\t%.6f\t%.6f\n", x, t, exact, y[0]);
        }
        printf("\n");
    }

    float val_cost = kann_cost_fnn1(ann, n, input, output);
    printf("Validation MSE: %.6f\n", val_cost);
    printf("(NX=%d, NT=%d, WIDTH=%d, DEPTH=%d)\n", NX, NT, WIDTH, DEPTH);

    /* Rollout divergence: MSE at each time slice t in [0, 3.0]
       t > 2.0 is outside the training range — tests temporal generalization */
    int n_t_eval = 25, n_x_eval = 100;
    printf("ROLLOUT");
    for (int jt = 0; jt <= n_t_eval; jt++) {
        float t = (float)(jt * 3.0 / n_t_eval);
        float mse_t = 0.0f;
        for (int ix = 0; ix < n_x_eval; ix++) {
            float x = (float)(ix * 2.0 * M_PI / (n_x_eval - 1));
            float exact = tanh(x) * expf(-t);
            float xv[2] = {x, t};
            const float *y = kann_apply1(ann, xv);
            float err = y[0] - exact;
            mse_t += err * err;
        }
        printf(" %.6f", mse_t / n_x_eval);
    }
    printf("\n");

    for (int i = 0; i < n; i++) { free(input[i]); free(output[i]); }
    free(input);
    free(output);
    kann_delete(ann);
    return 0;
}
