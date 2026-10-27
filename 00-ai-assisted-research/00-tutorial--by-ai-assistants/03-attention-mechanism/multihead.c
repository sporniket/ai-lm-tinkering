// Attention multi-têtes causale (libc + libm uniquement).
// Compilation : gcc -O2 multihead.c -o multihead -lm
#include <stdio.h>
#include <math.h>

#define T  4            // longueur de la séquence
#define H  2            // nombre de têtes
#define DH 3            // dimension par tête
#define DM (H * DH)     // dimension du modèle

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Q, K, V : [T][DM]. La tête h travaille sur les colonnes [h*DH, (h+1)*DH).
// Les sorties des têtes sont écrites côte à côte dans out : c'est la concaténation.
void multihead_attention(float Q[T][DM], float K[T][DM], float V[T][DM],
                         float out[T][DM]) {
    for (int h = 0; h < H; h++) {
        int off = h * DH;                       // début de la tranche de cette tête
        for (int t = 0; t < T; t++) {
            float s[T];
            for (int j = 0; j <= t; j++) {      // masque causal
                float dot = 0.0f;
                for (int d = 0; d < DH; d++)
                    dot += Q[t][off + d] * K[j][off + d];
                s[j] = dot / sqrtf((float)DH);
            }
            softmax(s, t + 1);
            for (int d = 0; d < DH; d++) {
                float acc = 0.0f;
                for (int j = 0; j <= t; j++) acc += s[j] * V[j][off + d];
                out[t][off + d] = acc;
            }
        }
    }
}

int main(void) {
    float Q[T][DM] = {{1,0,0, 0,1,0}, {0,1,0, 1,0,0}, {1,1,0, 0,0,1}, {0,0,1, 1,1,0}};
    float K[T][DM] = {{1,0,0, 0,1,0}, {0,1,0, 1,0,0}, {0,0,1, 0,0,1}, {1,1,0, 1,1,0}};
    float V[T][DM] = {{1,2,3, 1,0,0}, {4,5,6, 0,1,0}, {7,8,9, 0,0,1}, {10,11,12, 1,1,1}};
    float out[T][DM];

    multihead_attention(Q, K, V, out);

    for (int t = 0; t < T; t++) {
        printf("t=%d :", t);
        for (int d = 0; d < DM; d++) {
            if (d == DH) printf("  |");
            printf(" %7.3f", out[t][d]);
        }
        printf("\n");
    }
    return 0;
}
