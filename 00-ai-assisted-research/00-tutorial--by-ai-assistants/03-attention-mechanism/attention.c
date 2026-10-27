// Attention causale scaled dot-product, une seule tête.
// Compilation : gcc -O2 attention.c -o attention -lm
#include <stdio.h>
#include <math.h>

#define T 4   // longueur de la séquence
#define D 3   // dimension d'une tête

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// out[t] = somme_j poids(t,j) * V[j], avec j <= t (masque causal)
void attention(float Q[T][D], float K[T][D], float V[T][D], float out[T][D]) {
    for (int t = 0; t < T; t++) {
        float s[T];
        for (int j = 0; j <= t; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += Q[t][d] * K[j][d];
            s[j] = dot / sqrtf((float)D);   // mise à l'échelle
        }
        softmax(s, t + 1);                  // poids qui somment à 1
        for (int d = 0; d < D; d++) {
            out[t][d] = 0.0f;
            for (int j = 0; j <= t; j++) out[t][d] += s[j] * V[j][d];
        }
    }
}

int main(void) {
    float Q[T][D] = {{1,0,0},{0,1,0},{1,1,0},{0,0,1}};
    float K[T][D] = {{1,0,0},{0,1,0},{0,0,1},{1,1,0}};
    float V[T][D] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    float out[T][D];

    attention(Q, K, V, out);

    for (int t = 0; t < T; t++) {
        printf("t=%d :", t);
        for (int d = 0; d < D; d++) printf(" %7.3f", out[t][d]);
        printf("\n");
    }
    return 0;
}
