// Couche d'attention multi-têtes complète : X -> (Wq,Wk,Wv) -> têtes -> Wo.
// Compilation : gcc -O2 mha_layer.c -o mha_layer -lm
#include <stdio.h>
#include <math.h>

#define T  4            // longueur de la séquence
#define H  2            // nombre de têtes
#define DH 4            // dimension par tête
#define DM (H * DH)     // dimension du modèle

typedef struct {
    float Wq[DM][DM], Wk[DM][DM], Wv[DM][DM], Wo[DM][DM];
} MHA;

// Y[T][DM] = X[T][DM] * W[DM][DM]
static void matmul(float X[T][DM], float W[DM][DM], float Y[T][DM]) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < DM; j++) {
            float acc = 0.0f;
            for (int i = 0; i < DM; i++) acc += X[t][i] * W[i][j];
            Y[t][j] = acc;
        }
}

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Attention causale par tête, sorties concaténées dans cat.
static void heads(float Q[T][DM], float K[T][DM], float V[T][DM], float cat[T][DM]) {
    for (int h = 0; h < H; h++) {
        int off = h * DH;
        for (int t = 0; t < T; t++) {
            float s[T];
            for (int j = 0; j <= t; j++) {
                float dot = 0.0f;
                for (int d = 0; d < DH; d++) dot += Q[t][off + d] * K[j][off + d];
                s[j] = dot / sqrtf((float)DH);
            }
            softmax(s, t + 1);
            for (int d = 0; d < DH; d++) {
                float acc = 0.0f;
                for (int j = 0; j <= t; j++) acc += s[j] * V[j][off + d];
                cat[t][off + d] = acc;
            }
        }
    }
}

void mha_forward(const MHA *m, float X[T][DM], float out[T][DM]) {
    float Q[T][DM], K[T][DM], V[T][DM], cat[T][DM];
    matmul(X, (float (*)[DM])m->Wq, Q);     // projections apprises
    matmul(X, (float (*)[DM])m->Wk, K);
    matmul(X, (float (*)[DM])m->Wv, V);
    heads(Q, K, V, cat);                    // attention + concaténation
    matmul(cat, (float (*)[DM])m->Wo, out); // projection de sortie
}

// Générateur pseudo-aléatoire déterministe pour initialiser les poids.
static unsigned rng_state = 12345u;
static float rnd(void) {                    // uniforme dans [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void init(float W[DM][DM]) {
    float scale = 1.0f / sqrtf((float)DM);  // évite que les valeurs explosent
    for (int i = 0; i < DM; i++)
        for (int j = 0; j < DM; j++) W[i][j] = rnd() * scale;
}

int main(void) {
    MHA m;
    init(m.Wq); init(m.Wk); init(m.Wv); init(m.Wo);

    float X[T][DM], out[T][DM];
    for (int t = 0; t < T; t++)
        for (int i = 0; i < DM; i++) X[t][i] = rnd();

    mha_forward(&m, X, out);

    for (int t = 0; t < T; t++) {
        printf("t=%d :", t);
        for (int d = 0; d < DM; d++) printf(" %7.3f", out[t][d]);
        printf("\n");
    }
    return 0;
}
