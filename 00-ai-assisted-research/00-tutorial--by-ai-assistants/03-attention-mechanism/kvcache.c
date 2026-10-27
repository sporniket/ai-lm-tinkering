// KV cache : génération token par token sans recalculer K et V du passé.
// Compilation : gcc -O2 kvcache.c -o kvcache -lm
#include <stdio.h>
#include <math.h>

#define T  6            // longueur maximale (capacité du cache)
#define H  2            // nombre de têtes
#define DH 4            // dimension par tête
#define DM (H * DH)     // dimension du modèle

typedef struct {
    float Wq[DM][DM], Wk[DM][DM], Wv[DM][DM], Wo[DM][DM];
} MHA;

typedef struct {
    float K[T][DM], V[T][DM];   // clés et valeurs de tous les tokens déjà vus
    int len;                    // nombre de tokens en cache
} KVCache;

// y[DM] = x[DM] * W[DM][DM]
static void matvec(const float x[DM], const float W[DM][DM], float y[DM]) {
    for (int j = 0; j < DM; j++) {
        float acc = 0.0f;
        for (int i = 0; i < DM; i++) acc += x[i] * W[i][j];
        y[j] = acc;
    }
}

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Attention d'UNE requête q contre n clés/valeurs, toutes têtes, sortie concaténée.
static void attend(const float q[DM], float K[][DM], float V[][DM], int n, float cat[DM]) {
    for (int h = 0; h < H; h++) {
        int off = h * DH;
        float s[T];
        for (int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (int d = 0; d < DH; d++) dot += q[off + d] * K[j][off + d];
            s[j] = dot / sqrtf((float)DH);
        }
        softmax(s, n);
        for (int d = 0; d < DH; d++) {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += s[j] * V[j][off + d];
            cat[off + d] = acc;
        }
    }
}

// --- Version SANS cache : recalcule K et V de tout le préfixe à chaque appel ---
void mha_naive(const MHA *m, float X[][DM], int n, float out[DM]) {
    float K[T][DM], V[T][DM], q[DM], cat[DM];
    for (int t = 0; t < n; t++) {
        matvec(X[t], m->Wk, K[t]);
        matvec(X[t], m->Wv, V[t]);
    }
    matvec(X[n - 1], m->Wq, q);             // seule la requête du dernier token compte
    attend(q, K, V, n, cat);
    matvec(cat, m->Wo, out);
}

// --- Version AVEC cache : on ne projette que le nouveau token ---
void mha_step(const MHA *m, KVCache *c, const float x[DM], float out[DM]) {
    float q[DM], cat[DM];
    matvec(x, m->Wq, q);
    matvec(x, m->Wk, c->K[c->len]);         // on ajoute k et v au cache
    matvec(x, m->Wv, c->V[c->len]);
    c->len++;
    attend(q, c->K, c->V, c->len, cat);     // le cache sert de passé
    matvec(cat, m->Wo, out);
}

static unsigned rng_state = 12345u;
static float rnd(void) {                    // uniforme dans [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void init(float W[DM][DM]) {
    float scale = 1.0f / sqrtf((float)DM);
    for (int i = 0; i < DM; i++)
        for (int j = 0; j < DM; j++) W[i][j] = rnd() * scale;
}

int main(void) {
    MHA m;
    init(m.Wq); init(m.Wk); init(m.Wv); init(m.Wo);

    float X[T][DM];
    for (int t = 0; t < T; t++)
        for (int i = 0; i < DM; i++) X[t][i] = rnd();

    KVCache cache = { .len = 0 };
    float max_diff = 0.0f;

    for (int t = 0; t < T; t++) {
        float a[DM], b[DM];
        mha_naive(&m, X, t + 1, a);         // sans cache
        mha_step(&m, &cache, X[t], b);      // avec cache
        for (int d = 0; d < DM; d++) {
            float diff = fabsf(a[d] - b[d]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("t=%d (cache=%d) :", t, cache.len);
        for (int d = 0; d < DM; d++) printf(" %7.3f", b[d]);
        printf("\n");
    }
    printf("ecart max sans cache vs avec cache : %g\n", max_diff);
    return 0;
}
