// Bloc transformer pre-norm, avec attention multi-têtes et KV cache.
//   x = x + MHA(LayerNorm(x))
//   x = x + MLP(LayerNorm(x))
// Compilation : gcc -O2 transformer_block.c -o transformer_block -lm
#include <stdio.h>
#include <math.h>

#define T  6            // longueur maximale (capacité du cache)
#define H  2            // nombre de têtes
#define DH 4            // dimension par tête
#define DM (H * DH)     // dimension du modèle
#define FF (4 * DM)     // dimension cachée du MLP

typedef struct {
    float ln1_g[DM], ln1_b[DM];                  // LayerNorm avant l'attention
    float Wq[DM][DM], Wk[DM][DM], Wv[DM][DM], Wo[DM][DM];
    float ln2_g[DM], ln2_b[DM];                  // LayerNorm avant le MLP
    float W1[DM][FF], b1[FF];                    // MLP : DM -> FF
    float W2[FF][DM], b2[DM];                    // MLP : FF -> DM
} Block;

typedef struct {
    float K[T][DM], V[T][DM];
    int len;
} KVCache;

// ---------- briques de base ----------

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// LayerNorm : centre et réduit le vecteur, puis applique gamma et beta.
static void layernorm(const float x[DM], const float g[DM], const float b[DM], float y[DM]) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < DM; i++) mean += x[i];
    mean /= DM;
    for (int i = 0; i < DM; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= DM;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < DM; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}

// GELU (approximation tanh), l'activation standard des transformers.
static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// y[DM] = x[DM] * W[DM][DM]
static void matvec(const float x[DM], const float W[DM][DM], float y[DM]) {
    for (int j = 0; j < DM; j++) {
        float acc = 0.0f;
        for (int i = 0; i < DM; i++) acc += x[i] * W[i][j];
        y[j] = acc;
    }
}

// ---------- sous-couches ----------

static void attention(const Block *b, KVCache *c, const float x[DM], float out[DM]) {
    float q[DM], cat[DM];
    matvec(x, b->Wq, q);
    matvec(x, b->Wk, c->K[c->len]);
    matvec(x, b->Wv, c->V[c->len]);
    c->len++;
    for (int h = 0; h < H; h++) {
        int off = h * DH;
        float s[T];
        for (int j = 0; j < c->len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < DH; d++) dot += q[off + d] * c->K[j][off + d];
            s[j] = dot / sqrtf((float)DH);
        }
        softmax(s, c->len);
        for (int d = 0; d < DH; d++) {
            float acc = 0.0f;
            for (int j = 0; j < c->len; j++) acc += s[j] * c->V[j][off + d];
            cat[off + d] = acc;
        }
    }
    matvec(cat, b->Wo, out);
}

static void mlp(const Block *b, const float x[DM], float out[DM]) {
    float hid[FF];
    for (int j = 0; j < FF; j++) {
        float acc = b->b1[j];
        for (int i = 0; i < DM; i++) acc += x[i] * b->W1[i][j];
        hid[j] = gelu(acc);
    }
    for (int j = 0; j < DM; j++) {
        float acc = b->b2[j];
        for (int i = 0; i < FF; i++) acc += hid[i] * b->W2[i][j];
        out[j] = acc;
    }
}

// ---------- le bloc ----------

// Traite UN token : x est modifié en place (connexions résiduelles).
void block_step(const Block *b, KVCache *c, float x[DM]) {
    float n[DM], y[DM];

    layernorm(x, b->ln1_g, b->ln1_b, n);
    attention(b, c, n, y);
    for (int i = 0; i < DM; i++) x[i] += y[i];   // résiduel 1

    layernorm(x, b->ln2_g, b->ln2_b, n);
    mlp(b, n, y);
    for (int i = 0; i < DM; i++) x[i] += y[i];   // résiduel 2
}

// ---------- initialisation et démo ----------

static unsigned rng_state = 12345u;
static float rnd(void) {                         // uniforme dans [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void fill(float *w, int n, float scale) {
    for (int i = 0; i < n; i++) w[i] = rnd() * scale;
}

static void init(Block *b) {
    float s = 1.0f / sqrtf((float)DM);
    for (int i = 0; i < DM; i++) {
        b->ln1_g[i] = b->ln2_g[i] = 1.0f;
        b->ln1_b[i] = b->ln2_b[i] = 0.0f;
        b->b2[i] = 0.0f;
    }
    for (int j = 0; j < FF; j++) b->b1[j] = 0.0f;
    fill(&b->Wq[0][0], DM * DM, s);  fill(&b->Wk[0][0], DM * DM, s);
    fill(&b->Wv[0][0], DM * DM, s);  fill(&b->Wo[0][0], DM * DM, s);
    fill(&b->W1[0][0], DM * FF, s);
    fill(&b->W2[0][0], FF * DM, 1.0f / sqrtf((float)FF));
}

int main(void) {
    static Block blk;
    init(&blk);

    KVCache cache = { .len = 0 };
    for (int t = 0; t < T; t++) {
        float x[DM];
        for (int i = 0; i < DM; i++) x[i] = rnd();   // embedding fictif du token t
        block_step(&blk, &cache, x);
        printf("t=%d :", t);
        for (int i = 0; i < DM; i++) printf(" %7.3f", x[i]);
        printf("\n");
    }
    return 0;
}
