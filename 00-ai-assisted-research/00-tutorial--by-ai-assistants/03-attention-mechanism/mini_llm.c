// Mini LLM complet (poids aléatoires) : embeddings, RoPE, N blocs avec KV cache,
// LayerNorm finale, projection vers le vocabulaire, échantillonnage.
// Compilation : gcc -O2 mini_llm.c -o mini_llm -lm
#include <stdio.h>
#include <math.h>

#define VOCAB 32        // taille du vocabulaire
#define NL    2         // nombre de blocs
#define T     16        // longueur maximale (contexte)
#define H     2         // nombre de têtes
#define DH    4         // dimension par tête (paire, pour RoPE)
#define DM    (H * DH)  // dimension du modèle
#define FF    (4 * DM)  // dimension cachée du MLP

typedef struct {
    float ln1_g[DM], ln1_b[DM];
    float Wq[DM][DM], Wk[DM][DM], Wv[DM][DM], Wo[DM][DM];
    float ln2_g[DM], ln2_b[DM];
    float W1[DM][FF], b1[FF];
    float W2[FF][DM], b2[DM];
} Block;

typedef struct {
    float emb[VOCAB][DM];       // embeddings (réutilisés pour la sortie : "weight tying")
    Block blocks[NL];
    float lnf_g[DM], lnf_b[DM]; // LayerNorm finale
} Model;

typedef struct { float K[T][DM], V[T][DM]; int len; } KVCache;

// ---------- briques de base ----------

static void softmax(float *x, int n) {
    float max = x[0], sum = 0.0f;
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void layernorm(const float x[DM], const float g[DM], const float b[DM], float y[DM]) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < DM; i++) mean += x[i];
    mean /= DM;
    for (int i = 0; i < DM; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= DM;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < DM; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void matvec(const float x[DM], const float W[DM][DM], float y[DM]) {
    for (int j = 0; j < DM; j++) {
        float acc = 0.0f;
        for (int i = 0; i < DM; i++) acc += x[i] * W[i][j];
        y[j] = acc;
    }
}

// ---------- RoPE : rotation de chaque paire de coordonnées selon la position ----------

static void rope(float v[DM], int pos) {
    for (int h = 0; h < H; h++)
        for (int i = 0; i < DH / 2; i++) {
            float theta = pos * powf(10000.0f, -2.0f * i / DH);
            float c = cosf(theta), s = sinf(theta);
            float *p = &v[h * DH + 2 * i];
            float a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        }
}

// ---------- sous-couches ----------

static void attention(const Block *b, KVCache *c, const float x[DM], int pos, float out[DM]) {
    float q[DM], cat[DM];
    matvec(x, b->Wq, q);
    matvec(x, b->Wk, c->K[c->len]);
    matvec(x, b->Wv, c->V[c->len]);
    rope(q, pos);                       // RoPE sur q et k uniquement (pas sur v)
    rope(c->K[c->len], pos);            // k est mis en cache DÉJÀ tourné
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

static void block_step(const Block *b, KVCache *c, float x[DM], int pos) {
    float n[DM], y[DM];
    layernorm(x, b->ln1_g, b->ln1_b, n);
    attention(b, c, n, pos, y);
    for (int i = 0; i < DM; i++) x[i] += y[i];
    layernorm(x, b->ln2_g, b->ln2_b, n);
    mlp(b, n, y);
    for (int i = 0; i < DM; i++) x[i] += y[i];
}

// ---------- le modèle complet : token -> logits ----------

void model_step(const Model *m, KVCache caches[NL], int token, int pos, float logits[VOCAB]) {
    float x[DM], n[DM];
    for (int i = 0; i < DM; i++) x[i] = m->emb[token][i];       // 1. embedding
    for (int l = 0; l < NL; l++)                                // 2. pile de blocs
        block_step(&m->blocks[l], &caches[l], x, pos);
    layernorm(x, m->lnf_g, m->lnf_b, n);                        // 3. LayerNorm finale
    for (int v = 0; v < VOCAB; v++) {                           // 4. logits = n · emb^T
        float acc = 0.0f;
        for (int i = 0; i < DM; i++) acc += n[i] * m->emb[v][i];
        logits[v] = acc;
    }
}

// ---------- initialisation ----------

static unsigned rng_state = 12345u;
static float rnd(void) {                // uniforme dans [-1, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}
static float rnd01(void) { return 0.5f * (rnd() + 1.0f); }  // uniforme dans [0, 1)

static void fill(float *w, int n, float scale) {
    for (int i = 0; i < n; i++) w[i] = rnd() * scale;
}

static void init(Model *m) {
    float s = 1.0f / sqrtf((float)DM);
    fill(&m->emb[0][0], VOCAB * DM, 0.5f);
    for (int i = 0; i < DM; i++) { m->lnf_g[i] = 1.0f; m->lnf_b[i] = 0.0f; }
    for (int l = 0; l < NL; l++) {
        Block *b = &m->blocks[l];
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
}

// ---------- échantillonnage : glouton, ou température + top-k ----------

static int sample(const float logits[VOCAB], float temp, int k) {
    int best = 0;
    for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[best]) best = i;
    if (temp <= 0.0f) return best;                  // température 0 = argmax

    int idx[VOCAB], used[VOCAB] = {0};
    float p[VOCAB];
    if (k > VOCAB) k = VOCAB;
    for (int n = 0; n < k; n++) {                   // on garde les k meilleurs logits
        int b = -1;
        for (int i = 0; i < VOCAB; i++)
            if (!used[i] && (b < 0 || logits[i] > logits[b])) b = i;
        used[b] = 1; idx[n] = b; p[n] = logits[b] / temp;
    }
    softmax(p, k);
    float r = rnd01(), acc = 0.0f;
    for (int n = 0; n < k; n++) { acc += p[n]; if (r < acc) return idx[n]; }
    return idx[k - 1];
}

// ---------- génération : préremplissage du prompt, puis boucle autorégressive ----------

static void generate(const Model *m, const int *prompt, int n_prompt, float temp, int topk) {
    KVCache caches[NL];
    for (int l = 0; l < NL; l++) caches[l].len = 0;
    float logits[VOCAB];

    printf("prompt :");
    for (int i = 0; i < n_prompt; i++) {
        printf(" %d", prompt[i]);
        model_step(m, caches, prompt[i], i, logits);    // on remplit le cache
    }
    printf("\nsuite  :");
    for (int pos = n_prompt; pos < T; pos++) {
        int tok = sample(logits, temp, topk);           // choisir le prochain token
        printf(" %d", tok);
        model_step(m, caches, tok, pos, logits);        // le réinjecter
    }
    printf("\n");
}

int main(void) {
    static Model m;
    init(&m);

    // Vérification de RoPE : le produit scalaire ne dépend que de l'écart de position.
    float q[DM], k[DM], q2[DM], k2[DM];
    for (int i = 0; i < DM; i++) { q[i] = q2[i] = rnd(); k[i] = k2[i] = rnd(); }
    rope(q, 5);  rope(k, 3);                            // écart 2
    rope(q2, 9); rope(k2, 7);                           // écart 2 aussi
    float d1 = 0.0f, d2 = 0.0f;
    for (int i = 0; i < DM; i++) { d1 += q[i] * k[i]; d2 += q2[i] * k2[i]; }
    printf("RoPE : %.6f vs %.6f (positions 5/3 puis 9/7)\n\n", d1, d2);

    int prompt[] = {3, 7, 1};
    printf("--- glouton (temperature 0) ---\n");
    generate(&m, prompt, 3, 0.0f, 1);
    printf("--- echantillonnage (temperature 0.8, top-k 5) ---\n");
    generate(&m, prompt, 3, 0.8f, 5);
    return 0;
}
