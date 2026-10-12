/* Embeddings : lookup dans une table + encodage positionnel sinusoidal.
 * Compilation : gcc -std=c99 -O2 embeddings.c -o embeddings -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 10
#define D_MODEL    8

/* Nombre aleatoire gaussien (Box-Muller) */
static float randn(void)
{
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* Table d'embeddings E[VOCAB_SIZE][D_MODEL], init aleatoire.
 * Dans un vrai modele, ces valeurs sont apprises par retropropagation. */
static void init_embeddings(float *E)
{
    for (int i = 0; i < VOCAB_SIZE * D_MODEL; i++)
        E[i] = 0.02f * randn();
}

/* Encodage positionnel sinusoidal ("Attention Is All You Need") :
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 */
static void positional_encoding(int pos, float *pe)
{
    for (int i = 0; i < D_MODEL; i += 2) {
        float freq = powf(10000.0f, (float)i / D_MODEL);
        pe[i] = sinf(pos / freq);
        if (i + 1 < D_MODEL)
            pe[i + 1] = cosf(pos / freq);
    }
}

/* Pour chaque token : x[t] = E[id] + PE(t) */
static void embed(const float *E, const int *ids, int n, float *out)
{
    for (int t = 0; t < n; t++) {
        const float *row = E + (size_t)ids[t] * D_MODEL; /* lookup */
        float pe[D_MODEL];
        positional_encoding(t, pe);
        for (int j = 0; j < D_MODEL; j++)
            out[t * D_MODEL + j] = row[j] + pe[j];
    }
}

int main(void)
{
    srand(42);

    float E[VOCAB_SIZE * D_MODEL];
    init_embeddings(E);

    /* Sortie du tokenizer (IDs fictifs) */
    int ids[] = {3, 7, 1, 7};
    int n = (int)(sizeof ids / sizeof ids[0]);

    float out[4 * D_MODEL];
    embed(E, ids, n, out);

    for (int t = 0; t < n; t++) {
        printf("pos %d (token %d) : ", t, ids[t]);
        for (int j = 0; j < D_MODEL; j++)
            printf("%7.3f ", out[t * D_MODEL + j]);
        printf("\n");
    }
    /* Remarque : le token 7 apparait aux positions 1 et 3 ;
     * meme ligne de E, mais vecteurs finaux differents grace a PE. */
    return 0;
}
