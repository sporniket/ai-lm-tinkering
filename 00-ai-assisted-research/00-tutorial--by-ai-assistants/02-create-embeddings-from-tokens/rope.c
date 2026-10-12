/* RoPE : rotation des paires de dimensions de Q et K selon la position.
 * Compilation : gcc -std=c99 -O2 rope.c -o rope -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 10
#define D_MODEL    8   /* doit etre pair */

static float randn(void)
{
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* Applique RoPE en place a un vecteur x de dimension d, a la position pos.
 * Paire i : (x[2i], x[2i+1]) tourne de l'angle pos * theta_i,
 * avec theta_i = 10000^(-2i/d). */
static void rope_apply(float *x, int d, int pos)
{
    for (int i = 0; i < d / 2; i++) {
        float theta = powf(10000.0f, -2.0f * i / d);
        float c = cosf(pos * theta);
        float s = sinf(pos * theta);
        float a = x[2 * i];
        float b = x[2 * i + 1];
        x[2 * i]     = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/* Backward : la rotation inverse (angle negatif) s'applique au gradient.
 * Equivalent a rope_apply avec -pos. */
static void rope_backward(float *grad, int d, int pos)
{
    rope_apply(grad, d, -pos);
}

static float dot(const float *a, const float *b, int d)
{
    float s = 0.0f;
    for (int i = 0; i < d; i++)
        s += a[i] * b[i];
    return s;
}

/* Score d'attention q(m) . k(n) apres RoPE */
static float score(const float *q, const float *k, int m, int n)
{
    float qr[D_MODEL], kr[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        qr[i] = q[i];
        kr[i] = k[i];
    }
    rope_apply(qr, D_MODEL, m);
    rope_apply(kr, D_MODEL, n);
    return dot(qr, kr, D_MODEL);
}

int main(void)
{
    srand(42);

    /* 1) Table d'embeddings : simple lookup, sans vecteur de position */
    float E[VOCAB_SIZE * D_MODEL];
    for (int i = 0; i < VOCAB_SIZE * D_MODEL; i++)
        E[i] = 0.02f * randn();

    int ids[] = {3, 7, 1, 7};
    int n = (int)(sizeof ids / sizeof ids[0]);

    /* 2) Dans une couche d'attention, Q et K viennent de projections
     *    lineaires de x. Ici on prend directement x comme q et k. */
    printf("Le meme token (7) a deux positions donne des q differents :\n");
    for (int t = 0; t < n; t++) {
        float q[D_MODEL];
        for (int j = 0; j < D_MODEL; j++)
            q[j] = E[ids[t] * D_MODEL + j] * 50.0f; /* mise a l'echelle pour lisibilite */
        rope_apply(q, D_MODEL, t);
        printf("pos %d (token %d) : ", t, ids[t]);
        for (int j = 0; j < D_MODEL; j++)
            printf("%7.3f ", q[j]);
        printf("\n");
    }

    /* 3) Propriete cle : le score ne depend que de la distance n - m */
    float q[D_MODEL], k[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        q[i] = randn();
        k[i] = randn();
    }
    printf("\nScore q(m).k(n) pour une distance fixe de 3 :\n");
    int ms[] = {0, 2, 10, 100};
    for (int i = 0; i < 4; i++)
        printf("m=%3d n=%3d : %.5f\n", ms[i], ms[i] + 3,
               score(q, k, ms[i], ms[i] + 3));

    /* 4) Verification du backward : R(-m) . R(m) = identite */
    float g[D_MODEL];
    for (int i = 0; i < D_MODEL; i++)
        g[i] = q[i];
    rope_apply(g, D_MODEL, 7);
    rope_backward(g, D_MODEL, 7);
    float err = 0.0f;
    for (int i = 0; i < D_MODEL; i++)
        err += fabsf(g[i] - q[i]);
    printf("\nErreur aller-retour (rotation puis inverse) : %.2e\n", err);

    return 0;
}
