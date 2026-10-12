#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_DIM 4
#define OUTPUT_DIM 4
#define BATCH_SIZE 3
#define EPOCHS 500
#define LEARNING_RATE 0.5f

// Structure de la couche 1.58-bit
typedef struct {
    float W_FP[OUTPUT_DIM][INPUT_DIM]; // Poids maîtres (haute précision)
    float b[OUTPUT_DIM];               // Biais
    float W_quant[OUTPUT_DIM][INPUT_DIM]; // Stockage des poids quantifiés (-1, 0, 1)
    
    // Buffers pour le backward pass
    float cached_x[INPUT_DIM][BATCH_SIZE];
} BitLinearLayer;

// Initialisation aléatoire des poids
void init_layer(BitLinearLayer *layer) {
    for (int i = 0; i < OUTPUT_DIM; i++) {
        layer->b[i] = 0.0f;
        for (int j = 0; j < INPUT_DIM; j++) {
            // Initialisation normalisée de type Xavier / Glorot
            layer->W_FP[i][j] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sqrtf(2.0f / INPUT_DIM);
        }
    }
}

// Phase Forward
void forward(BitLinearLayer *layer, float x[INPUT_DIM][BATCH_SIZE], float out[OUTPUT_DIM][BATCH_SIZE]) {
    // 1. Mettre en cache l'entrée pour le backward pass
    for (int i = 0; i < INPUT_DIM; i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            layer->cached_x[i][b] = x[i][b];
        }
    }

    // 2. Quantifier les poids en ternaire (-1, 0, 1) selon l'approche BitNet b1.58
    float sum_abs = 0.0f;
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            sum_abs += fabsf(layer->W_FP[i][j]);
        }
    }
    float scale_w = (sum_abs / (OUTPUT_DIM * INPUT_DIM)) + 1e-5f;

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            // Division par l'échelle, arrondi, puis contrainte entre -1 et 1
            float val = roundf(layer->W_FP[i][j] / scale_w);
            if (val > 1.0f) val = 1.0f;
            if (val < -1.0f) val = -1.0f;
            layer->W_quant[i][j] = val;
        }
    }

    // 3. Multiplication matricielle : out = W_quant * x + b
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            out[i][b] = layer->b[i]; // Commencer par le biais
            for (int j = 0; j < INPUT_DIM; j++) {
                out[i][b] += layer->W_quant[i][j] * x[j][b];
            }
        }
    }
}

// Phase Backward & Mise à jour (Optimiseur SGD intégré via le Straight-Through Estimator)
void backward_and_update(BitLinearLayer *layer, float dout[OUTPUT_DIM][BATCH_SIZE]) {
    float dW[OUTPUT_DIM][INPUT_DIM] = {0};
    float db[OUTPUT_DIM] = {0};

    // Calcul du gradient par rapport aux poids (dW = dout * x^T) et biais
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            db[i] += dout[i][b];
            for (int j = 0; j < INPUT_DIM; j++) {
                dW[i][j] += dout[i][b] * layer->cached_x[j][b];
            }
        }
    }

    // Application de la descente de gradient sur les poids MAÎTRES flottants (STE)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        layer->b[i] -= LEARNING_RATE * (db[i] / BATCH_SIZE);
        for (int j = 0; j < INPUT_DIM; j++) {
            layer->W_FP[i][j] -= LEARNING_RATE * (dW[i][j] / BATCH_SIZE);
        }
    }
}

// Calcul de la Loss (Softmax + Cross Entropy) et génération du gradient initial (dout)
float compute_loss_and_dout(float logits[OUTPUT_DIM][BATCH_SIZE], int targets[BATCH_SIZE], float dout[OUTPUT_DIM][BATCH_SIZE]) {
    float total_loss = 0.0f;

    for (int b = 0; b < BATCH_SIZE; b++) {
        // Trouver le max pour la stabilité numérique du Softmax
        float max_logit = logits[0][b];
        for (int i = 1; i < OUTPUT_DIM; i++) {
            if (logits[i][b] > max_logit) max_logit = logits[i][b];
        }

        // Calcul des exponentielles et de leur somme
        float sum_exp = 0.0f;
        float probs[OUTPUT_DIM];
        for (int i = 0; i < OUTPUT_DIM; i++) {
            probs[i] = expf(logits[i][b] - max_logit);
            sum_exp += probs[i];
        }

        // Normalisation des probabilités et calcul de la Cross-Entropy
        for (int i = 0; i < OUTPUT_DIM; i++) {
            probs[i] /= sum_exp;
            dout[i][b] = probs[i]; // Préparation du gradient (Probs)
        }

        // dout = Probs - 1 pour la bonne classe cible
        dout[targets[b]][b] -= 1.0f;

        // Ajouter à la perte globale (Loss)
        total_loss -= logf(probs[targets[b]] + 1e-15f);
    }

    return total_loss / BATCH_SIZE;
}

int main() {
    // Initialisation du générateur pseudo-aléatoire
    srand(42);

    // 1. Données d'entraînement simulées (Mots encodés en One-Hot)
    // Vocabulaire de 4 mots (0: le, 1: chat, 2: mange, 3: souris)
    // Séquence : "le" -> "chat" (0->1), "chat" -> "mange" (1->2), "mange" -> "souris" (2->3)
    float X_train[INPUT_DIM][BATCH_SIZE] = {
        {1.0f, 0.0f, 0.0f}, // Mot 0 ("le")
        {0.0f, 1.0f, 0.0f}, // Mot 1 ("chat")
        {0.0f, 0.0f, 1.0f}, // Mot 2 ("mange")
        {0.0f, 0.0f, 0.0f}  // Mot 3 ("souris")
    };
    int y_train[BATCH_SIZE] = {1, 2, 3}; // Cibles (chat, mange, souris)

    // 2. Instanciation du modèle
    BitLinearLayer model;
    init_layer(&model);

    printf("--- Entraînement du modèle 1-bit en C Pur ---\n");

    float logits[OUTPUT_DIM][BATCH_SIZE];
    float dout[OUTPUT_DIM][BATCH_SIZE];

    // 3. Boucle principale d'entraînement
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        forward(&model, X_train, logits);
        float loss = compute_loss_and_dout(logits, y_train, dout);
        backward_and_update(&model, dout);

        if (epoch % 100 == 0) {
            printf("Époque %3d | Perte (Loss) : %.4f\n", epoch, loss);
        }
    }

    // 4. Affichage des poids finaux (strictement ternaires)
    printf("\n--- Entraînement terminé ---\n");
    printf("Matrice de poids finale (uniquement -1, 0, 1) :\n");
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("[ ");
        for (int j = 0; j < INPUT_DIM; j++) {
            printf("%2.0f ", model.W_quant[i][j]);
        }
        printf("]\n");
    }

    return 0;
}

