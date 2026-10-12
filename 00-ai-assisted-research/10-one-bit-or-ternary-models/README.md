---
source: https://share.google/aimode/DPYftPaoU19nzanIF
---

# Entraîner un un modèle LLM sans GPU

> train a llm model without gpu

Il est possible d'entraîner ou de fine-tuner (ajuster) un petit modèle de langage (SLM ou LLM miniature) sans carte graphique (GPU), mais cela reste extrêmement lent et limité aux tout petits modèles sur processeur (CPU). [1, 2] 
## Ce qui est possible sans GPU

* Entraîner un modèle miniature : Vous pouvez entraîner un modèle de quelques millions de paramètres (par exemple 10 à 50 millions) sur un petit jeu de données textuel via votre CPU, ce qui prendra plusieurs heures. [1] 
* Le Fine-Tuning par Distillation ou LoRA léger : Adapter un petit modèle pré-existant (comme Phi-3 ou Qwen 0.5B) sur un CPU puissant avec beaucoup de mémoire vive (RAM). [3] 

## Les étapes pour s'entraîner sur CPU

   1. Choisir un modèle minuscule : Ne tentez pas d'entraîner un modèle de 7 milliards de paramètres (7B) sur un CPU. Visez un modèle de taille microscopique (ex: 0.5B ou moins). [4, 5] 
   2. Utiliser un outil adapté : Des bibliothèques Python comme PyTorch ou Hugging Face transformers fonctionnent sur CPU, mais l'entraînement de grands volumes de données y est proscrit. Pour un apprentissage pédagogique sur PC sans GPU, inspirez-vous de tutoriels de micro-entraînement (comme l'entraînement d'un mini-modèle sur les œuvres de Shakespeare). [1] 
   3. Préférer la location cloud bon marché : Si votre projet nécessite un vrai entraînement, il est fortement conseillé d'utiliser un service de GPU par le biais du cloud (comme Google Colab, RunPod ou Modal) pour quelques centimes/euros de l'heure plutôt que de bloquer votre ordinateur pendant des semaines sur un CPU.

Si vous le souhaitez, dites-moi :

* Quel type de données ou de texte vous souhaitez utiliser pour l'entraînement ?
* La configuration de votre ordinateur (quantité de RAM et type de processeur) ?

Je peux vous orienter vers une solution adaptée ou une alternative par fine-tuning léger.

[1] [https://www.youtube.com](https://www.youtube.com/watch?v=T9egZA5ppQw)
[2] [https://www.youtube.com](https://www.youtube.com/watch?v=93CgQ7dAQBU)
[3] [https://www.distillabs.ai](https://www.distillabs.ai/learn/fine-tune-llm-without-gpu/)
[4] [https://www.reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/1bq9mtb/how_do_i_play_with_llms_if_i_dont_have_a_gpu_at/)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=ckWULAbePmo)

---

# Entrainement sur du texte, modèle sur 1 ou 2 bits

> Je veux faire un entrainement sur du texte, avec un modele travaillant sur 1 ou 2 bits maximum

C'est un choix technique très pertinent. Travailler sur 1 ou 2 bits maximum correspond à l'architecture des 1-bit LLMs, popularisée par Microsoft avec BitNet b1.58. Ces modèles utilisent des poids ternaires (-1, 0, 1) qui ne requièrent que 1,58 bits par poids. [1, 2, 3] 
Le grand avantage de cette architecture est qu'elle remplace les lourdes multiplications matricielles par de simples additions et soustractions. C'est l'approche idéale pour faire du calcul efficace sur CPU. [2, 4] 
Cependant, il y a un piège fondamental à comprendre concernant l'entraînement :
## La règle d'or : On n'entraîne pas en 1 bit, on entraîne vers 1 bit
Durant l'entraînement, un CPU (ou GPU) ne sait pas calculer des gradients et modifier des variables sur seulement 1 ou 2 bits. L'apprentissage nécessite une précision mathématique fine.
On utilise donc le QAT (Quantization-Aware Training) : [3, 5] 

   1. Les poids "maîtres" du modèle restent stockés en haute précision (Float32 ou BFloat16) en arrière-plan. [4] 
   2. Lors de la phase de calcul (Forward Pass), une fonction (comme BitLinear) quantifie à la volée ces poids en 1,58 bits (-1, 0, 1) pour exécuter les calculs textuels. [1, 3] 
   3. La correction des erreurs (Backward Pass) met à jour les poids maîtres en haute précision. [5] 
   4. Une fois l'entraînement fini, on supprime les poids maîtres et on ne garde que le modèle final figé en 1,58 bits pour l'utiliser sur le processeur. [4, 6] 

------------------------------
## Code d'exemple minimal (PyTorch)
Voici comment implémenter une couche linéaire fonctionnant en 1 bit (via l'approche BitNet simplifiée) pour votre script d'entraînement sur CPU. Vous pouvez utiliser ce bloc pour remplacer le classique nn.Linear de votre réseau. [2] 

import torchimport torch.nn as nn
class BitLinear(nn.Module):
    """
    Couche Linéaire 1.58-bit (BitNet). 
    Quantifie les poids en (-1, 0, 1) à la volée durant l'entraînement.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Poids maîtres conservés en haute précision pour accumuler les gradients
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # 1. Quantisation des activations (sur 8 bits par exemple pour la stabilité)
        # On utilise une normalisation standard (RMSNorm ou par moyenne)
        x_norm = x - x.mean(dim=-1, keepdim=True)
        scale_x = 127.0 / (x_norm.abs().max(dim=-1, keepdim=True).values + 1e-5)
        x_quant = torch.clamp(torch.round(x_norm * scale_x), -128, 127)

        # 2. Quantisation des poids en 1.58 bits (-1, 0, 1)
        scale_w = self.weight.abs().mean()
        # "Straight-Through Estimator" (STE) pour permettre au gradient de passer à travers l'arrondi
        weight_scaled = self.weight / (scale_w + 1e-5)
        weight_quant = torch.clamp(torch.round(weight_scaled), -1, 1)
        weight_final = weight_quant + (self.weight - self.weight.detach()) # Astuce STE PyTorch

        # 3. Opération linéaire (qui n'utilise graphiquement que des additions/soustractions)
        # Remarque : Sur PyTorch CPU standard, cela reste émulé en flottants.
        out = nn.functional.linear(x_quant, weight_final)
        
        # Redimensionnement pour correspondre aux échelles d'origine
        return out / (scale_x + 1e-5)

## Comment procéder pour votre projet ?
Pour éviter de réécrire tout un modèle de type Transformer depuis zéro, vous pouvez vous appuyer sur des projets open source existants conçus exactement pour cela :

   1. Utiliser des implémentations communautaires clés en main : Le dépôt GitHub [kyegomez/BitNet](https://github.com/kyegomez/BitNet) fournit une structure PyTorch complète avec un script train.py déjà configuré pour s'entraîner sur un corpus de texte. [7, 8] 
   2. Créer un tout petit jeu de données : Pour un CPU, préparez un fichier texte propre (par exemple quelques Mo de romans du domaine public ou vos propres données spécialisées). [9] 
   3. Lancer l'entraînement : En forçant PyTorch sur votre processeur (device = "cpu"), le modèle va converger petit à petit. Comme le modèle intègre l'astuce de la formule mathématique ci-dessus, les poids vont naturellement se regrouper autour des valeurs -1, 0, 1 au fil de l'apprentissage. [3, 8, 10] 

Quelques questions rapides si vous avez le temps :

* Cette explication du QAT était-elle claire ?
* Souhaitez-vous un script d'entraînement complet ?


[1] [https://github.com](https://github.com/dhakalnirajan/LLaMA-BitNet)
[2] [https://www.jmlr.org](http://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf)
[3] [https://huggingface.co](https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5)
[4] [https://www.youtube.com](https://www.youtube.com/watch?v=vkQJ2lJzjKY)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=MXWlB9nDAFU)
[6] [https://github.com](https://github.com/tiiuae/onebitllms)
[7] [https://github.com](https://github.com/kyegomez/BitNet)
[8] [https://github.com](https://github.com/kyegomez/BitNet/blob/main/train.py)
[9] [https://www.youtube.com](https://www.youtube.com/watch?v=pRe_bq2AElE&t=9)
[10] [https://www.youtube.com](https://www.youtube.com/watch?v=tO8aprFGb54)

---

# Se passer de pytorch

> Comment se passer de pytorch, autrement dit, reimplementer soi-même la boucle d'entrainement ?

Pour se passer entièrement de PyTorch, vous devez réimplémenter les deux piliers du Deep Learning en Python pur (ou idéalement avec NumPy pour que les calculs de matrices ne prennent pas des siècles sur votre CPU) : [1] 

   1. Le calcul matriciel (Forward pass).
   2. La rétropropagation (Backward pass) et la descente de gradient. [2] 

Voici comment concevoir de A à Z une couche linéaire de 1.58-bit (BitNet) et sa boucle d'entraînement sans aucun framework, en utilisant NumPy. [3] 
## 1. Le Piège de l'optimisation en 1 bit sans framework
Pour rappel, vous devez conserver deux versions des poids en mémoire : des poids maîtres réels (flottants) pour accumuler précisément les micro-changements du gradient, et des poids ternaires (-1, 0, 1) pour la prédiction. [3, 4] 
En n'utilisant pas PyTorch, vous devez calculer la dérivée de la quantification vous-même. On utilise l'astuce du STE (Straight-Through Estimator) : pendant la rétropropagation, on fait comme si l'opération d'arrondi n'existait pas, et on transmet le gradient directement aux poids flottants. [2, 5] 
------------------------------
## 2. Le Code Complet : Modèle 1-bit et Boucle d'entraînement (NumPy)
Voici l'implémentation d'une couche réseau, d'une fonction de perte (Loss) et de la boucle d'apprentissage d'un mini-réseau de texte. [1] 

import numpy as np
# 1. DÉFINITION DE LA COUCHE 1.58-BIT (BitNet)class TinyBitLinear:
    def __init__(self, in_features, out_features, lr=0.01):
        self.lr = lr
        # Poids maîtres en haute précision (flottants)
        self.W_FP = np.random.randn(out_features, in_features) * 0.1
        self.b = np.zeros((out_features, 1))
        
        # Variables pour stocker les états requis pour le backward pass
        self.x = None
        self.W_quant = None

    def forward(self, x):
        """
        x: matrice d'entrée de taille (in_features, batch_size)
        """
        self.x = x
        
        # Quantifier les poids en -1, 0, 1 (BitNet b1.58)
        scale_w = np.mean(np.abs(self.W_FP)) + 1e-5
        self.W_quant = np.clip(np.round(self.W_FP / scale_w), -1, 1)
        
        # Calcul de la sortie : Y = W_quant * X + b
        # (Sur un vrai processeur basse consommation dédié, cette étape n'utiliserait pas de multiplication)
        out = np.dot(self.W_quant, x) + self.b
        return out

    def backward(self, dout):
        """
        dout: gradient provenant de la couche suivante (out_features, batch_size)
        """
        batch_size = dout.shape[1]
        
        # Gradients par rapport aux poids et aux biais
        dW = np.dot(dout, self.x.T) / batch_size
        db = np.sum(dout, axis=1, keepdims=True) / batch_size
        
        # Gradient par rapport à l'entrée X (Utile si vous enchaînez plusieurs couches)
        dx = np.dot(self.W_quant.T, dout)
        
        # MISE À JOUR DES PARAMÈTRES (Descente de gradient)
        # On applique le gradient dW (calculé sur les poids 1-bit) directement sur les poids Master FP (STE)
        self.W_FP -= self.lr * dW
        self.b -= self.lr * db
        
        return dx
# 2. FONCTION DE PERTE (Softmax + Cross Entropy pour la prédiction de texte)def cross_entropy_loss(logits, targets):
    """
    logits: Sortie brute du modèle (vocab_size, batch_size)
    targets: Indices des vrais mots attendus (batch_size,)
    """
    batch_size = logits.shape[1]
    
    # Softmax stable pour éviter les débordements numériques
    shift_logits = logits - np.max(logits, axis=0, keepdims=True)
    exps = np.exp(shift_logits)
    probs = exps / np.sum(exps, axis=0, keepdims=True)
    
    # Calcul de la Loss (Cross-Entropy)
    loss = -np.log(probs[targets, np.arange(batch_size)] + 1e-15)
    mean_loss = np.mean(loss)
    
    # Calcul du gradient initial (dout) pour la rétropropagation
    dout = probs.copy()
    dout[targets, np.arange(batch_size)] -= 1.0
    
    return mean_loss, dout
# 3. DONNÉES DE SIMULATION (Un mini vocabulaire de texte)# Disons que notre modèle doit prédire le mot suivant parmi 4 mots possiblesvocab = {"le": 0, "chat": 1, "mange": 2, "souris": 3}# Exemple de données (X -> Y) : "le" -> "chat", "chat" -> "mange", etc.X_train = np.array([[1, 0, 0, 0],  # "le" (One-hot)
                    [0, 1, 0, 0],  # "chat"
                    [0, 0, 1, 0]]).T # (taille: 4, 3)y_train = np.array([1, 2, 3])      # "chat" (1), "mange" (2), "souris" (3)
# 4. LA BOUCLE D'ENTRAÎNEMENT 100% MAISON
np.random.seed(42)model_layer = TinyBitLinear(in_features=4, out_features=4, lr=0.5)

print("--- Début de l'entraînement sans framework ---")for epoch in range(500):
    # 1. Forward Pass
    logits = model_layer.forward(X_train)
    
    # 2. Calcul de l'erreur
    loss, dout = cross_entropy_loss(logits, y_train)
    
    # 3. Backward Pass (Met à jour les poids en interne)
    model_layer.backward(dout)
    
    if epoch % 100 == 0:
        print(f"Époque {epoch} | Perte (Loss): {loss:.4f}")
# 5. VÉRIFICATION DU RÉSULTAT FINAL (Modèle converti définitivement en 1.58-bit)
print("\n--- Entraînement terminé ---")scale_w = np.mean(np.abs(model_layer.W_FP))final_1bit_weights = np.clip(np.round(model_layer.W_FP / scale_w), -1, 1)

print("Poids finaux du modèle (strictement restreints à -1, 0, 1) :\n", final_1bit_weights)

------------------------------
## Évaluation des choix techniques (Avantages & Inconvénients)

* 
* Le grand gain : Vous contrôlez l'allocation mémoire au bit près. En stockant ces matrices NumPy sous forme d'entiers (np.int8), le modèle ne pèse presque rien en RAM et s'exécute très vite, même sur un vieux processeur.
* La limite : NumPy n'a pas d'instructions natives pour exécuter des additions pures à la place des multiplications pour les matrices ternaires. Pour exploiter réellement la vitesse théorique du 1-bit sur CPU, il faudrait écrire la fonction forward en C++ ou en Rust (en utilisant des opérations de masquage binaire "Bit Manipulation").
* 

Si vous souhaitez aller plus loin, dites-moi :

* 
* Souhaitez-vous que nous ajoutions une couche d'Attention (le cœur d'un LLM) à cette structure NumPy ?
* Ou préférez-vous voir comment exporter ces poids appris vers un script en C++ pour maximiser la vitesse sur votre CPU ?
* 


[1] [https://medium.com](https://medium.com/@murmuarpan530/building-a-neural-network-from-scratch-in-python-e78203471870)
[2] [https://www.youtube.com](https://www.youtube.com/watch?v=MQzG1hfhow4&t=1836)
[3] [https://medium.com](https://medium.com/@kondwani0099/reimagining-ai-efficiency-a-practical-guide-to-using-bitnets-1-bit-llm-on-cpus-without-ef804d3fb875)
[4] [https://www.youtube.com](https://www.youtube.com/watch?v=w-Htc7sK5AU&t=911)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=iE1lccrHfok&t=5)

---

# Coder en C pur

> En fait j'aimerais tout faire en c.

Réimplémenter un réseau de neurones binaire/ternaire 100% en C (sans aucune bibliothèque externe, pas même pour les matrices) est un excellent exercice. C’est exactement ce que font des projets d'ingénierie comme [bitnet.c de kevin-pek](https://github.com/kevin-pek/bitnet.c) ou bitnet.cpp de Microsoft pour s'exécuter de manière ultra-rapide sur CPU. [1, 2] 
Voici un programme complet, valide, en C pur (C99). Il implémente la couche linéaire 1.58-bit (poids maîtres en float pour le gradient, quantification à la volée en -1, 0, 1, technique du STE) ainsi qu'une boucle d'entraînement par descente de gradient avec une fonction de perte Cross-Entropy. [3, 4, 5] 
## Le Code Complet en C (main.c)

#include <stdio.h>#include <stdlib.h>#include <math.h>
#define INPUT_DIM 4#define OUTPUT_DIM 4#define BATCH_SIZE 3#define EPOCHS 500#define LEARNING_RATE 0.5f
// Structure de la couche 1.58-bittypedef struct {
    float W_FP[OUTPUT_DIM][INPUT_DIM]; // Poids maîtres (haute précision)
    float b[OUTPUT_DIM];               // Biais
    float W_quant[OUTPUT_DIM][INPUT_DIM]; // Stockage des poids quantifiés (-1, 0, 1)
    
    // Buffers pour le backward pass
    float cached_x[INPUT_DIM][BATCH_SIZE];
} BitLinearLayer;
// Initialisation aléatoire des poidsvoid init_layer(BitLinearLayer *layer) {
    for (int i = 0; i < OUTPUT_DIM; i++) {
        layer->b[i] = 0.0f;
        for (int j = 0; j < INPUT_DIM; j++) {
            // Initialisation normalisée de type Xavier / Glorot
            layer->W_FP[i][j] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * sqrtf(2.0f / INPUT_DIM);
        }
    }
}
// Phase Forwardvoid forward(BitLinearLayer *layer, float x[INPUT_DIM][BATCH_SIZE], float out[OUTPUT_DIM][BATCH_SIZE]) {
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
// Phase Backward & Mise à jour (Optimiseur SGD intégré via le Straight-Through Estimator)void backward_and_update(BitLinearLayer *layer, float dout[OUTPUT_DIM][BATCH_SIZE]) {
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
// Calcul de la Loss (Softmax + Cross Entropy) et génération du gradient initial (dout)float compute_loss_and_dout(float logits[OUTPUT_DIM][BATCH_SIZE], int targets[BATCH_SIZE], float dout[OUTPUT_DIM][BATCH_SIZE]) {
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

## Comment compiler et exécuter ce code ?
Ouvrez un terminal et utilisez votre compilateur C favori (comme gcc ou clang) :

# Compilation en liant la bibliothèque mathématique (-lm)
gcc -O3 main.c -o train_1bit -lm
# Exécution
./train_1bit

## Ce qui change radicalement en C

   1. La gestion de la mémoire à la main : Contrairement à Python ou PyTorch qui créent des tenseurs dynamiques à chaque ligne de calcul, ici toutes les structures ont une taille fixe connue à la compilation (float logits[OUTPUT_DIM][BATCH_SIZE]). Le CPU n'a aucune surcharge liée à l'allocation de mémoire.
   2. Le coût réel des opérations : Dans la fonction forward, la ligne out[i][b] += layer->W_quant[i][j] * x[j][b]; effectue une multiplication de flottants. Cependant, comme W_quant ne vaut que -1, 0, ou 1, si vous poussiez l'implémentation en assembleur ou avec des intrinsèques SIMD (comme AVX2), vous pourriez remplacer cette ligne par une simple instruction conditionnelle d'addition / soustraction, décuplant la vitesse sur CPU.

Souhaitez-vous que l'on optimise le code avec du Bit-Packing (compacter plusieurs poids 1-bit dans un seul entier int pour réduire la taille en mémoire) ou préfèrez-vous que nous ajoutions le traitement d'un vrai fichier texte externe ?

[1] [https://github.com](https://github.com/kevin-pek/bitnet.c)
[2] [https://github.com](https://github.com/microsoft/BitNet/blob/main/README.md)
[3] [https://bitnet.xin](https://bitnet.xin/blog/bitnet-b158-architecture-decoding-the-1-bit-llm-breakthrough)
[4] [https://medium.com](https://medium.com/@nimritakoul01/1-bit-llm-and-the-1-58-bit-llm-the-magic-of-model-quantization-ee47697c561a)
[5] [https://medium.com](https://medium.com/analytics-vidhya/building-neural-network-framework-in-c-using-backpropagation-8ad589a0752d)

---

# THE END
