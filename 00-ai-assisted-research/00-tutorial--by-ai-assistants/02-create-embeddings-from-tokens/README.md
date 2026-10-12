---
source: https://claude.ai/chat/b9ac634f-5157-426d-899c-e901963744a3
---

# Créer des embeddings à partir de tokens

> Bonjour, je m'interesse au fonctionnement des LLM. Explique moi comment on fabrique les embeddings, après l'étape de tokenization. Pour l'exemple de code, je veux du C standard, avec seulement la libc

Après la tokenization, chaque token est un simple entier (son ID). L'embedding consiste à :

1. **Lookup** : une matrice apprise `E` de taille `vocab_size × d_model` contient un vecteur par token, et l'ID sert d'indice de ligne. Au départ, la matrice est initialisée aléatoirement, puis ajustée par rétropropagation pendant l'entraînement.
2. **Position** : l'ordre des tokens n'est pas connu du lookup. On l'injecte soit par addition d'un vecteur de position (sinusoïdal ou appris, comme dans le Transformer d'origine), soit, dans les modèles récents, par RoPE, qui applique une rotation aux vecteurs Q/K dans l'attention.

Voici un exemple en C standard (libc + libm pour `sin`, `cos`, `sqrt`, `log`) :Le code compile et tourne : le token 7 apparaît aux positions 1 et 3 avec la même ligne de `E`, mais les vecteurs finaux diffèrent grâce à l'encodage positionnel.

Deux précisions : le programme demande `-lm` à la compilation (la libm fait partie de la libc standard sous Linux mais est liée séparément), et les valeurs de `E` sont ici aléatoires, alors que dans un vrai LLM elles sont apprises. Je peux détailler RoPE ou le calcul du gradient sur la table d'embeddings si ça t'intéresse.

# RoPE et calcul du gradient

> Explique moi RoPE et le calcul du gradient.

## RoPE (Rotary Position Embedding)

Au lieu d'**ajouter** un vecteur de position à l'embedding, RoPE **fait tourner** les vecteurs Q et K dans l'attention, à chaque couche. On découpe un vecteur en paires de dimensions `(x₂ᵢ, x₂ᵢ₊₁)`, vues comme des points d'un plan 2D, et on tourne chaque paire d'un angle proportionnel à la position `m` :

```
x'₂ᵢ   = x₂ᵢ·cos(m·θᵢ) − x₂ᵢ₊₁·sin(m·θᵢ)
x'₂ᵢ₊₁ = x₂ᵢ·sin(m·θᵢ) + x₂ᵢ₊₁·cos(m·θᵢ)
avec θᵢ = 10000^(−2i/d)
```

L'intérêt vient du produit scalaire de l'attention. Comme une rotation transposée composée avec une autre rotation donne une rotation de la différence d'angles, `q(m)·k(n)` ne dépend que de `n − m`. L'attention voit donc directement la **distance relative** entre tokens, sans table de positions apprise. Les premières paires tournent vite (positions proches), les dernières lentement (longue portée). V n'est pas touché, et l'embedding d'entrée reste un simple lookup.

## Gradient sur la table d'embeddings

Le lookup `y = E[id]` équivaut à `onehot(id) × E`. Le backward est donc très simple : si `∂L/∂y` est le gradient arrivant de la couche suivante,

```
∂L/∂E[id] += ∂L/∂y        (pour la ligne utilisée)
∂L/∂E[j]   = 0            (pour toutes les autres lignes)
```

Le gradient est donc **creux** : seules les lignes des tokens présents dans le batch sont mises à jour. Si un token apparaît plusieurs fois, ses gradients **s'accumulent** (`+=`). Ensuite, SGD fait simplement `E[id] -= lr · grad`. Adam, lui, garde aussi des moments par ligne.

Points utiles :
- L'encodage sinusoïdal est fixe, donc sans gradient. Avec des positions apprises, ce serait le même mécanisme que pour `E`.
- Pour RoPE, il n'y a pas de paramètre à apprendre, mais le backward doit traverser la rotation : le gradient subit la rotation **inverse** (`−m·θᵢ`).
- Si les poids d'entrée et de sortie sont partagés (*weight tying*), chaque ligne de `E` reçoit aussi un gradient de la projection finale vers le vocabulaire, donc **toutes** les lignes sont mises à jour, plus seulement celles du batch.

Je peux ajouter à l'exemple C un backward + une étape SGD, ou une version de RoPE, si tu veux.

# implémentation de RoPE

> Ajoute une version rope

J'ai ajouté `rope.c`, qui compile et s'exécute correctement. Il montre trois choses :

- **Lookup sans position** : l'embedding d'entrée reste une simple ligne de `E`, et RoPE s'applique ensuite aux vecteurs q/k (ici, `x` joue directement ce rôle pour simplifier).
- **Propriété relative** : pour une distance fixe de 3, le score `q(m)·k(n)` est identique (−0.05342) en `m=0, 2, 10, 100`. Seule la différence `n − m` compte.
- **Backward** : appliquer la rotation inverse (`−pos`) au gradient restitue le vecteur de départ, avec une erreur d'environ 1e-7 (arrondi flottant).

Dans un vrai modèle, q et k viennent de projections `x·Wq` et `x·Wk`, et RoPE est appliqué à chaque couche, par tête d'attention.

# Des questions

> Après relecture de la conversation, j'ai quelques questions : 
> 1- je comprends qu'une dimension de la matrice E est la taille du dictionnaire de tokens. Comment est choisie l'autre dimension ?
> 2- pourquoi la position (absolue ou relative ?) d'un token doit être injectée, via sinusoide ou rope ?
> 3- rappelle-moi ce que sont q et k. "Query" et "Key" ? J'ai un peu oublié.

## 1. Comment choisir `d_model` ?

C'est un **hyperparamètre**, fixé avant l'entraînement : il n'y a pas de formule, c'est un compromis entre capacité et coût. Plus `d_model` est grand, plus chaque token peut encoder de nuances, mais plus le modèle est lourd en mémoire et en calcul (il intervient dans presque toutes les matrices du réseau).

Ordres de grandeur : 512 pour le Transformer d'origine, 768 pour GPT-2 small, 4096 pour les modèles de classe 7B, 12 000 et plus pour les très gros. On le choisit généralement en même temps que le nombre de couches et de têtes d'attention (`d_model` doit être divisible par le nombre de têtes), en suivant des lois d'échelle empiriques et le budget de calcul.

## 2. Pourquoi injecter la position ?

Parce que l'attention est **invariante à l'ordre** : elle compare des tokens deux à deux, sans notion de « avant » ou « après ». Sans information de position, « le chien mord l'homme » et « l'homme mord le chien » donneraient les mêmes représentations (à permutation près), alors que le sens est opposé.

Les deux approches se distinguent ainsi :
- **Sinusoïdal / appris** : position **absolue** (« je suis le token n° 5 »), ajoutée à l'embedding.
- **RoPE** : encodage **absolu** à l'entrée (on tourne selon `m`), mais qui produit un effet **relatif** dans le score d'attention, car celui-ci ne dépend que de `n − m`. C'est ce qui rend RoPE agréable : le modèle raisonne en distance, ce qui généralise mieux aux contextes longs.

## 3. Q et K

Oui, **Query** (requête) et **Key** (clé), plus **Value** (valeur). Pour chaque token `x`, trois projections linéaires apprises :

```
q = x·Wq    k = x·Wk    v = x·Wv
```

L'analogie est celle d'une recherche dans une base : chaque token émet une **requête** (« qu'est-ce que je cherche comme information ? »), et expose une **clé** (« voici ce que je contiens ») et une **valeur** (« voici l'information que je transmets »). Le score `q·k` mesure la pertinence entre deux tokens ; après softmax, il donne des poids avec lesquels on fait la moyenne des `v` :

```
sortie = softmax(Q·Kᵀ / √d) · V
```

C'est pour ça que RoPE s'applique à Q et K (qui entrent dans le score), mais pas à V (qui porte seulement l'information transmise).


