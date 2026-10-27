---
source: https://claude.ai/chat/de633541-7013-4f77-b76c-822ba536700f
---

# Mécanisme d'attention dans les LLMs

> Bonjour, je m'interesse aux llm. Parle-moi de l'attention. Pour le code, utilise le language C avec juste la libc et libm.

L'attention permet à chaque token d'une séquence de « regarder » les autres tokens pour décider quelle information récupérer.

**Le principe.** Chaque token produit trois vecteurs : une requête **Q** (ce que je cherche), une clé **K** (ce que je propose) et une valeur **V** (le contenu que je transmets). Pour un token donné :

1. On calcule le produit scalaire de sa requête avec toutes les clés, ce qui donne des scores de similarité.
2. On divise par √d pour éviter que les scores explosent.
3. Un softmax transforme les scores en poids qui somment à 1.
4. La sortie est la moyenne pondérée des valeurs V.

En formule : `Attention(Q,K,V) = softmax(QKᵀ / √d) · V`

**Dans un LLM.** Le masque causal interdit à un token de voir le futur (seulement j ≤ t), ce qui permet de prédire le token suivant. On utilise aussi plusieurs têtes en parallèle (multi-head), chacune apprenant à repérer un type de relation différent (syntaxe, coréférence, etc.), puis on concatène leurs sorties. Le coût est quadratique en longueur de séquence, d'où les optimisations comme le KV cache et FlashAttention.

J'ai écrit une implémentation en C pur (libc + libm), compilée et testée : attention.c. Elle fait de l'attention causale à une tête, avec une petite démo en `main`. Pour compiler : `gcc -O2 attention.c -o attention -lm`.

Je peux ensuite ajouter le multi-head, les projections Wq/Wk/Wv ou le KV cache si ça vous intéresse.

# Attention multi-têtes

> Commence par le multi-head

Voici multihead.c, compilé et testé.

**L'idée.** Au lieu d'une seule attention sur un vecteur de dimension `DM`, on découpe ce vecteur en `H` tranches de taille `DH = DM / H`. Chaque tête fait sa propre attention sur sa tranche, avec ses propres poids softmax. Puis on concatène les sorties pour retrouver un vecteur de dimension `DM`.

**Dans le code.** La tête `h` lit les colonnes `[h*DH, (h+1)*DH)` de Q, K et V, et écrit son résultat aux mêmes colonnes de `out`. La concaténation se fait donc toute seule, sans copie. Dans la sortie de la démo, la barre `|` sépare les deux têtes : la tête 0 reproduit exactement le résultat du fichier précédent, tandis que la tête 1 produit des poids différents puisqu'elle voit d'autres données.

**Ce qui manque encore** pour une vraie couche : les projections apprises `Wq`, `Wk`, `Wv` qui produisent Q, K, V à partir de l'entrée, et la projection de sortie `Wo` appliquée après la concaténation. Ce sont de simples multiplications matrice-vecteur, et c'est ce qui permet à chaque tête de se spécialiser.

On continue avec ces projections ?

# Couche d'attention multi-têtes complète

> Oui

Voici mha_layer.c, compilé et testé : c'est maintenant une couche d'attention multi-têtes complète.

**Le flux de données :**
1. L'entrée `X` (`T × DM`) est projetée trois fois : `Q = X·Wq`, `K = X·Wk`, `V = X·Wv`.
2. Chaque tête fait son attention causale sur sa tranche de colonnes (comme avant).
3. Les sorties des têtes sont concaténées, puis projetées par `Wo`, ce qui laisse les têtes « se mélanger » et redonne un vecteur de dimension `DM`.

**Ce qui change par rapport au fichier précédent.** Q, K et V ne sont plus donnés à la main : ils sont calculés à partir de `X`. Ce sont les matrices `Wq`, `Wk`, `Wv`, `Wo` que l'entraînement ajuste. Ici elles sont remplies avec un petit générateur pseudo-aléatoire déterministe (`rnd()`), mis à l'échelle par `1/√DM`. Les sorties de la démo n'ont donc pas de signification sémantique : le but est de vérifier que la mécanique est correcte.

**Pour la suite**, les deux étapes naturelles sont le KV cache (pour la génération token par token, sans recalculer K et V du passé) ou l'assemblage d'un bloc transformer complet avec connexion résiduelle, LayerNorm et MLP. Laquelle vous tente ?

# Cache KV

> Montre-moi le kv cache

Voici kvcache.c, compilé et testé. Il contient deux versions de la même couche, que la démo compare.

**Le problème.** En génération, on produit un token à la fois. Sans cache, chaque nouveau token oblige à recalculer K et V pour tout le préfixe. C'est ce que fait `mha_naive`, avec un coût qui croît à chaque pas.

**L'astuce.** Grâce au masque causal, les K et V d'un token ne dépendent que de lui-même et de ce qui le précède, donc ils ne changent jamais une fois calculés. On les stocke dans `KVCache`. À chaque pas, `mha_step` ne projette que le **nouveau** token : il calcule sa requête `q`, ajoute son `k` et son `v` au cache, puis fait l'attention de `q` contre tout le cache. Le Q des anciens tokens n'est jamais réutilisé, d'où le nom « KV cache » et pas « QKV cache ».

**Vérification.** La démo exécute les deux versions pas à pas et affiche l'écart maximal entre leurs sorties : 0, les résultats sont identiques. Le cache change donc le coût, pas le résultat.

**Le compromis.** Le calcul par token passe de linéaire à constant en projections (l'attention reste linéaire en longueur du cache), mais la mémoire grandit avec la séquence : `2 × couches × T × DM` flottants. C'est pourquoi, sur les vrais LLM, le KV cache est souvent le facteur limitant en contexte long, et pourquoi on utilise des variantes comme le multi-query ou le grouped-query attention pour le réduire.

Voulez-vous qu'on assemble ensuite le bloc transformer complet (résiduel, LayerNorm, MLP) ?

# Bloc transformeur complet

> Oui s'il-te-plait.

Voici transformer_block.c, compilé sans avertissement et testé. Il assemble tout ce qu'on a construit jusqu'ici.

**Le bloc (variante « pre-norm », celle des LLM modernes) :**
1. `x ← x + Attention(LayerNorm(x))`
2. `x ← x + MLP(LayerNorm(x))`

**Les nouvelles briques :**
- **LayerNorm** : centre et réduit chaque vecteur (moyenne 0, variance 1), puis applique les paramètres appris `gamma` et `beta`. Elle stabilise l'entraînement.
- **Connexions résiduelles** : chaque sous-couche *ajoute* sa contribution à `x` au lieu de le remplacer. Les gradients circulent ainsi directement à travers le réseau, ce qui permet d'empiler des dizaines de blocs.
- **MLP** : deux couches `DM → 4·DM → DM` avec activation GELU. Si l'attention sert à *échanger* de l'information entre tokens, le MLP la *transforme* token par token, et c'est là que se trouve la majorité des paramètres.

**Dans le code**, `block_step` traite un token à la fois en réutilisant le KV cache de l'étape précédente, et modifie `x` en place. Un vrai modèle empile `N` blocs, chacun avec son propre cache, et ajoute en entrée les embeddings de tokens avec les positions (par exemple RoPE), et en sortie une LayerNorm finale suivie d'une projection vers le vocabulaire.

Souhaitez-vous qu'on termine avec ces dernières pièces (empilement de blocs, RoPE, sortie vers le vocabulaire et échantillonnage) ?

# MLP, GELU ?

> Avant ça, redonne-moi la signification des accronymes MLP et GELU

**MLP** : *Multi-Layer Perceptron* (perceptron multicouche). C'est un réseau de neurones classique composé de couches linéaires successives séparées par des non-linéarités. Dans un transformer, on l'appelle aussi parfois *feed-forward network* (FFN).

**GELU** : *Gaussian Error Linear Unit*. C'est une fonction d'activation qui multiplie l'entrée `x` par la probabilité qu'une variable gaussienne standard soit inférieure à `x`, soit `x · Φ(x)`. Elle ressemble à ReLU, mais en version lisse : les valeurs très négatives sont écrasées vers 0 et les valeurs positives passent presque inchangées, avec une transition douce autour de 0. Dans le code, j'ai utilisé l'approximation avec `tanh`, plus facile à calculer qu'avec la fonction d'erreur.

# Modèle complet

> Ok. Maintenant on peut terminer avec les derniers élements

Voici mini_llm.c, compilé sans avertissement et testé : un petit modèle de langage complet, de l'identifiant de token jusqu'au token suivant.

**Les pièces ajoutées :**
- **Embeddings** : une table `emb[VOCAB][DM]` associe chaque identifiant de token à un vecteur. On la réutilise en sortie (*weight tying*), ce qui économise des paramètres.
- **Pile de blocs** : `NL` blocs successifs, chacun avec son propre KV cache.
- **RoPE** (*Rotary Position Embedding*) : on fait tourner chaque paire de coordonnées de `q` et `k` d'un angle proportionnel à la position. Le produit scalaire `q·k` ne dépend alors que de l'**écart** entre les positions, pas de leurs valeurs absolues. La démo le vérifie : positions 5/3 et 9/7 donnent exactement le même résultat (0.019823). Seuls `q` et `k` sont tournés, et `k` est mis en cache déjà tourné.
- **Sortie** : LayerNorm finale, puis produit scalaire avec chaque embedding pour obtenir un score (*logit*) par token du vocabulaire.
- **Échantillonnage** : température 0 donne l'argmax (glouton). Sinon on garde les `k` meilleurs logits, on les divise par la température, on applique le softmax et on tire au hasard.

**La boucle de génération** se fait en deux phases : le *préremplissage* passe le prompt dans le modèle pour remplir les caches, puis chaque token choisi est réinjecté pour obtenir le suivant.

**Un point important :** les poids sont aléatoires, donc les suites de tokens n'ont aucun sens. Le mode glouton boucle sur le token 17, un comportement classique d'un modèle non entraîné. Tout le mécanisme d'inférence est là, mais il manque l'entraînement (rétropropagation, optimiseur, données) et un tokenizer comme le BPE. Pour charger de vrais poids, il faudrait aussi lire un format de fichier et agrandir les dimensions, mais la structure reste la même.

Voulez-vous qu'on aborde l'entraînement, ou le tokenizer BPE ?


