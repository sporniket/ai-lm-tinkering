---
source: https://claude.ai/chat/5f14e8e9-4d2a-4deb-83ed-e3301115762f
---

# Les concepts du LLM

> Je m'intéresse aux LLM et en particulier leur construction, et j'essaye de comprendre les concepts.

Super sujet ! Pour vous donner une carte du territoire, voici les grandes étapes de la construction d'un LLM, de la plus basique à la plus avancée.

**1. Les données et la tokenisation.** Le texte est découpé en « tokens » (morceaux de mots, via des algorithmes comme le BPE), puis chaque token devient un identifiant numérique. Le modèle ne voit jamais de lettres, seulement des nombres.

**2. Les embeddings.** Chaque token est converti en un vecteur de nombres, une position dans un espace à plusieurs milliers de dimensions. Des mots proches par le sens se retrouvent à des endroits proches, mais cette structure est apprise, pas programmée.

**3. L'architecture Transformer.** C'est le cœur du système. Son mécanisme clé, l'*attention*, permet à chaque token de « regarder » les autres tokens du contexte et de pondérer leur importance pour mieux se représenter. On empile des dizaines de couches, chacune combinant attention et petits réseaux de neurones (MLP).

**4. Le pré-entraînement.** Le modèle apprend une tâche très simple : prédire le token suivant, sur des milliers de milliards de tokens. Les paramètres (les « poids ») sont ajustés par descente de gradient pour réduire l'erreur de prédiction. C'est là que se forment l'essentiel des connaissances et des capacités du modèle.

**5. Le post-entraînement.** Un modèle pré-entraîné complète du texte mais ne se comporte pas comme un assistant. On l'affine donc avec du fine-tuning supervisé sur des exemples de bonnes réponses, puis avec de l'apprentissage par renforcement (à partir de retours humains ou d'autres signaux) pour le rendre utile, honnête et sûr.

**6. L'inférence.** À l'usage, le modèle génère une réponse token par token, en échantillonnant parmi les prédictions les plus probables (paramètres comme la température).

Ce qui est souvent le plus déroutant au début, c'est que tout ceci repose sur des idées simples répétées à très grande échelle : les capacités « émergent » de la taille du modèle, des données et du calcul (les *scaling laws*).

Pour vous aider plus précisément : y a-t-il une étape où vous sentez que ça coince, ou préférez-vous qu'on commence par le mécanisme d'attention, qui est généralement le plus difficile à saisir ?

---

# Tokenisation

> Parlons d'abord de la tokenisation. Des exemples que j'ai vu, j'ai intégré qu'on transforme donc un texte en entrée en flux de tokens. Par exemple, une représentation décodée de "hello world" pourrait donner "<control:start><syllabus:hel><syllabus:lo><control:whitespace><syllabus:wor><syllabus:ld><control:end>".
> Chaque token est identifié par une clé (un id numérique), de sorte que le flux de token est un flux d'identifiants de tokens. Par exemple, la séquence représentée précédemment sera en faite une séquence de nombre, mettons : 1,10112, 20234,2,55213,3045,0.

Vous avez bien compris le principe central : texte → découpage en morceaux → identifiants numériques. Votre exemple est pédagogique, avec quelques nuances qui valent la peine d'être corrigées pour que votre modèle mental colle à la réalité.

**Ce qui est exact.** La séquence d'ids est bien la seule chose que le modèle manipule. Il existe un vocabulaire fixe (typiquement 30 000 à 200 000 entrées) qui associe chaque morceau de texte à un id, et un décodeur qui fait le chemin inverse. Les tokens spéciaux de début et de fin de séquence existent aussi (souvent notés `<s>`, `</s>`, ou `<|endoftext|>` selon les modèles).

**Les nuances.**

- **Pas de « syllabes ».** Les tokens ne correspondent pas à des syllabes linguistiques. Ce sont des fragments appris statistiquement : avec le BPE (Byte Pair Encoding), on part des caractères ou des octets, puis on fusionne itérativement les paires les plus fréquentes dans un grand corpus. Un token peut être un mot entier courant (« the »), un fragment (« ld »), un seul caractère, ou même un octet isolé. « hello » est d'ailleurs très probablement un seul token dans la plupart des vocabulaires.

- **L'espace fait partie du token.** Il n'y a généralement pas de token « whitespace » séparé. L'espace est collé au début du mot suivant : on aura par exemple `hello` puis ` world` (avec l'espace devant). C'est pourquoi « world » en début de phrase et « ␣world » au milieu d'une phrase ont des ids différents. Selon les tokeniseurs, l'espace est représenté par un symbole spécial (`Ġ` en GPT-2, `▁` en SentencePiece).

- **Le vocabulaire est construit avant l'entraînement du modèle.** Le tokeniseur est entraîné à part, sur un corpus, puis figé. Les ids n'ont aucune signification en eux-mêmes : 10112 n'est pas « proche » de 10113. Le sens n'apparaît qu'à l'étape suivante, avec les embeddings.

Un exemple plus réaliste serait donc : `["hello", " world"]` → `[15339, 1917]` (ids fictifs mais plausibles), avec éventuellement un token de début de séquence devant.

Cela explique plusieurs comportements surprenants des LLM : leurs difficultés à compter les lettres d'un mot (ils ne voient pas les lettres), à gérer certaines langues moins bien représentées dans le vocabulaire (davantage de tokens par mot, donc plus coûteux), ou à manipuler des nombres.

Souhaitez-vous qu'on creuse le fonctionnement du BPE sur un petit exemple pas à pas, ou qu'on passe à l'étape suivante, celle où ces ids deviennent des vecteurs ?

---

# Syllabes ?

> Merci pour les explications. Je comprends que les LLM existants ont construit leur espace de token avec une analyse statistique. Et je pense que rien ne m'empêcherait de décider d'utiliser les syllabes et d'avoir un token spécifique pour représenter une espace pour construire mon espace de token. À moins que cela rende impossible le fonctionnement d'un LLM basé sur un tel espace de tokens ?

Vous avez raison : rien ne l'empêche, et un LLM fonctionnerait avec un tel vocabulaire. Le Transformer ne sait pas ce qu'est un token, il manipule simplement des ids d'un vocabulaire de taille fixe. La tokenisation est un choix de conception, pas une contrainte du modèle. Le même principe existe d'ailleurs déjà aux deux extrêmes : des modèles au niveau du caractère ou de l'octet (ByT5, par exemple), et d'anciens modèles au niveau du mot.

La vraie question est donc celle des compromis. Pour une tokenisation par syllabes avec un token d'espace dédié, plusieurs points se posent.

**Le problème de la définition.** Il faut un découpeur syllabique fiable. En français, c'est faisable avec des règles, en anglais c'est beaucoup plus flou (« hello » : hel-lo ou hell-o ?), et pour des langues comme le chinois ou l'arabe la notion ne s'applique pas du tout de la même façon. Il faut aussi prévoir comment représenter tout ce qui n'est pas une syllabe : chiffres, ponctuation, code, emojis, noms propres étrangers. Sans mécanisme de repli (par exemple vers des caractères ou des octets), vous aurez des « mots inconnus » que le modèle ne pourra ni lire ni écrire. C'est précisément pour cela que le BPE part des octets : tout texte est représentable.

**Le coût en longueur de séquence.** Le tokeniseur statistique regroupe les mots fréquents en un seul token (« the », « bonjour »). Avec des syllabes, un mot courant en prend souvent 2 ou 3. Les séquences s'allongent donc, ce qui coûte cher : le coût de l'attention croît avec le carré de la longueur, et la fenêtre de contexte se remplit plus vite. En contrepartie, un vocabulaire syllabique est plus petit (quelques milliers d'entrées pour le français), ce qui allège les matrices d'embeddings.

**Le token d'espace séparé.** C'est le point le moins efficace. Dans un texte, une espace sépare chaque mot, donc environ un token sur cinq ou six serait un simple espace, ce qui gaspille de la longueur de séquence pour peu d'information. Coller l'espace au mot suivant, comme le font les tokeniseurs actuels, est plus économique. Cela dit, un token d'espace explicite reste parfaitement viable, et il a l'avantage de la netteté : « world » avec ou sans espace devant ne donne plus deux ids différents.

**Ce que le modèle apprendrait.** Les syllabes porteraient peu de sens en elles-mêmes (« lo » n'a pas de signification propre), donc le modèle devrait recomposer les mots à partir de fragments, ce qui consomme de la capacité dans les premières couches. Ce n'est pas rédhibitoire, car les tokeniseurs statistiques produisent déjà beaucoup de fragments sans sens, mais c'est un peu moins efficace que des unités qui correspondent souvent à des mots ou des radicaux entiers.

En résumé, votre idée donnerait un LLM qui fonctionne, avec des séquences plus longues et un effort d'ingénierie pour gérer les cas non syllabiques. L'approche statistique l'a emporté parce qu'elle est indépendante de la langue, couvre tout texte, et trouve automatiquement un bon équilibre entre longueur de séquence et taille de vocabulaire.

Si le sujet vous intéresse, c'est aussi un terrain de recherche actif : certains travaux tentent de supprimer le tokeniseur en travaillant directement sur les octets, avec des architectures qui regroupent dynamiquement les octets en « patches ».

Voulez-vous voir sur un exemple chiffré ce que donneraient les deux approches en longueur de séquence, ou préférez-vous passer aux embeddings ?

---

# Embeddings

> Passons aux embeddings. Je ne sais pas s'il s'agit de cette notion, alors je vais décrire une chose concernant les données d'entraînement : 
> 
> L'entraînement a pour but de permettre une prédiction du prochain token. Pour obtenir ce résultat, on fournit un corpus de texte. On tokenise, ce qui donne une séquence d'ids.
> Chaque Id est ensuite transmit comme une donnée d'apprentissage par transformation en vecteur dans l'espace des tokens : l'élément i du vecteur est 0 si son index n'est pas l'id du token, et 1 sinon (je suppose ici que si on a N tokens dans cet espace, alors chaque token a son identifiant entre 0 et N-1 inclus) Par exemple, si on l'espace des tokens compte 1000 éléments, chaque donnée d'entraînement sera un vecteur de 1000 élément.
> 
> S'agit-il de ce qu'on appelle l'embedding ?

Presque, mais ce que vous décrivez est une étape juste en amont : c'est l'**encodage one-hot** (« un parmi N »). L'embedding est ce qui vient ensuite.

**Ce que vous décrivez : le one-hot.** Un vecteur de taille N (la taille du vocabulaire), avec un 1 à la position de l'id et des 0 partout ailleurs. C'est une représentation correcte, mais elle a deux défauts : elle est énorme (100 000 dimensions pour un seul token, presque toutes nulles) et elle ne porte aucune information sémantique. Dans cet espace, tous les tokens sont à égale distance les uns des autres : « chat » n'est pas plus proche de « chien » que de « frigo ».

**L'embedding.** C'est une projection de ce vecteur one-hot vers un espace beaucoup plus petit (typiquement 1 000 à 12 000 dimensions), dont les coordonnées sont des nombres réels. Mathématiquement, c'est la multiplication du one-hot (taille N) par une matrice E de taille N × d. Comme le one-hot ne contient qu'un seul 1, ce produit revient simplement à **lire la ligne de E correspondant à l'id**. En pratique, on ne construit donc jamais le vecteur one-hot : on fait directement une consultation de table (« lookup »), id → ligne de la matrice.

Cette matrice E est composée de **paramètres du modèle, appris pendant l'entraînement** comme tous les autres poids. Au départ, ses valeurs sont aléatoires. Au fil de l'entraînement, des tokens qui jouent des rôles similaires dans les phrases voient leurs lignes se rapprocher, ce qui produit la structure sémantique dont on parle souvent.

**Un point important : le one-hot n'est pas la donnée d'entraînement.** Il y a en fait deux usages distincts des tokens :

- **En entrée** : la séquence d'ids passe par la matrice d'embeddings pour devenir une séquence de vecteurs denses, que le Transformer traite.
- **En sortie (la cible)** : le modèle produit, pour chaque position, un score pour chacun des N tokens possibles, transformé en distribution de probabilité (via un softmax). La cible d'apprentissage est alors le token réellement suivant, et c'est là que le vecteur one-hot intervient vraiment : on mesure l'écart entre la distribution prédite et ce one-hot (c'est la *cross-entropy*). Votre intuition sur le vecteur de taille N a donc bien une place, mais côté sortie.

Pour résumer le flux sur « hello world » :

1. Texte → ids : `[15339, 1917]`
2. Chaque id → ligne de E : deux vecteurs denses de dimension d
3. Transformer → à chaque position, un vecteur de dimension d
4. Projection finale (taille d → N) + softmax : probabilités sur tout le vocabulaire
5. Comparaison avec le vrai token suivant (one-hot) → erreur → mise à jour des poids

Une subtilité au passage : de nombreux modèles réutilisent la même matrice (transposée) pour l'étape 4, c'est le *weight tying*.

Souhaitez-vous qu'on regarde sur un mini-exemple chiffré (vocabulaire de 5 tokens, embeddings de dimension 3) comment le lookup et la prédiction s'enchaînent, ou préférez-vous passer au mécanisme d'attention ?

---

# Mécanisme d'attention

> Passons à l'attention. Je n'ai pas de connaissances en la matière.

Le problème que l'attention résout est le suivant. Après l'étape des embeddings, chaque token a un vecteur qui ne dépend que de lui-même : le mot « avocat » aura le même vecteur dans « l'avocat plaide » et dans « j'ai mangé un avocat ». Or le sens d'un mot dépend de son contexte. Il faut donc un mécanisme qui permette à chaque token d'aller chercher de l'information chez les autres tokens de la séquence et d'enrichir son propre vecteur.

**L'idée générale.** Prenons la phrase « La souris a mangé le fromage car elle avait faim ». Pour bien représenter « elle », le modèle doit savoir que ce mot renvoie à « souris » et non à « fromage ». L'attention fait ceci : le vecteur de « elle » est remplacé par une **moyenne pondérée** des informations venant de tous les tokens précédents, avec un poids élevé pour « souris » et des poids faibles pour les autres. Ces poids ne sont pas fixés à la main, ils sont calculés à chaque fois en fonction du contenu, et c'est ce calcul qui est appris.

**Le mécanisme : requête, clé, valeur.** À partir du vecteur x de chaque token, on calcule trois vecteurs, en multipliant x par trois matrices de paramètres apprises (W_Q, W_K, W_V) :

- la **requête** (query) : « voici ce que je cherche » ;
- la **clé** (key) : « voici ce que je suis, voici à quelles requêtes je peux répondre » ;
- la **valeur** (value) : « voici l'information que je transmets si l'on vient me consulter ».

Pour un token donné, par exemple « elle », le calcul se déroule ainsi :

1. On compare sa requête à la clé de chaque token précédent par un **produit scalaire**. Plus les deux vecteurs pointent dans la même direction, plus le score est élevé. On obtient un score par token.
2. On divise les scores par √d (d étant la dimension des clés), simplement pour éviter qu'ils ne deviennent trop grands et ne rendent l'apprentissage instable.
3. On applique un **softmax** : les scores deviennent des poids positifs dont la somme vaut 1.
4. Le nouveau vecteur de « elle » est la somme des **valeurs** de tous les tokens, pondérées par ces poids.

On peut le voir comme une recherche dans un dictionnaire « floue » : au lieu de retrouver une seule entrée dont la clé correspond exactement, on récupère un mélange de toutes les valeurs, dosé par la ressemblance entre la requête et chaque clé. Le modèle apprend, via W_Q, W_K et W_V, à produire des requêtes et des clés qui se « répondent » quand c'est pertinent. Personne ne lui dit que les pronoms doivent regarder leur référent : cela émerge parce que cela aide à prédire le token suivant.

**Le masque causal.** Pendant l'entraînement, on traite toute la séquence d'un coup, et chaque position doit prédire le token suivant. Si « elle » pouvait regarder les tokens qui le suivent, il pourrait simplement recopier la réponse. On interdit donc à chaque token de regarder vers l'avant : avant le softmax, on met les scores des positions futures à moins l'infini, ce qui leur donne un poids nul. C'est ce qui distingue les modèles génératifs des modèles de type « compréhension » comme BERT.

**En version matricielle.** Tous les tokens sont traités simultanément, ce qui est la grande force des Transformers sur les GPU. Avec Q, K et V les matrices empilant les vecteurs de tous les tokens, tout tient dans une formule : softmax(QKᵀ/√d)·V. La matrice QKᵀ contient un score pour chaque paire de tokens, soit n × n valeurs pour n tokens. C'est exactement l'origine du coût quadratique en longueur de séquence dont nous parlions pour la tokenisation.

**Plusieurs têtes en parallèle.** Un seul jeu de (W_Q, W_K, W_V) ne capture qu'un type de relation à la fois. On en utilise donc plusieurs en parallèle, les **têtes d'attention** (de quelques-unes à une centaine), chacune avec ses propres matrices et de plus petite dimension. L'une peut suivre les références des pronoms, une autre les accords grammaticaux, une autre le token précédent immédiat, etc. Leurs sorties sont concaténées puis recombinées par une dernière matrice.

**Deux compléments pour situer la pièce dans l'ensemble.**

- L'attention ne connaît pas l'ordre des tokens : elle ne voit qu'un ensemble de vecteurs. Il faut donc injecter l'information de position, par des encodages de position ajoutés aux embeddings ou, dans les modèles récents, par une rotation des vecteurs requête et clé (RoPE).
- Un bloc Transformer combine une couche d'attention (les tokens échangent de l'information) puis un petit réseau MLP appliqué à chaque token séparément (chaque token « réfléchit » sur ce qu'il a reçu), avec des connexions résiduelles autour de chaque étape. On empile ces blocs des dizaines de fois : les premières couches captent plutôt des relations simples et locales, les suivantes des relations plus abstraites.

Souhaitez-vous que l'on déroule un mini-exemple chiffré (3 tokens, vecteurs de dimension 2) pour voir concrètement les scores, le softmax et le mélange des valeurs, ou préférez-vous passer au bloc Transformer complet et à la façon dont les couches s'empilent ?

---

# Le transformer

> Passons au bloc Transformer

Le bloc Transformer est l'unité qu'on répète des dizaines de fois (environ 12 pour un petit modèle, 80 à 120 pour les plus gros). Chaque bloc prend une séquence de vecteurs et en produit une autre de **même taille**, ce qui permet de les empiler librement.

**Le flux des données : le « flux résiduel ».** Pour comprendre le bloc, il faut d'abord une image : imaginez une autoroute qui traverse tout le modèle et transporte, pour chaque position, un vecteur de dimension d. C'est le *flux résiduel*. Il démarre avec l'embedding du token (plus l'information de position), et chaque couche vient **lire** ce flux, calculer quelque chose, puis **ajouter** son résultat au flux. Elle ne le remplace jamais. À la fin, le flux contient l'embedding initial plus la somme des contributions de toutes les couches, et c'est lui qui est projeté vers le vocabulaire pour prédire le token suivant.

**Le contenu d'un bloc.** Un bloc contient deux sous-couches, chacune enveloppée dans le même schéma :

1. **Normalisation** du flux (on recentre et remet à l'échelle le vecteur, par exemple avec LayerNorm ou RMSNorm).
2. **Sous-couche** : d'abord l'attention multi-têtes, puis le MLP.
3. **Addition** du résultat au flux d'origine (connexion résiduelle).

En pseudo-code, avec x le flux :

```
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

**Rôle de chaque sous-couche.**

- **L'attention** est le seul endroit où les positions communiquent entre elles. Chaque token y va chercher de l'information chez les tokens précédents, comme vu précédemment.
- **Le MLP** (réseau « feed-forward ») s'applique à chaque position **indépendamment**, sans aucun échange entre tokens. C'est typiquement : une matrice qui élargit le vecteur de d à environ 4d dimensions, une non-linéarité (GELU, ou SwiGLU dans les modèles récents), puis une matrice qui revient à d. La non-linéarité est indispensable : sans elle, empiler des couches équivaudrait à une seule transformation linéaire. On considère souvent que le MLP sert de **mémoire** où sont stockés les faits et associations appris (par exemple « la capitale de la France → Paris »), et il concentre environ les deux tiers des paramètres du modèle.

Une façon simple de résumer : l'attention **déplace** de l'information entre les positions, le MLP **transforme** l'information présente à chaque position.

**Pourquoi les connexions résiduelles sont essentielles.** Sans elles, le signal et le gradient devraient traverser des dizaines de transformations successives, et l'entraînement deviendrait quasiment impossible (le gradient s'évanouit ou explose). Avec l'addition, il existe toujours un chemin direct de la sortie vers l'entrée, le gradient circule bien, et chaque couche n'a qu'à apprendre une petite *correction* au lieu de recréer toute la représentation. Cela permet aussi aux couches de se spécialiser : l'une écrit dans le flux une information, une couche plus loin la relit et l'utilise.

**Pourquoi la normalisation.** Les valeurs du flux ont tendance à grossir au fil des additions, ce qui déstabilise l'entraînement. La normalisation maintient les vecteurs dans une échelle raisonnable avant chaque sous-couche. Dans la version moderne décrite ici (la normalisation *avant* la sous-couche, dite « pre-norm »), l'entraînement est plus stable que dans le Transformer d'origine de 2017, qui normalisait après.

**L'empilement.** On enchaîne N blocs identiques en structure mais avec des paramètres différents. Les observations de recherche suggèrent une certaine progression : les premières couches traitent plutôt la syntaxe locale et la reconstitution des mots à partir de fragments, les couches intermédiaires construisent des représentations plus sémantiques, et les dernières préparent la prédiction du token suivant. Ce n'est pas une règle stricte, mais c'est une bonne intuition.

**Le modèle complet, de bout en bout.** Avec ce que nous avons vu, vous avez maintenant toutes les pièces d'un LLM de type GPT :

```
ids → embeddings (+ positions)
    → [ bloc Transformer ] × N
    → normalisation finale
    → projection vers le vocabulaire (d → V)
    → softmax → probabilités du token suivant
```

Les paramètres se répartissent donc entre la matrice d'embeddings, les matrices W_Q, W_K, W_V et de sortie de l'attention dans chaque bloc, les deux matrices du MLP dans chaque bloc, et la projection finale (souvent partagée avec les embeddings). Les hyperparamètres qui définissent un modèle sont essentiellement : la dimension d, le nombre de blocs N, le nombre de têtes, la taille du vocabulaire et la longueur de contexte maximale.

Souhaitez-vous qu'on calcule ensemble le nombre de paramètres d'un modèle concret à partir de ces hyperparamètres (c'est très éclairant pour comprendre d'où viennent les « 7 milliards » ou « 70 milliards »), ou préférez-vous passer au pré-entraînement et à la façon dont les poids sont réellement ajustés ?

---

# THE END







