---
source: https://claude.ai/chat/431ef346-9797-4c19-8539-be79bff78d88
---

# Reinforcemnet learning et LLM

> Bonjour, je m'intéresse aux llm. Parle moi un peu du reinforcement learning.

Bonjour ! Le reinforcement learning (RL, apprentissage par renforcement) est une approche où un agent apprend à prendre des décisions en interagissant avec un environnement : il agit, reçoit une **récompense** (positive ou négative), et ajuste son comportement pour maximiser les récompenses cumulées sur le long terme.

**Dans le contexte des LLM**, le RL intervient surtout après le pré-entraînement :

- **RLHF** (Reinforcement Learning from Human Feedback) : des humains comparent plusieurs réponses du modèle, on entraîne un « modèle de récompense » qui imite ces préférences, puis on optimise le LLM avec un algorithme comme PPO pour produire des réponses mieux notées. C'est ce qui rend les modèles plus utiles, polis et sûrs.
- **RLAIF** : même principe, mais le feedback vient d'une IA (guidée par des principes écrits) plutôt que d'humains.
- **RL avec récompenses vérifiables (RLVR)** : pour les maths, le code ou la logique, on peut vérifier automatiquement si la réponse est correcte. Le modèle est récompensé quand il a juste, ce qui favorise l'émergence de raisonnements longs, étape par étape. C'est la base de nombreux modèles de « raisonnement ».
- **Alternatives plus simples** : DPO, GRPO et consorts évitent certaines complexités de PPO (par exemple le modèle de valeur séparé).

**Pourquoi c'est important :** le pré-entraînement apprend à prédire du texte, pas à être utile ou à bien raisonner. Le RL oriente le modèle vers un *comportement désirable*.

**Difficultés principales :** le « reward hacking » (le modèle exploite les failles de la récompense au lieu de bien faire la tâche), l'instabilité de l'entraînement, et le coût du feedback de qualité.

Tu veux que j'approfondisse un point en particulier : le fonctionnement de PPO/GRPO, le RLHF pas à pas, ou le RL pour le raisonnement ?

# Définitions

> Rappelle moi la signification des accronymes PPO, GRPO ET DPO

- **PPO** : *Proximal Policy Optimization* (optimisation de politique proximale). Algorithme de RL qui met à jour le modèle par petits pas contrôlés, pour éviter qu'une mise à jour trop brutale ne dégrade ce qu'il a déjà appris.

- **GRPO** : *Group Relative Policy Optimization* (optimisation de politique relative au groupe). Variante de PPO qui génère plusieurs réponses par question et compare leur récompense à la moyenne du groupe, ce qui évite d'avoir un modèle de valeur séparé.

- **DPO** : *Direct Preference Optimization* (optimisation directe des préférences). Méthode qui entraîne le modèle directement sur des paires « réponse préférée / réponse rejetée », sans passer par un modèle de récompense ni par du RL à proprement parler.

# PPO et GRPO

> Ok, commençons par ppo et grpo

## PPO

L'idée : améliorer le modèle (la « politique ») sans jamais le laisser trop changer d'un coup.

**Ingrédients pour un LLM :**
- la **politique** : le LLM qu'on entraîne
- un **modèle de récompense** qui note les réponses
- un **modèle de valeur** (le « critique ») qui estime la récompense attendue à chaque token
- un **modèle de référence** (le LLM initial figé), pour ne pas trop s'en éloigner

**Fonctionnement :**
1. Le LLM génère des réponses, qui sont notées.
2. Le critique permet de calculer l'**avantage** : cette réponse était-elle meilleure ou pire que prévu ?
3. On met à jour le LLM pour favoriser les actions à avantage positif, avec un **clipping** : on calcule le rapport entre les probabilités nouvelle et ancienne politique, et on le bride dans [1-ε, 1+ε]. Ça empêche les mises à jour trop violentes.
4. Une pénalité **KL** garde le modèle proche de la référence.

**Défaut :** il faut faire tourner 4 modèles en mémoire, et le critique est difficile à entraîner correctement.

## GRPO

Même objectif, mais **sans critique**.

1. Pour chaque question, on génère un **groupe** de G réponses (par ex. 8 ou 16).
2. On note chacune.
3. L'avantage de chaque réponse = sa note **moins la moyenne du groupe**, divisée par l'écart-type. Le groupe sert de référence.
4. On garde le clipping de PPO, et la pénalité KL est ajoutée directement dans la fonction de perte.

**Intuition :** si une réponse est meilleure que ses « sœurs », on la renforce ; sinon on la décourage.

**Avantages :** moins de mémoire, plus simple, et très adapté aux récompenses vérifiables (maths, code). C'est ce qu'a popularisé DeepSeek (DeepSeekMath, puis R1).

Tu veux que je détaille la formule du clipping, ou qu'on passe à DPO ?

# Le clipping

> Detaille la formule de clipping

## La formule

**Rapport de probabilités :**

r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)

C'est la probabilité du token a_t sous la nouvelle politique, divisée par sa probabilité sous l'ancienne (celle qui a généré les données). Si r_t = 1, rien n'a changé ; si r_t = 1,3, le token est 30 % plus probable qu'avant.

**Objectif PPO clippé :**

L(θ) = E_t [ min( r_t · A_t , clip(r_t, 1−ε, 1+ε) · A_t ) ]

- **A_t** : l'avantage (positif = mieux que prévu, négatif = pire)
- **ε** : petit hyperparamètre, typiquement 0,2
- **clip(r, 1−ε, 1+ε)** : ramène r dans [0,8 ; 1,2]

On *maximise* cette quantité.

## Intuition par cas

**Cas 1 : A_t > 0 (bonne action).** On veut augmenter sa probabilité, donc r monte. Mais dès que r dépasse 1+ε, le terme clippé devient constant et le gradient s'annule : on arrête de pousser. Le « min » garde la valeur la plus pessimiste.

**Cas 2 : A_t < 0 (mauvaise action).** On veut diminuer sa probabilité, donc r baisse. Dès que r passe sous 1−ε, le gradient s'annule : inutile de l'écraser davantage.

**Cas asymétrique important :** si l'action est mauvaise (A < 0) mais que r est *supérieur* à 1+ε (la mise à jour a rendu l'action plus probable par erreur), le min choisit le terme non clippé, donc le gradient reste actif et corrige l'erreur. Le clipping ne limite que les changements « dans la bonne direction ».

## Exemple chiffré

ε = 0,2, A = +2.
- r = 1,1 → objectif = 1,1 × 2 = 2,2 (gradient actif)
- r = 1,5 → min(3, 1,2 × 2 = 2,4) = 2,4 (plafonné, gradient nul)

## Dans GRPO

La formule est identique, sauf que **A_t est l'avantage relatif au groupe** (note moins moyenne, divisée par l'écart-type, la même valeur pour tous les tokens d'une réponse), et qu'on ajoute le terme −β · KL(π_θ ‖ π_réf) à l'objectif.

Tu veux qu'on passe à DPO, ou qu'on code un mini-exemple de cette perte ?

# DPO

> Passe à dpo

## DPO : l'idée

Le RLHF classique passe par deux étapes lourdes : entraîner un modèle de récompense, puis faire du RL. DPO montre qu'on peut **sauter les deux** : on optimise directement le LLM sur des préférences, avec une simple perte de classification.

**Données :** des triplets (prompt x, réponse préférée y_w, réponse rejetée y_l), « w » pour *win*, « l » pour *lose*.

**Modèles :** seulement deux : la politique π_θ (qu'on entraîne) et la référence π_réf (figée, souvent le modèle après SFT).

## La formule

L(θ) = − E [ log σ( β · ( log π_θ(y_w|x)/π_réf(y_w|x) − log π_θ(y_l|x)/π_réf(y_l|x) ) ) ]

- **σ** : la fonction sigmoïde
- **β** : hyperparamètre (typiquement 0,1) qui contrôle l'écart toléré par rapport à la référence
- **log π_θ/π_réf** : le « log-ratio implicite », qui joue le rôle de récompense

## Intuition

Chaque log-ratio mesure de combien le modèle a augmenté la probabilité d'une réponse par rapport à la référence. La perte pousse à :
- **augmenter** la probabilité de la réponse préférée (relativement à la référence),
- **diminuer** celle de la réponse rejetée.

C'est comme une classification : « la préférée doit avoir un meilleur score implicite que la rejetée ».

## D'où ça vient

Les auteurs partent de l'objectif RLHF (maximiser la récompense avec une contrainte KL) dont la solution optimale est connue en forme fermée. En l'inversant, on exprime la **récompense en fonction de la politique** : r(x,y) = β · log π_θ/π_réf + constante. Injectée dans le modèle de préférences de Bradley-Terry, la constante s'annule et il reste la formule ci-dessus. D'où le slogan : *« votre modèle de langage est secrètement un modèle de récompense »*.

## PPO/GRPO vs DPO

| | PPO / GRPO | DPO |
|---|---|---|
| Données | Réponses générées + récompense | Paires de préférences fixes |
| Modèle de récompense | Oui (ou vérifieur) | Non |
| Génération pendant l'entraînement | Oui (*on-policy*) | Non (*offline*) |
| Complexité | Élevée | Faible, stable |

**Limites de DPO :** il apprend sur des données figées, donc ne explore pas de nouvelles réponses ; il peut aussi baisser la probabilité des deux réponses à la fois, et il est moins adapté au raisonnement à récompense vérifiable, où GRPO domine.

Tu veux qu'on code un mini-exemple de la perte DPO, ou qu'on compare les trois sur un cas concret ?


