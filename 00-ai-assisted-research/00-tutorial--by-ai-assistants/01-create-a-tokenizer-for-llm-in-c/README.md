---
source: https://claude.ai/chat/e08080b5-62bd-4e7f-8e79-e46bceccf573
---

# Create a tokenizer for LLM in C

> Bonjour, je m'intéresse à la creation d'un LLM domestique. Je voudrais commencer par le tokenizer. Supposons que j'ai le texte du livre "Les lettres de mon moulin", d'Alphonse Daudet. C'est un recueil d'histoires indépendante, ce qui peut faciliter le partitionnement de l'oeuvre au besoin.
>
> Je voudrais donc creer un tokenizer à partir de tout ou partie de ce corpus. En language C, sans dépendance, juste la librairie standard.
>
> J'ai compris qu'on analyse statistiquement le corpus pour dégager des combinaisons fréquente et les considérer comme un symbole à part entière (token) afin de limiter ensuite la taille des inputs et donc des calculs.

Votre description est juste : c'est l'algorithme **BPE** (Byte Pair Encoding). Voici une implémentation en C99, libc uniquement (~250 lignes), compilée et testée sur un petit corpus : le texte est bien restitué à l'identique après encodage puis décodage.

**Compilation et usage**
```
gcc -O2 -Wall -o bpe bpe.c
./bpe train daudet.txt 1024 modele.bpe
./bpe pieces modele.bpe < extrait.txt
./bpe encode modele.bpe < extrait.txt | ./bpe decode modele.bpe
```

**Choix de conception**
- **Niveau octet** : le vocabulaire de départ compte les 256 valeurs d'octet. Les accents UTF-8 (é, è, ç, œ…) sont donc toujours encodables, sans token « inconnu ». Les premiers tokens appris recollent naturellement les octets des lettres accentuées.
- **Pré-découpage en mots** : un espace simple reste collé au mot qui suit (` moulin`), et la ponctuation est isolée. Les fusions ne traversent jamais ces frontières, ce qui donne des tokens plus propres.
- **Entraînement efficace** : on compte les mots uniques avec leur fréquence, puis on compte les paires pondérées par cette fréquence. C'est bien plus rapide que de rescanner tout le texte à chaque fusion.
- **Encodage** : on rejoue les fusions dans l'ordre où elles ont été apprises.

**Conseils pour votre corpus**
- *Les Lettres de mon moulin* fait environ 250 Ko. Visez un vocabulaire de **512 à 1024** : au-delà, vous apprendriez surtout des mots entiers rares, sans généralisation.
- Gardez quelques contes de côté pour tester : comparez le ratio octets/token sur du texte non vu pour mesurer si le tokenizer généralise.
- Le texte est dans le domaine public (Wikisource ou Gutenberg). Retirez les en-têtes et licences, qui fausseraient les statistiques.

**Limites connues**
- L'apostrophe typographique (’) est traitée comme une lettre, donc `l’eau` reste un seul segment. Si votre texte en contient beaucoup, c'est à ajuster dans `char_class`.
- Il n'y a pas de tokens spéciaux (`<fin_de_texte>`, etc.). Ce sera la prochaine étape pour entraîner un modèle.
- Le comptage des paires parcourt une table `V×V` à chaque itération. C'est simple et rapide à cette échelle, mais il faudrait des listes de paires incrémentales pour un gros corpus.

Voulez-vous qu'on passe ensuite aux tokens spéciaux, ou à la préparation du dataset (corpus tokenisé, découpé en séquences) pour le modèle ?


# Token speciaux

> Je voudrais passer aux token spéciaux

J'ai mis à jour `bpe.c` (le format du modèle passe à `BPE2`, il faut donc réentraîner). Testé : l'aller-retour encodage/décodage est exact et le mode `--literal` fonctionne.

**Ce qui a changé**
- **Trois tokens spéciaux par défaut** : `<|fin_de_texte|>`, `<|debut|>` et `<|pad|>`. Ils prennent les derniers ids, **après** les fusions (ex. 297, 298, 299 pour un vocabulaire de 300). Si vous changez la taille du vocabulaire, leurs ids changent.
- **Pas d'apprentissage statistique pour eux** : à l'entraînement, ils sont repérés par correspondance exacte et exclus des comptages. Sinon, `<|fin_de_texte|>` serait découpé en morceaux ou fusionné avec les mots voisins.
- **Taille du vocabulaire** : l'argument est désormais le total (octets + fusions + spéciaux). Avec `1024`, vous obtenez 765 fusions et 3 spéciaux.
- **Nouvelle commande** : `./bpe info modele.bpe` affiche la composition du vocabulaire et les ids spéciaux.
- **Nouvelle option** : `--literal` désactive la reconnaissance des spéciaux à l'encodage. Elle sert quand le texte vient d'un utilisateur et ne doit pas pouvoir injecter un `<|fin_de_texte|>`.

**Usage pour votre corpus**
Chaque conte est séparé par la fin de texte, puis on entraîne :
```
for f in contes/*.txt; do cat "$f"; echo "<|fin_de_texte|>"; done > daudet.txt
./bpe train daudet.txt 1024 modele.bpe
```
Le rôle de chaque token apparaîtra à l'entraînement du modèle :
- `<|fin_de_texte|>` délimite les contes. Le modèle apprend à le produire quand une histoire se termine, ce qui sert de condition d'arrêt à la génération.
- `<|debut|>` s'ajoute au début de chaque séquence si vous voulez un point de départ explicite.
- `<|pad|>` complète les séquences courtes dans un batch. Sa perte devra être masquée.

Les tokens spéciaux sont stockés dans le fichier modèle (une ligne par nom). Pour en ajouter, par exemple un `<|dialogue|>`, il suffit de modifier `default_specials` puis de réentraîner. Le format est prévu pour 16 spéciaux au maximum.

La prochaine étape logique est la préparation du dataset : encoder tout le corpus en un fichier binaire d'ids (`uint16`, suffisant jusqu'à 65 536 tokens), puis le découper en séquences de longueur fixe avec les paires (entrée, cible décalée d'un token). On s'y met ?


