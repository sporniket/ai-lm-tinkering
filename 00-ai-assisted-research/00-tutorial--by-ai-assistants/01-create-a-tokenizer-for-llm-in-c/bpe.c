/*
 * bpe.c - Tokenizer BPE (Byte Pair Encoding) niveau octet, C99, sans dependance.
 *
 * Compilation :  gcc -O2 -Wall -o bpe bpe.c
 *
 * Usage :
 *   ./bpe train  corpus.txt 1024 modele.bpe   # apprend 1024-256 fusions
 *   ./bpe encode modele.bpe < texte.txt       # affiche les ids
 *   ./bpe encode modele.bpe --literal < t.txt # idem, sans interpreter les tokens speciaux
 *   ./bpe info   modele.bpe                   # taille du vocabulaire et ids speciaux
 *   ./bpe pieces modele.bpe < texte.txt       # affiche les tokens separes par |
 *   ./bpe encode modele.bpe < texte.txt | ./bpe decode modele.bpe
 *
 * Principe :
 *  - Le vocabulaire de base = les 256 valeurs d'octet (donc tout texte UTF-8,
 *    accents compris, est encodable, sans token "inconnu").
 *  - Le texte est d'abord coupe en "mots" (pre-tokenisation) : les fusions
 *    ne traversent jamais une frontiere de mot.
 *  - Entrainement : on compte les paires de tokens adjacentes, on fusionne la
 *    plus frequente en un nouveau token, et on recommence.
 *  - Tokens speciaux (<|fin_de_texte|>, ...) : ajoutes APRES les fusions, jamais
 *    appris statistiquement. Reconnus par correspondance exacte, hors entrainement.
 *  - Encodage : on rejoue les fusions dans l'ordre ou elles ont ete apprises.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_VOCAB 4096
#define HASH_SIZE (1u << 20)

static int n_merges = 0;
static int merge_a[MAX_VOCAB], merge_b[MAX_VOCAB];
#define MAX_SPECIAL 16
static unsigned char *tok_bytes[MAX_VOCAB + MAX_SPECIAL];
static int tok_len[MAX_VOCAB + MAX_SPECIAL];

/* Tokens speciaux : id = 256 + n_merges + indice */
static int n_special = 0;
static char *special_name[MAX_SPECIAL];
static int parse_special = 1;   /* 0 = traiter "<|...|>" comme du texte ordinaire */
static const char *default_specials[] = { "<|fin_de_texte|>", "<|debut|>", "<|pad|>" };

static void set_default_specials(void) {
    n_special = (int)(sizeof default_specials / sizeof *default_specials);
    for (int i = 0; i < n_special; i++) special_name[i] = strdup(default_specials[i]);
}

static void die(const char *msg) { fprintf(stderr, "erreur: %s\n", msg); exit(1); }

static unsigned char *read_all(FILE *f, size_t *n) {
    size_t cap = 1 << 16, len = 0;
    unsigned char *buf = malloc(cap);
    if (!buf) die("memoire");
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, f)) > 0) {
        len += r;
        if (len == cap) { cap *= 2; buf = realloc(buf, cap); if (!buf) die("memoire"); }
    }
    *n = len;
    return buf;
}

/* ---------- Pre-tokenisation ---------- */

/* 0 = lettre (y compris octets >= 0x80 : accents UTF-8), 1 = chiffre,
   2 = espace, 3 = autre (ponctuation) */
static int char_class(unsigned char c) {
    if (c >= 0x80 || isalpha(c)) return 0;
    if (isdigit(c)) return 1;
    if (isspace(c)) return 2;
    return 3;
}

/* Longueur du prochain segment commencant en t[0]. Un espace simple colle au
   mot qui suit (" moulin"), comme dans GPT-2. */
static size_t seg_len(const unsigned char *t, size_t n) {
    size_t i = 0;
    if (t[0] == ' ' && n > 1 && char_class(t[1]) != 2) i = 1;
    int c = char_class(t[i]);
    if (c == 3) return i + 1;           /* ponctuation : 1 octet */
    size_t j = i + 1;
    while (j < n && char_class(t[j]) == c) j++;
    if (c == 2 && j < n && j - i > 1 && t[j - 1] == ' ') j--; /* laisse l'espace au mot suivant */
    return j;
}

/* Indice du token special qui commence en t[0], ou -1. */
static int match_special(const unsigned char *t, size_t n) {
    if (!parse_special || t[0] != '<') return -1;
    for (int s = 0; s < n_special; s++) {
        size_t l = strlen(special_name[s]);
        if (l <= n && memcmp(t, special_name[s], l) == 0) return s;
    }
    return -1;
}

/* Prochaine unite : un token special (*sp >= 0) ou un segment ordinaire (*sp = -1). */
static size_t next_piece(const unsigned char *t, size_t n, int *sp) {
    *sp = match_special(t, n);
    if (*sp >= 0) return strlen(special_name[*sp]);
    size_t l = seg_len(t, n);
    /* un espace ne doit pas s'accrocher a un token special qui suit */
    if (t[0] == ' ' && l > 1 && match_special(t + 1, n - 1) >= 0) l = 1;
    return l;
}

/* ---------- Table des mots uniques (entrainement) ---------- */

typedef struct {
    unsigned char *key; int klen;
    int *ids; int len;
    long freq;
} Word;

static Word *words = NULL;
static size_t n_words = 0, cap_words = 0;
static int *slots;

static unsigned hash_bytes(const unsigned char *p, int n) {
    unsigned h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static void add_word(const unsigned char *p, int n) {
    unsigned h = hash_bytes(p, n) & (HASH_SIZE - 1);
    while (slots[h] >= 0) {
        Word *w = &words[slots[h]];
        if (w->klen == n && memcmp(w->key, p, n) == 0) { w->freq++; return; }
        h = (h + 1) & (HASH_SIZE - 1);
    }
    if (n_words >= HASH_SIZE / 2) die("trop de mots uniques");
    if (n_words == cap_words) {
        cap_words = cap_words ? cap_words * 2 : 1024;
        words = realloc(words, cap_words * sizeof(Word));
        if (!words) die("memoire");
    }
    Word *w = &words[n_words];
    w->key = malloc(n); memcpy(w->key, p, n); w->klen = n;
    w->ids = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) w->ids[i] = p[i];
    w->len = n; w->freq = 1;
    slots[h] = (int)n_words++;
}

/* ---------- Vocabulaire ---------- */

static void build_vocab(void) {
    for (int i = 0; i < 256; i++) {
        tok_bytes[i] = malloc(1); tok_bytes[i][0] = (unsigned char)i; tok_len[i] = 1;
    }
    for (int m = 0; m < n_merges; m++) {
        int id = 256 + m, a = merge_a[m], b = merge_b[m];
        tok_len[id] = tok_len[a] + tok_len[b];
        tok_bytes[id] = malloc(tok_len[id]);
        memcpy(tok_bytes[id], tok_bytes[a], tok_len[a]);
        memcpy(tok_bytes[id] + tok_len[a], tok_bytes[b], tok_len[b]);
    }
    for (int i = 0; i < n_special; i++) {
        int id = 256 + n_merges + i;
        tok_len[id] = (int)strlen(special_name[i]);
        tok_bytes[id] = malloc(tok_len[id]);
        memcpy(tok_bytes[id], special_name[i], tok_len[id]);
    }
}

/* ---------- Entrainement ---------- */

static void train(const char *corpus, int vocab_size, const char *out) {
    set_default_specials();
    parse_special = 1;
    /* vocab_size = taille TOTALE (octets + fusions + tokens speciaux) */
    int n_target = vocab_size - 256 - n_special;
    if (n_target < 1 || vocab_size > MAX_VOCAB) die("taille de vocabulaire invalide");
    FILE *f = fopen(corpus, "rb");
    if (!f) die("impossible d'ouvrir le corpus");
    size_t n; unsigned char *text = read_all(f, &n); fclose(f);

    slots = malloc(HASH_SIZE * sizeof(int));
    memset(slots, 0xff, HASH_SIZE * sizeof(int));   /* -1 */
    size_t n_sp_seen = 0;
    for (size_t pos = 0; pos < n; ) {
        int sp;
        size_t l = next_piece(text + pos, n - pos, &sp);
        if (sp < 0) add_word(text + pos, (int)l);
        else n_sp_seen++;                 /* les speciaux sont exclus des statistiques */
        pos += l;
    }
    fprintf(stderr, "corpus: %zu octets, %zu mots uniques, %zu tokens speciaux ignores\n",
            n, n_words, n_sp_seen);

    int V = 256 + n_target;               /* octets + fusions (sans les speciaux) */
    int *cnt = malloc((size_t)V * V * sizeof(int));
    if (!cnt) die("memoire");

    for (int m = 0; m < n_target; m++) {
        memset(cnt, 0, (size_t)V * V * sizeof(int));
        for (size_t w = 0; w < n_words; w++) {
            Word *wd = &words[w];
            for (int k = 0; k + 1 < wd->len; k++)
                cnt[wd->ids[k] * V + wd->ids[k + 1]] += (int)wd->freq;
        }
        int best = 0; size_t bi = 0;
        for (size_t i = 0; i < (size_t)V * V; i++)
            if (cnt[i] > best) { best = cnt[i]; bi = i; }
        if (best < 2) { fprintf(stderr, "plus de paire repetee, arret\n"); break; }

        int a = (int)(bi / V), b = (int)(bi % V), nid = 256 + m;
        merge_a[m] = a; merge_b[m] = b; n_merges = m + 1;

        for (size_t w = 0; w < n_words; w++) {
            Word *wd = &words[w];
            if (wd->len < 2) continue;
            int o = 0;
            for (int r = 0; r < wd->len; ) {
                if (r + 1 < wd->len && wd->ids[r] == a && wd->ids[r + 1] == b) {
                    wd->ids[o++] = nid; r += 2;
                } else wd->ids[o++] = wd->ids[r++];
            }
            wd->len = o;
        }
        if (m % 50 == 0 || m == n_target - 1) fprintf(stderr, "fusion %d/%d (x%d)\n", m + 1, n_target, best);
    }

    FILE *o = fopen(out, "w");
    if (!o) die("impossible d'ecrire le modele");
    fprintf(o, "BPE2\n%d\n", n_merges);
    for (int m = 0; m < n_merges; m++) fprintf(o, "%d %d\n", merge_a[m], merge_b[m]);
    fprintf(o, "%d\n", n_special);
    for (int i = 0; i < n_special; i++) fprintf(o, "%s\n", special_name[i]);
    fclose(o);
    fprintf(stderr, "modele ecrit: %s (%d tokens dont %d speciaux)\n",
            out, 256 + n_merges + n_special, n_special);
}

/* ---------- Chargement / encodage / decodage ---------- */

static int *pair_id = NULL;   /* pair_id[a*V+b] = id du token fusionne, ou -1 */
static int V_loaded;

static void load_model(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die("modele introuvable");
    char magic[8];
    if (fscanf(f, "%7s %d", magic, &n_merges) != 2 || strcmp(magic, "BPE2") != 0
        || n_merges < 0 || n_merges > MAX_VOCAB - 256) die("format de modele invalide");
    for (int m = 0; m < n_merges; m++)
        if (fscanf(f, "%d %d", &merge_a[m], &merge_b[m]) != 2) die("modele tronque");
    if (fscanf(f, "%d", &n_special) != 1 || n_special < 0 || n_special > MAX_SPECIAL)
        die("tokens speciaux invalides");
    for (int i = 0; i < n_special; i++) {
        char name[128];
        if (fscanf(f, "%127s", name) != 1) die("modele tronque");
        special_name[i] = strdup(name);
    }
    fclose(f);
    build_vocab();
    V_loaded = 256 + n_merges;          /* taille hors speciaux : sert a la table des paires */
    pair_id = malloc((size_t)V_loaded * V_loaded * sizeof(int));
    if (!pair_id) die("memoire");
    for (size_t i = 0; i < (size_t)V_loaded * V_loaded; i++) pair_id[i] = -1;
    for (int m = 0; m < n_merges; m++) pair_id[merge_a[m] * V_loaded + merge_b[m]] = 256 + m;
}

/* Encode un segment ; ecrit dans out, renvoie le nombre de tokens. */
static int encode_segment(const unsigned char *p, int n, int *out) {
    int len = n;
    for (int i = 0; i < n; i++) out[i] = p[i];
    for (;;) {
        int best = -1;      /* la fusion apprise en premier a la priorite */
        for (int k = 0; k + 1 < len; k++) {
            int id = pair_id[out[k] * V_loaded + out[k + 1]];
            if (id >= 0 && (best < 0 || id < best)) best = id;
        }
        if (best < 0) break;
        int a = merge_a[best - 256], b = merge_b[best - 256], o = 0;
        for (int r = 0; r < len; ) {
            if (r + 1 < len && out[r] == a && out[r + 1] == b) { out[o++] = best; r += 2; }
            else out[o++] = out[r++];
        }
        len = o;
    }
    return len;
}

static void encode_stdin(int show_pieces) {
    size_t n; unsigned char *text = read_all(stdin, &n);
    int *buf = malloc((n + 1) * sizeof(int));
    if (!buf) die("memoire");
    size_t total = 0;
    for (size_t pos = 0; pos < n; ) {
        int sp;
        size_t l = next_piece(text + pos, n - pos, &sp);
        int k;
        if (sp >= 0) { buf[0] = V_loaded + sp; k = 1; }
        else k = encode_segment(text + pos, (int)l, buf);
        for (int i = 0; i < k; i++) {
            if (show_pieces) { fwrite(tok_bytes[buf[i]], 1, tok_len[buf[i]], stdout); fputs("\xc2\xa6", stdout); }
            else printf("%d ", buf[i]);
        }
        total += k; pos += l;
    }
    putchar('\n');
    fprintf(stderr, "%zu octets -> %zu tokens (%.2f octets/token)\n", n, total,
            total ? (double)n / total : 0.0);
}

static void decode_stdin(void) {
    int id;
    while (scanf("%d", &id) == 1) {
        if (id < 0 || id >= V_loaded + n_special) die("id hors vocabulaire");
        fwrite(tok_bytes[id], 1, tok_len[id], stdout);
    }
}

static void info(void) {
    printf("vocabulaire: %d tokens = 256 octets + %d fusions + %d speciaux\n",
           V_loaded + n_special, n_merges, n_special);
    for (int i = 0; i < n_special; i++) printf("  %d  %s\n", V_loaded + i, special_name[i]);
}

int main(int argc, char **argv) {
    if (argc == 5 && !strcmp(argv[1], "train")) { train(argv[2], atoi(argv[3]), argv[4]); return 0; }
    if (argc == 4 && !strcmp(argv[3], "--literal")) parse_special = 0, argc = 3;
    if (argc == 3 && !strcmp(argv[1], "encode")) { load_model(argv[2]); encode_stdin(0); return 0; }
    if (argc == 3 && !strcmp(argv[1], "pieces")) { load_model(argv[2]); encode_stdin(1); return 0; }
    if (argc == 3 && !strcmp(argv[1], "decode")) { load_model(argv[2]); decode_stdin(); return 0; }
    if (argc == 3 && !strcmp(argv[1], "info"))   { load_model(argv[2]); info(); return 0; }
    fprintf(stderr, "usage:\n  %s train corpus.txt taille_vocab modele.bpe\n"
                    "  %s encode|pieces [--literal] | decode | info  modele.bpe\n", argv[0], argv[0]);
    return 1;
}
