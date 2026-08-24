"""
Nettoyage + assemblage du corpus Dostoievski.

Prend une liste de fichiers texte bruts, corrige leurs artefacts d'extraction,
et les assemble en UN seul corpus propre : dostoievski_corpus_clean.txt

Artefacts geres (certains fichiers en ont, d'autres non -> chaque etape est
un no-op si l'artefact est absent) :
  1. BOM UTF-8 en tete de fichier.
  2. En-tete / pied de licence Project Gutenberg (si present).
  3. Mots coupes par un tiret en fin de ligne ("or-\nganise" -> "organise").
  4. Retour a la ligne "dur" (texte wrappe a ~60 caracteres par la mise en page).
  5. Paragraphes separes par des suites d'espaces (>= 3) au lieu de lignes vides
     (cas des fichiers ou tout le texte est sur une seule ligne).
  6. Espaces multiples ecrases en une seule.

Les fichiers sources ne sont JAMAIS modifies.
"""

import re

SOURCES = [
    "dostoievski_corpus.txt",   # les 8 oeuvres deja concatenees (wrap + cesure)
    "dostoievski-10.txt",       # nouveau (une seule ligne + espaces)
    "dostoievski-11.txt",       # nouveau (une seule ligne + espaces)
]
DST = "dostoievski_corpus_clean.txt"


def strip_gutenberg(text):
    """Retire tout ce qui precede *** START ... *** et suit *** END ... ***."""
    start = re.search(r"\*\*\*\s*START OF TH.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\*\s*END OF TH.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if end:
        text = text[: end.start()]
    return text


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg(text)

    # #3 : recoller les mots coupes en fin de ligne
    text = re.sub(r"(\w)-\n[ \t]*(\w)", r"\1\2", text)

    # #5 : une suite de >= 3 espaces marque une frontiere de paragraphe
    text = re.sub(r"[ \t]{3,}", "\n\n", text)

    # Decoupe en paragraphes (lignes vides), puis de-wrap chacun
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for p in paragraphs:
        p = re.sub(r"\s*\n\s*", " ", p)   # #4 : sauts de ligne internes -> espace
        p = re.sub(r"[ \t]+", " ", p)     # #6 : espaces multiples -> une seule
        p = p.strip()
        if p:
            out.append(p)
    return out


all_paragraphs = []
report = []
for src in SOURCES:
    with open(src, encoding="utf-8-sig") as f:   # utf-8-sig retire le BOM (#1, #2 via strip)
        raw = f.read()
    paras = clean(raw)
    all_paragraphs.extend(paras)
    report.append((src, len(raw), len(paras)))

result = "\n\n".join(all_paragraphs) + "\n"

with open(DST, "w", encoding="utf-8") as f:
    f.write(result)

print("=== Nettoyage + assemblage ===")
for src, nchars, nparas in report:
    print(f"  {src:28s} {nchars:>10,} chars -> {nparas:>7,} paragraphes")
print(f"--> {DST} : {len(result):,} chars, {len(all_paragraphs):,} paragraphes")
