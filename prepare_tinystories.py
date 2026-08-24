from datasets import load_dataset

N_STORIES = 400_000  # ~35-40M tokens, ~130 Mo (15x ton corpus Dosto)
OUT = "tinystories_corpus.txt"

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

written = 0
with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        t = ex["text"].strip()
        if t:
            f.write(t + "\n\n")  # ligne vide = séparateur entre histoires
            written += 1
        if written >= N_STORIES:
            break

print(f"{written} histoires écrites dans {OUT}")
