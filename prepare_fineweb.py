# Streams a subset of FineWeb-Edu (high-quality educational web text) into a
# plain-text corpus. streaming=True => we pull only N_DOCS docs on the fly
# instead of downloading the whole ~10B-token shard.
from datasets import load_dataset

N_DOCS = 400_000
OUT = "fineweb_corpus.txt"

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
)
written = 0
with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        t = ex["text"].strip()
        if t:
            f.write(t + "\n\n")
            written += 1
        if written >= N_DOCS:
            break
print(f"{written} docs -> {OUT}")
