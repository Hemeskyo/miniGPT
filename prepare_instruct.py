from datasets import load_dataset

OUT = "instruct_corpus.txt"

ds = load_dataset("yahma/alpaca-cleaned", split="train")

with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        instr = ex["instruction"].strip()
        inp = ex["input"].strip()
        out = ex["output"].strip()
        text = f"### Instruction:\n{instr}\n"
        if inp:
            text += f"\n### Input:\n{inp}\n"
        text += f"\n### Response:\n{out}\n\n"
        f.write(text)

print(f"Wrote {len(ds)} instructions -> {OUT}")
