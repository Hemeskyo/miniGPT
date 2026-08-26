# Interactive CLI to generate text from a trained model.
# Loads a checkpoint and rebuilds the EXACT architecture from the saved config
# (no hard-coded hyperparameters), then generates token-by-token with the KV-cache.
import os
import torch
from model import MiniGPT
from tokenizer import BPETokenizerWrapper

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


checkpoint_path = "fineweb_gpt.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Weight not found. Please train the model before")

ckpt = torch.load(checkpoint_path, map_location=device)
cfg = ckpt["config"]
tokenizer = BPETokenizerWrapper()

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=cfg["block_size"],
    embed_dim=cfg["embed_dim"],
    ffn_dim=cfg["ffn_dim"],
    num_heads=cfg["num_heads"],
    num_kv_heads=cfg["num_kv_heads"],
    num_layers=cfg["num_layers"],
).to(device)

model.load_state_dict(ckpt["model"])

model.eval()


print("=" * 60)
print("Welcome to Fineweb GPT (CLI mode)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

print("Type 'exit' to exit")
print("=" * 60)

# Sampling settings: lower temperature / top_k / top_p = safer & more coherent,
# higher = more diverse (see model.generate for how each one filters the logits)
temperature = 0.9
top_k = 20
top_p = 0.95
max_tokens = 500
repetion_penalty = 1.3

while True:
    try:
        prompt = input("\n[Prompt] > ")
        if prompt.strip().lower() in ["exit"]:
            print("Fermeture de la session.")
            break
        if not prompt.strip():
            continue

        raw_ids = tokenizer.encode(prompt)
        idx = torch.tensor([raw_ids], dtype=torch.long, device=device)

        print(f"\n--- Generation started (Temp: {temperature} | Top-K: {top_k} ---)")

        with torch.no_grad():
            generated_idxs = model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetion_penalty,
            )

        result = tokenizer.decode(generated_idxs[0].tolist())
        print(f"\n{result}\n")
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n Session interrompue.")
        break
