import os
import torch
from model import MiniGPT
from tokenizer import BPETokenizerWrapper

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


checkpoint_path = "tinystories_gpt.pt"
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
print("Welcome to Tinystories GPT (CLI mode)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

print("Type 'exit' to exit")
print("=" * 60)

temperature = 0.6
top_k = 20
max_tokens = 200

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

        print(f"\n--- Génération en cours (Temp: {temperature} | Top-K: {top_k} ---)")

        with torch.no_grad():
            generated_idxs = model.generate(
                idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k
            )

        result = tokenizer.decode(generated_idxs[0].tolist())
        print(f"\n{result}\n")
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n Session interrompue.")
        break
