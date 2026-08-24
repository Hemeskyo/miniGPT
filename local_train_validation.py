import math
import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import BPETokenizerWrapper
from dataset import get_batch

# 1. Selection du device (MPS pour Mac M1)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Running DRY RUN on {device.type.upper()} ---")

# 2. Charger un extrait de données ou le corpus complet
with open("dostoievski_corpus_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizerWrapper()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
train_data, val_data = data[:10000], data[10000:12000]  # Petit slice pour aller vite

# 3. Hyperparamètres réduits au minimum
BLOCK_SIZE = 64
BATCH_SIZE = 8
STEPS = 50

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=BLOCK_SIZE,
    embed_dim=64,
    ffn_dim=256,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Total parameters (Dry Run): {sum(p.numel() for p in model.parameters()):,}")

# 4. Boucle de validation rapide
model.train()
for step in range(STEPS):
    x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)

    # MPS supporte mal le mixed precision dans certains cas, on reste en float32 pur sur Mac
    logits, loss, _ = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step:2d}/{STEPS} | Loss: {loss.item():.4f}")

# 5. Test rapide du KV Cache / Génération
print("\n--- Testing Generation with KV Cache ---")
prompt_test = "La ville"
raw_ids = tokenizer.encode(prompt_test)
idx = torch.tensor([raw_ids], dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    generated_idx = model.generate(idx, max_new_tokens=30, temperature=0.8, top_k=5)

print(f"Prompt: {prompt_test}")
print(f"Generated: {tokenizer.decode(generated_idx[0].tolist())}")
print("\n🟢 ALL SYSTEMS GO! Le pipeline est prêt pour le RunPod.")
