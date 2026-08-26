# Fine-tuning script (instruction-tuning the pretrained base model).
# Differs from train.py in 3 ways: it LOADS the pretrained weights instead of
# starting from scratch, uses a much LOWER learning rate + FEW steps (a gentle
# nudge, not a retrain), and trains on the instruction corpus.
import math
import os
import wandb
import torch
import array

import numpy as np
from model import MiniGPT
from tokenizer import BPETokenizerWrapper
from dataset import get_batch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Training will start on {device.type.upper()}")

if device.type == "cuda" and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
    use_amp = True
else:
    amp_dtype = torch.float32
    use_amp = False

print(f"Mixed precision (bf16): {use_amp}")

with open("instruct_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Text size is: {len(text):,} chars")

tokenizer = BPETokenizerWrapper()

# --- Token cache (tokenize once, reuse) ---
# Tokenize the (small) instruction corpus once and cache the ids as a compact
# uint16 .npy, so later runs reload in seconds instead of re-tokenizing.
TOKENS_CACHE = "instruct_tokens_.npy"

if os.path.exists(TOKENS_CACHE):
    # Fast path: reload the pre-tokenized corpus (int64 for the embedding lookup)
    print("Loading cached tokens...")
    data = torch.from_numpy(np.load(TOKENS_CACHE).astype(np.int64))
else:
    # Slow path (first run): tokenize in 5M-char chunks to keep RAM bounded,
    # then cache the result to disk for next time
    CHUNK = 5_000_000
    ids = array.array("i")
    for i in range(0, len(text), CHUNK):
        ids.extend(tokenizer.encode(text[i : i + CHUNK]))
        print(f"  tokenizing... {i // CHUNK + 1}/{len(text) // CHUNK + 1}")
    np.save(TOKENS_CACHE, np.array(ids, dtype=np.uint16))
    data = torch.tensor(ids, dtype=torch.long)
    print(f"Saved token cache -> {TOKENS_CACHE}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Fine-tuning hyperparameters: a LOW learning rate (~1/10 of pretraining) and
# FEW steps — a gentle nudge that adds instruction-following behaviour without
# overwriting the base model's knowledge (catastrophic forgetting).
wandb.init(
    project="Fineweb",
    config={
        "block_size": 512,
        "batch_size": 32,
        "embed_dim": 768,
        "ffn_dim": 3072,
        "num_heads": 12,
        "num_kv_heads": 4,
        "num_layers": 12,
        "max_lr": 5e-5,
        "min_lr": 5e-6,
        "warmup_steps": 100,
        "steps": 2000,
        "grad_clip": 1.0,
        "checkpoint_path": "fineweb_instruct.pt",
    },
)

config = wandb.config


def get_lr(it):
    if it < config.warmup_steps:
        return config.max_lr * (it + 1) / config.warmup_steps
    if it > config.steps:
        return config.min_lr

    decay_ratio = (it - config.warmup_steps) / (config.steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


# Model
raw_model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=config.block_size,
    embed_dim=config.embed_dim,
    ffn_dim=config.ffn_dim,
    num_heads=config.num_heads,
    num_kv_heads=config.num_kv_heads,
    num_layers=config.num_layers,
).to(device)

model = torch.compile(raw_model) if device.type == "cuda" else raw_model

# THE key difference from train.py: we start from the pretrained base model's
# weights (not from scratch), then keep training gently on the instruction data.
ckpt = torch.load("fineweb_gpt.pt", map_location=device)
raw_model.load_state_dict(ckpt["model"])
print("Loaded base model fineweb_gpt.pt for fine-tuning.")


optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")


@torch.no_grad()
def estimate_loss(eval_iters=20):
    model.eval()
    out = {}
    for split, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split_data, config.block_size, config.batch_size, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


print("\n--- Starting training! ---")

best_val_loss = float("inf")
patience = 5
patience_counter = 0

for step in range(config.steps):

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch(train_data, config.block_size, config.batch_size, device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        logits, loss, _ = model(x, y)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    optimizer.step()

    metrics = {
        "train_batch_loss": loss.item(),
        "learning_rate": lr,
    }

    if step % 200 == 0 or step == config.steps - 1:
        losses = estimate_loss()
        metrics["eval/train_loss"] = losses["train"]
        metrics["eval/val_loss"] = losses["val"]
        print(
            f"(Step {step:4d} / {config.steps} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f})"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            patience_counter = 0
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "config": dict(config),
                    "step": step,
                    "val_loss": best_val_loss,
                },
                config.checkpoint_path,
            )
            print(f"Best model (val {best_val_loss:.4f}) saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping : val is stagnating since {patience} evals")
                break

    wandb.log(metrics)


wandb.finish()
