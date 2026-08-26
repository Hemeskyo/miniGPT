import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- RMSNorm (Llama/Qwen-style normalization) ---
# Normalizes each token vector by its root-mean-square (no mean subtraction,
# no bias), then rescales with a learned weight. Cheaper than LayerNorm and
# just as stable.
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


# --- Rotary Position Embeddings (RoPE) ---
# Instead of a learned position table, positions are injected by ROTATING the
# query/key vectors by an angle that depends on their position. This encodes
# RELATIVE position (attention between i and j depends only on i-j) and works
# cleanly with the KV-cache. cos/sin are precomputed once for every position.
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        t = torch.arange(max_seq_len, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return cos, sin


# Helpers that actually apply the RoPE rotation to q and k.
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# --- GQA helper ---
# Repeats each key/value head n_rep times so a few KV heads can be shared
# across many query heads (see GroupedQueryAttention).
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, n_kv_heads, T, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, D)
        .reshape(B, n_kv_heads * n_rep, T, D)
    )


# --- Grouped-Query Attention (GQA) + KV-cache ---
# Many query heads share fewer key/value heads (e.g. 12 Q for 4 KV). Fewer KV
# projections => a smaller KV-cache and less memory bandwidth at inference, for
# almost no quality loss. During generation, past keys/values are cached so each
# new token only computes its own attention: O(n) decoding instead of O(n^2).
class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, embed_dim, max_seq_len, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads  # how many Q heads per KV head
        self.head_dims = embed_dim // num_heads
        self.dropout_p = dropout

        # --- QK-norm ---
        # An RMSNorm applied to each head's queries and keys (over head_dim)
        # before attention. Recent trick (Qwen3, Gemma2) that stabilizes training.
        self.q_norm = RMSNorm(self.head_dims)
        self.k_norm = RMSNorm(self.head_dims)

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dims, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dims, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dims, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dims, max_seq_len=max_seq_len)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # Project to Q/K/V and split into heads -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dims).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dims).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dims).transpose(1, 2)

        # QK-norm: normalize each head's q/k before RoPE (see __init__)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Position offset: 0 normally, or the cached length during generation
        start_pos = 0 if kv_cache is None else kv_cache[0].shape[-2]

        cos, sin = self.rotary_emb(q, seq_len=start_pos + T)
        cos = cos[:, :, start_pos : start_pos + T, :]
        sin = sin[:, :, start_pos : start_pos + T, :]

        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV-cache: append this step's keys/values to the ones from past tokens
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)
        # Expand the shared KV heads up to the number of query heads
        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        # Causal mask only for the full-prompt pass; a single cached token may
        # attend to all past positions, so no mask is needed there.
        is_causal = True if (T > 1 and kv_cache is None) else False

        # Fused, memory-efficient attention (applies the causal mask internally)
        out = F.scaled_dot_product_attention(
            q,
            k_rep,
            v_rep,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out, new_kv_cache


# --- SwiGLU feed-forward ---
# Gated MLP: silu(w1 x) * (w3 x), then project back with w2. Beats a plain
# ReLU/GELU MLP at the same parameter budget. hidden_dim uses the 2/3 rule and
# is rounded to a multiple of 8 for hardware efficiency.
class SwiGLUFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.2):
        super().__init__()

        hidden_dim = int(2 * ffn_dim / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# --- Transformer block (pre-norm + residuals) ---
#   x = x + Attention(RMSNorm(x))
#   x = x + SwiGLU(RMSNorm(x))
# Normalizing BEFORE each sublayer (pre-norm) plus residual connections keeps
# gradients stable in deep stacks. Same block shape as Llama/Qwen.
class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, ffn_dim, max_seq_len):
        super().__init__()
        self.sa = GroupedQueryAttention(num_heads, num_kv_heads, embed_dim, max_seq_len)
        self.ffwd = SwiGLUFeedForward(embed_dim, ffn_dim)
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)

    def forward(self, x, kv_cache=None):
        sa_out, new_kv_cache = self.sa(self.ln1(x), kv_cache=kv_cache)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv_cache


# --- The full model ---
# token embedding -> N transformer blocks -> final RMSNorm -> output head.
# The output head shares its weights with the embedding (weight tying = fewer
# params). forward() also returns the cross-entropy loss when targets are given.
class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len=256,
        embed_dim=128,
        ffn_dim=512,
        num_heads=4,
        num_kv_heads=2,
        num_layers=3,
    ):
        super().__init__()
        # Token embedding: maps each token id to a vector
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_seq_len
        # The stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    embed_dim, num_heads, num_kv_heads, ffn_dim, max_seq_len
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization before the output head
        self.ln_f = RMSNorm(embed_dim)

        # Output head: hidden vector -> logits over the vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: embedding and output head share the same matrix
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.size()

        x = self.token_embedding(idx)
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        # Training path: compare predictions to the next-token targets
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_kv_caches

    # --- Autoregressive generation (the KV-cache in action) ---
    # First call encodes the whole prompt and fills the cache; afterwards only
    # the LAST token is fed in (idx[:, -1:]) and attention reuses the cache.
    # Sampling pipeline applied to the last position's logits, in order:
    # repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample.
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
    ):
        # Crop a prompt that is already longer than the context window
        if idx.size(1) > self.max_seq_len:
            idx = idx[:, -self.max_seq_len :]
        kv_caches = None
        for _ in range(max_new_tokens):
            # Stop before running past the RoPE / KV window
            if idx.size(1) >= self.max_seq_len:
                break
            idx_cond = idx if kv_caches is None else idx[:, -1:]

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)

            # Keep only the LAST position's scores -> (B, vocab) = next-token logits
            logits = logits[:, -1, :]

            # Repetition penalty: push down the logit of every token already used
            # (set(idx[b]) = the token history) so the model stops looping.
            # Divide if the logit is >0, multiply if <0 — either way it moves the
            # score toward "less likely", whatever its sign.
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    for tok in set(idx[b].tolist()):
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus): keep the smallest set of tokens whose cumulative
            # probability reaches top_p, then drop the rest
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                remove = remove.scatter(1, sorted_idx, remove)
                logits = logits.masked_fill(remove, float("-inf"))
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
