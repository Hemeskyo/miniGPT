import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


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


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, n_kv_heads, T, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, D)
        .reshape(B, n_kv_heads * n_rep, T, D)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, embed_dim, max_seq_len, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dims = embed_dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dims, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dims, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dims, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dims, max_seq_len=max_seq_len)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dims).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dims).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dims).transpose(1, 2)

        start_pos = 0 if kv_cache is None else kv_cache[0].shape[-2]

        cos, sin = self.rotary_emb(q, seq_len=start_pos + T)
        cos = cos[:, :, start_pos : start_pos + T, :]
        sin = sin[:, :, start_pos : start_pos + T, :]

        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)
        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        is_causal = True if (T > 1 and kv_cache is None) else False

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
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_seq_len
        # Multiple layers of inference
        self.blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    embed_dim, num_heads, num_kv_heads, ffn_dim, max_seq_len
                )
                for _ in range(num_layers)
            ]
        )

        # Normalization due to multiple layers
        self.ln_f = RMSNorm(embed_dim)

        # Output head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

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

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if idx.size(1) > self.max_seq_len:
            idx = idx[:, -self.max_seq_len :]
        kv_caches = None
        for _ in range(max_new_tokens):
            if idx.size(1) >= self.max_seq_len:
                break
            idx_cond = idx if kv_caches is None else idx[:, -1:]

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
