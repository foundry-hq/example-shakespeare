"""
Train a character-level transformer on Tiny Shakespeare.

A minimal example for Backprop — trains in ~3 minutes on an L4 GPU.
Writes metrics to $BP_RUN_DIR/metrics.jsonl and saves generated samples.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
BLOCK_SIZE = 128
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 50
LOG_INTERVAL = 10
SAMPLE_INTERVAL = 1000
SAMPLE_LENGTH = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/input.txt")


def load_data() -> str:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


text = load_data()
chars = sorted(set(text))
VOCAB_SIZE = len(chars)
print(f"Dataset: {len(text):,} chars, {VOCAB_SIZE} unique tokens")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=DEVICE))
        x = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------
BP_RUN_DIR = Path(os.environ.get("BP_RUN_DIR"))
METRICS_FILE = BP_RUN_DIR / "metrics.jsonl"
OUTPUT_DIR = BP_RUN_DIR / "outputs"
BP_RUN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_metric(name: str, value: float, step: int = None, **kwargs):
    record = {"ts": time.time(), "name": name, "value": value}
    if step is not None:
        record["step"] = step
    record.update(kwargs)
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    model = CharTransformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    t0 = time.time()

    for step in range(MAX_STEPS):
        # Evaluate periodically
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"{elapsed:.1f}s"
            )
            log_metric("train/loss", round(losses["train"], 4), step=step)
            log_metric("eval/loss", round(losses["val"], 4), step=step)
            log_metric("train/elapsed_s", round(elapsed, 1), step=step)

        # Generate sample periodically
        if step > 0 and step % SAMPLE_INTERVAL == 0:
            ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            sample = decode(model.generate(ctx, SAMPLE_LENGTH)[0].tolist())
            sample_path = OUTPUT_DIR / f"sample_step_{step}.txt"
            sample_path.write_text(sample)
            print(f"\n--- Sample at step {step} ---\n{sample[:200]}...\n")

        # Training step
        _, loss = model(*get_batch("train"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Log training loss every N steps
        if step % LOG_INTERVAL == 0:
            log_metric("train/loss", round(loss.item(), 4), step=step)

    # Final generation
    print("\n=== Final generation ===\n")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample = decode(model.generate(ctx, 1000)[0].tolist())
    print(sample)

    (OUTPUT_DIR / "final_sample.txt").write_text(sample)

    # Save model checkpoint
    model_path = OUTPUT_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    total_time = time.time() - t0
    print(f"Training complete in {total_time:.1f}s")
    log_metric("train/total_time_s", round(total_time, 1), step=MAX_STEPS)


if __name__ == "__main__":
    main()
