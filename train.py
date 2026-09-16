import os
import time
import torch
from config import (
    DATA_FILE, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR,
    BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP,
    EVAL_INTERVAL, EVAL_ITERS, SAVE_INTERVAL, SEED, DEVICE
)
from model import MyAI
from dataset import get_dataloader
from utils import set_seed, ensure_dir, count_lines

set_seed(SEED)
ensure_dir(MODEL_DIR)
ensure_dir(CHECKPOINT_DIR)
ensure_dir(LOG_DIR)

loader, ds = get_dataloader(DATA_FILE, BLOCK_SIZE, BATCH_SIZE)
vocab_size = ds.vocab_size

print(f"Vocab size: {vocab_size}")
print(f"Dataset size: {len(ds)}")
print(f"Device: {DEVICE}")

model = MyAI(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

print(f"Params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = next(iter(loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


step = 0
model.train()
start = time.time()

for epoch in range(EPOCHS):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        step += 1

        if step % 50 == 0:
            elapsed = time.time() - start
            print(f"epoch {epoch} step {step} loss {loss.item():.4f} time {elapsed:.1f}s")

        if step % EVAL_INTERVAL == 0:
            val = estimate_loss()
            print(f"--- eval step {step} loss {val:.4f} ---")

        if step % SAVE_INTERVAL == 0:
            ckpt_path = CHECKPOINT_DIR / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
            print(f"saved {ckpt_path}")

final_path = MODEL_DIR / "model.pt"
torch.save({"model": model.state_dict(), "chars": ds.chars}, final_path)
print(f"Done. Saved to {final_path}")