import torch
from config import MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, DEVICE, DATA_FILE
from model import MyAI
from dataset import TextDataset


def load_model():
    ds = TextDataset(DATA_FILE, 1)
    model = MyAI(ds.vocab_size).to(DEVICE)
    ckpt = torch.load(MODEL_DIR / "model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ds.stoi, ds.itos


@torch.no_grad()
def generate(model, stoi, itos, prompt, max_new_tokens=MAX_NEW_TOKENS):
    full_prompt = f"<|user|> {prompt} <|bot|>"
    ids = [stoi.get(c, 0) for c in full_prompt]
    if not ids:
        ids = [0]

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    result = ""

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / TEMPERATURE

        if TOP_K > 0:
            v, _ = torch.topk(logits, TOP_K)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

        ch = itos[next_id.item()]
        result += ch

        if result.endswith("<|end|>"):
            break

    return result.replace("<|end|>", "").strip()


if __name__ == "__main__":
    model, stoi, itos = load_model()
    while True:
        prompt = input("You: ")
        if prompt.lower() in ("exit", "quit"):
            break
        print("AI:", generate(model, stoi, itos, prompt))