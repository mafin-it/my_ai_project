import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from config import MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, DEVICE, HOST, PORT, DATA_FILE
from model import MyAI
from dataset import TextDataset

app = FastAPI()

ds = TextDataset(DATA_FILE, 1)
model = MyAI(ds.vocab_size).to(DEVICE)

ckpt_path = MODEL_DIR / "model.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print("Model loaded")

model.eval()


class Msg(BaseModel):
    text: str


@torch.no_grad()
def generate_text(prompt, max_new_tokens=MAX_NEW_TOKENS):
    full_prompt = f"<|user|> {prompt} <|bot|>"
    ids = [ds.stoi.get(c, 0) for c in full_prompt]
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

        ch = ds.itos[next_id.item()]
        result += ch

        if result.endswith("<|end|>"):
            break

    return result.replace("<|end|>", "").strip()


@app.post("/generate")
def generate(msg: Msg):
    reply = generate_text(msg.text)
    return {"reply": reply}


@app.get("/")
def index():
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)