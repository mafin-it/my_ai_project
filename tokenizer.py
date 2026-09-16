class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.chars = data["chars"]
        obj.vocab_size = len(obj.chars)
        obj.stoi = {c: i for i, c in enumerate(obj.chars)}
        obj.itos = {i: c for c, i in obj.stoi.items()}
        return obj


if __name__ == "__main__":
    text = "привет как дела"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    print("vocab:", tok.vocab_size)
    print("ids:", ids)
    print("back:", tok.decode(ids))