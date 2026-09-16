import torch
from torch.utils.data import Dataset, DataLoader
from config import DATA_FILE, BLOCK_SIZE, BATCH_SIZE


class TextDataset(Dataset):
    def __init__(self, path, block_size):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.block_size = block_size
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def get_dataloader(path=DATA_FILE, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, shuffle=True):
    dataset = TextDataset(path, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader, dataset


if __name__ == "__main__":
    loader, ds = get_dataloader()
    print("Vocab size:", ds.vocab_size)
    print("Dataset size:", len(ds))
    x, y = next(iter(loader))
    print("x:", x.shape)
    print("y:", y.shape)