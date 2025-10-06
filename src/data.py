import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

class AutoRegressiveDataset(Dataset):
    def __init__(self, data_path='../data/', max_length=2048, dataset_size=10000,
                 tokenizer_name='gpt2', device='cuda'):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = self._load_autoregressive(
            data_path=data_path, max_length=self.max_length, dataset_size=dataset_size
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = self.dataset[idx]
        pad_id = self.tokenizer.pad_token_id or 0
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [pad_id] * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[: self.max_length]
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        return x, x.clone().detach()

    def _load_autoregressive(self, data_path='../data/', max_length=2048, dataset_size=10000):
        os.makedirs(data_path, exist_ok=True)
        dataset_path = os.path.join(data_path, 'autoregressive_dataset.pth')
        if os.path.exists(dataset_path):
            dataset = torch.load(dataset_path)
        else:
            ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                              split=f"train[0:{dataset_size}]")
            ds = ds.shuffle(seed=123)
            dataset = [self.tokenizer.encode(row['content'])[:max_length]
                       for row in tqdm(ds, desc="Tokenizing")]
            torch.save(dataset, dataset_path)
        return dataset

def make_loaders(tokenizer_name: str, data_root: str, seq_len: int, batch_size: int,
                 dataset_size: int, device: str):
    full = AutoRegressiveDataset(
        data_path=data_root,
        max_length=seq_len,
        dataset_size=dataset_size,
        tokenizer_name=tokenizer_name,
        device=device,
    )
    n = len(full)
    split = int(0.9 * n)
    idx_train = list(range(0, split))
    idx_val   = list(range(split, n))
    train_subset = torch.utils.data.Subset(full, idx_train)
    val_subset   = torch.utils.data.Subset(full, idx_val)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader
