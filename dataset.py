import numpy as np
import os
import torch
from torch.utils.data import Dataset

class CausalPairFeatureDataset(Dataset):
    def __init__(self, feat_dir, split="train_cda"):
        self.feat_dir = feat_dir
        self.img_pool = np.load(os.path.join(feat_dir, f"{split}_img_pool.npy"))
        self.anchor_text_pool = np.load(os.path.join(feat_dir, f"{split}_anchor_text_pool.npy"))
        self.anchor_text_seq = np.load(os.path.join(feat_dir, f"{split}_anchor_text_seq.npy"))
        self.positive_text_pool = np.load(os.path.join(feat_dir, f"{split}_positive_text_pool.npy"))
        self.positive_text_seq = np.load(os.path.join(feat_dir, f"{split}_positive_text_seq.npy"))
        self.hardneg_text_pool = np.load(os.path.join(feat_dir, f"{split}_hardneg_text_pool.npy"))
        self.hardneg_text_seq = np.load(os.path.join(feat_dir, f"{split}_hardneg_text_seq.npy"))
        self.labels = np.load(os.path.join(feat_dir, f"{split}_labels.npy"))
        
        strategy_weights_path = os.path.join(feat_dir, f"{split}_strategy_weights.npy")
        if os.path.exists(strategy_weights_path):
            self.strategy_weights = np.load(strategy_weights_path)
        else:
            self.strategy_weights = np.ones(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_pool": self.img_pool[idx],
            "anchor_text_pool": self.anchor_text_pool[idx],
            "anchor_text_seq": self.anchor_text_seq[idx],
            "positive_text_pool": self.positive_text_pool[idx],
            "positive_text_seq": self.positive_text_seq[idx],
            "hardneg_text_pool": self.hardneg_text_pool[idx],
            "hardneg_text_seq": self.hardneg_text_seq[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "strategy_weight": torch.tensor(self.strategy_weights[idx], dtype=torch.float32)
        }


class StandardFeatureDataset(Dataset):
    def __init__(self, feat_dir, split):
        self.feat_dir = feat_dir
        self.img_pool = np.load(os.path.join(feat_dir, f"{split}_img_pool.npy"))
        self.text_pool = np.load(os.path.join(feat_dir, f"{split}_text_pool.npy"))
        self.text_seq = np.load(os.path.join(feat_dir, f"{split}_text_seq.npy"))
        self.labels = np.load(os.path.join(feat_dir, f"{split}_labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_pool": self.img_pool[idx],
            "text_pool": self.text_pool[idx],
            "text_seq": self.text_seq[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def custom_collate_fn(batch):
    collated = {
        "img_pool": torch.tensor(np.stack([x["img_pool"] for x in batch]), dtype=torch.float32),
        "anchor_text_pool": torch.tensor(np.stack([x["anchor_text_pool"] for x in batch]), dtype=torch.float32) if "anchor_text_pool" in batch[0] else None,
        "anchor_text_seq": torch.tensor(np.stack([x["anchor_text_seq"] for x in batch]), dtype=torch.float32) if "anchor_text_seq" in batch[0] else None,
        "positive_text_pool": torch.tensor(np.stack([x["positive_text_pool"] for x in batch]), dtype=torch.float32) if "positive_text_pool" in batch[0] else None,
        "positive_text_seq": torch.tensor(np.stack([x["positive_text_seq"] for x in batch]), dtype=torch.float32) if "positive_text_seq" in batch[0] else None,
        "hardneg_text_pool": torch.tensor(np.stack([x["hardneg_text_pool"] for x in batch]), dtype=torch.float32) if "hardneg_text_pool" in batch[0] else None,
        "hardneg_text_seq": torch.tensor(np.stack([x["hardneg_text_seq"] for x in batch]), dtype=torch.float32) if "hardneg_text_seq" in batch[0] else None,
        "text_pool": torch.tensor(np.stack([x["text_pool"] for x in batch]), dtype=torch.float32) if "text_pool" in batch[0] else None,
        "text_seq": torch.tensor(np.stack([x["text_seq"] for x in batch]), dtype=torch.float32) if "text_seq" in batch[0] else None,
        "label": torch.stack([x["label"] for x in batch])
    }
    
    if "strategy_weight" in batch[0]:
        collated["strategy_weight"] = torch.stack([x["strategy_weight"] for x in batch])
        
    return collated