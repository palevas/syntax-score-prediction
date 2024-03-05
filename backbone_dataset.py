import os
import json
import pydicom
import numpy as np
import torch

from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset


class SyntaxDataset(Dataset):
    def __init__(
        self, 
        root: str,      # dataset dir
        meta: str,      # metadata
        train: bool,    # training mode
        length: int,    # video length
        label: str,     # label field name
        artery_bin,     # arterym, 0 is left, 1 is right
        transform: Optional[Callable] = None, 
    ) -> None:
        self.root = root
        self.train = train
        self.length = length
        self.label = label
        self.transform = transform
        with open(os.path.join(root, meta)) as f:
            dataset = json.load(f)

        if artery_bin is not None:
            assert artery_bin in (0, 1)
            dataset = [rec for rec in dataset if rec["artery"] == artery_bin]

        if self.train:
            self.dataset = [rec for rec in dataset if rec[self.label] > 0]
            self.negative_dataset = [rec for rec in dataset if rec[self.label] == 0]

            for rec in self.dataset:
                rec["weight"] = 1.0
            for rec in self.negative_dataset:
                rec["weight"] = 1.0
        else:
            self.dataset = dataset
            self.negative_dataset = None
            for rec in self.dataset:
                rec["weight"] = 1.0


    def __len__(self):
        coef = 2 if self.negative_dataset else 1
        return coef * len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        if self.negative_dataset:
            if idx % 2 == 0:
                idx = idx // 2
                rec = self.dataset[idx]
            else:
                idx = torch.randint(low=0, high=len(self.negative_dataset), size=(1,))
                rec = self.negative_dataset[idx]
        else:
            rec = self.dataset[idx]

        path = rec["path"]
        weight = rec["weight"]
        full_path = os.path.join(self.root, path)
        video = pydicom.dcmread(full_path).pixel_array # Time, H, W
        label = torch.tensor([int(rec[self.label] > 0)], dtype=torch.float32)

        while len(video) < self.length:
            video = np.concatenate([video, video])
        t = len(video)
        if self.train:
            begin = torch.randint(low=0, high=t-self.length+1, size=(1,))
            end = begin + self.length
            video = video[begin:end, :, :]
        else:
            # begin = (t - self.length) // 2
            # end = begin + self.length
            # video = video[begin:end, :, :]
            video = video # src length video, valid batch_size=1
        
        video = torch.tensor(np.stack([video, video, video], axis=-1))

        if self.transform is not None:
            video = self.transform(video)

        return video, label, weight, path


