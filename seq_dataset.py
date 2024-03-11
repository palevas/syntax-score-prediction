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
        artery: str,    # left or right artery
        inference: bool = False,
        transform: Optional[Callable] = None, 
    ) -> None:
        self.root = root
        self.train = train
        self.length = length
        self.label = label
        self.artery = artery
        self.inference = inference
        self.transform = transform
        with open(os.path.join(root, meta)) as f:
            dataset = json.load(f)

        if not self.inference:
            dataset = [rec for rec in dataset if len(rec[f"videos_{artery}"]) > 0]

        if self.train:
            self.dataset = [rec for rec in dataset if rec[f"syntax_{artery}"] > 0]
            self.negative_dataset = [rec for rec in dataset if rec[f"syntax_{artery}"] == 0]
            assert len(self.dataset) + len(self.negative_dataset) == len(dataset)

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

        weight = rec["weight"]
        sid = rec["study_id"]
        label = torch.tensor([int(rec[self.label] > 0)], dtype=torch.float32)
        target = torch.tensor([np.log(1.0+rec[self.label])], dtype=torch.float32)

        nv = len(rec[f"videos_{self.artery}"])
        if self.inference:
            if nv == 0:
                return 0, label, target, weight, sid
            seq = range(nv)
        else:
            seq = torch.randint(low=0, high=nv, size = (4,))

        videos = []
        for vi in seq:
            video_rec = rec[f"videos_{self.artery}"][vi]
            path = video_rec["path"]
            full_path = os.path.join(self.root, path)
            video = pydicom.dcmread(full_path).pixel_array # Time, HW or WH
            while len(video) < self.length:
                video = np.concatenate([video, video])
            t = len(video)
            if self.train:
                begin = torch.randint(low=0, high=t-self.length+1, size=(1,))
                end = begin + self.length
                video = video[begin:end, :, :]
            else:
                begin = (t - self.length) // 2
                end = begin + self.length
                video = video[begin:end, :, :]
            
            video = torch.tensor(np.stack([video, video, video], axis=-1))

            if self.transform is not None:
                video = self.transform(video)
            videos.append(video)
        videos = torch.stack(videos, dim=0)

        return videos, label, target, weight, sid
