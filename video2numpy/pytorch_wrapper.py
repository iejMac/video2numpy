"""Wrapper around FrameReader so it appears to be a PyTorch DataLoader."""
import torch
from .frame_reader import FrameReader


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, frame_reader: FrameReader):
        self.frame_reader = frame_reader

    def __len__(self):
        return 10**9

    def __iter__(self):
        for item in self.frame_reader:
            yield item


def fr2dl(frame_reader):
    dataset = Dataset(frame_reader)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0]
    )
    return dataloader
