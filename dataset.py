import torch
import random
import itertools
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from cogan.hk_polyu import class_filter


class ContrastiveDataset(Dataset):
    def __init__(self,
                 nir_dataset,
                 vis_dataset,
                 positive_prob=0.5):
        super().__init__()
        self.nir = nir_dataset
        self.vis = vis_dataset
        self.positive_prob = positive_prob
        self.h = {}

        print(len(self.nir))
        print(len(self.vis))

        for i in range(len(self.vis)):
            vis_address = self.vis.imgs[i][0]
            subject_id = vis_address.split('/')[-2]
            for j in range(len(self.nir.imgs)):
                nir_address = self.nir.imgs[j][0]
                if subject_id in nir_address:
                    if i in self.h:
                        self.h[i].append(j)
                    else:
                        self.h[i] = [j]

    def __getitem__(self, index):
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob
        img_0, label_0 = self.vis[index]

        nir_samples = self.h[index]
        if same_class:
            rnd_idx = random.randint(0, len(nir_samples) - 1)
            index_1 = nir_samples[rnd_idx]
            img_1, label_1 = self.nir[index_1]
        else:
            while True:
                index_1 = random.randint(0, self.__len__() - 1)
                if index_1 not in self.h[index]:
                    img_1, label_1 = self.nir[index_1]
                    break
        # print(same_class, '<<')
        # plot_tensor([img_0, img_1])
        return img_0, img_1, same_class

    def __len__(self):
        return min(len(self.nir), len(self.vis))


class AllVsAllDataset(Dataset):
    def __init__(self,
                 nir_dataset,
                 vis_dataset,
                 sync_capture):
        super().__init__()
        self.nir = nir_dataset
        self.vis = vis_dataset

        print(len(self.nir))
        print(len(self.vis))

        if sync_capture:
            self.pairs = _sync_capture_pairs(self.vis, self.nir)

    def __getitem__(self, index):
        vis_idx, nir_idx, label = self.pairs[index]
        return self.vis[vis_idx], self.nir[nir_idx], label

    def __len__(self):
        return len(self.pairs)


def get_dev_dataset(batch_size,
                    vis_dir,
                    nir_dir,
                    valid_classes_fp,
                    vis_mean_fp,
                    vis_std_fp,
                    nir_mean_fp,
                    nir_std_fp):
    vis_mean = np.loadtxt(vis_mean_fp) / 255.0
    vis_std = np.loadtxt(vis_std_fp) / 255.0
    nir_mean = np.loadtxt(nir_mean_fp) / 255.0
    nir_std = np.loadtxt(nir_std_fp) / 255.0

    if valid_classes_fp is not None:
        valid_classes = np.loadtxt(valid_classes_fp, dtype=str)
        is_valid_file_fn = class_filter(valid_classes)
    else:
        is_valid_file_fn = None
    # Is there a way to do synchronized augmentation?
    # i.e. RandomRotation(15)/RandomHorizontalFlip()
    vis_dataset = datasets.ImageFolder(
        root=vis_dir,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Pad(16),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=vis_mean, std=vis_std)
        ]),
        is_valid_file=is_valid_file_fn
    )

    nir_dataset = datasets.ImageFolder(
        root=nir_dir,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Pad(16),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=nir_mean, std=nir_std)
        ]),
        is_valid_file=is_valid_file_fn
    )

    data_loader = torch.utils.data.DataLoader(
        ContrastiveDataset(nir_dataset, vis_dataset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    return data_loader


def get_test_dataset(batch_size,
                     vis_dir,
                     nir_dir,
                     valid_classes_fp,
                     vis_mean_fp,
                     vis_std_fp,
                     nir_mean_fp,
                     nir_std_fp,
                     sync_capture):
    vis_mean = np.loadtxt(vis_mean_fp) / 255.0
    vis_std = np.loadtxt(vis_std_fp) / 255.0
    nir_mean = np.loadtxt(nir_mean_fp) / 255.0
    nir_std = np.loadtxt(nir_std_fp) / 255.0

    valid_classes = np.loadtxt(valid_classes_fp, dtype=str)
    is_valid_file_fn = class_filter(valid_classes)

    vis_dataset = datasets.ImageFolder(
        root=vis_dir,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=vis_mean, std=vis_std)
        ]),
        is_valid_file=is_valid_file_fn
    )

    nir_dataset = datasets.ImageFolder(
        root=nir_dir,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=nir_mean, std=nir_std)
        ]),
        is_valid_file=is_valid_file_fn
    )

    data_loader = torch.utils.data.DataLoader(
        AllVsAllDataset(nir_dataset, vis_dataset, sync_capture),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    return data_loader


def _sync_capture_pairs(vis_dataset, nir_dataset):
    assert len(vis_dataset) == len(nir_dataset)
    classes = set([class_id for _, class_id in vis_dataset.imgs])
    pairs = []
    for class_id in classes:
        vis_dataset_idx = [i
                           for i, sample in enumerate(vis_dataset.imgs)
                           if sample[1] == class_id]
        nir_dataset_idx = [i
                           for i, sample in enumerate(nir_dataset.imgs)
                           if sample[1] == class_id]
        combinations = itertools.combinations(range(len(vis_dataset_idx)), r=2)
        pairs += [(vis_dataset_idx[vis_idx], nir_dataset_idx[nir_idx], 1)
                  for vis_idx, nir_idx in combinations]
        vis_dataset_idx = [i
                           for i, sample in enumerate(vis_dataset.imgs)
                           if sample[1] != class_id]
        pairs += [(vis_idx, nir_idx, 0)
                  for nir_idx in nir_dataset_idx
                  for vis_idx in vis_dataset_idx]
    return pairs
