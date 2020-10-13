import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
            photo_address = self.vis.imgs[i][0]
            subject_id = photo_address.split('/')[-2]
            for j in range(len(self.nir.imgs)):
                print_address = self.nir.imgs[j][0]
                if subject_id in print_address:
                    if i in self.h:
                        self.h[i].append(j)
                    else:
                        self.h[i] = [j]

    def __getitem__(self, index):
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob
        img_0, label_0 = self.vis[index]

        print_samples = self.h[index]
        if same_class:
            rnd_idx = random.randint(0, len(print_samples) - 1)
            index_1 = print_samples[rnd_idx]
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


def get_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    photo_dataset = datasets.ImageFolder(
        args.photo_folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.Pad(16),
            transforms.RandomCrop(256),
            transforms.RandomRotation(15),
            # transforms.RandomCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))

    print_dataset = datasets.ImageFolder(
        args.print_folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.Pad(16),
            transforms.RandomCrop(256),
            transforms.RandomRotation(15),
            # transforms.RandomCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))

    train_loader = torch.utils.data.DataLoader(
        ContrastiveDataset(print_dataset, photo_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    return train_loader
