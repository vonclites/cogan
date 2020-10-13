import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ContrastiveDataset(Dataset):
    def __init__(self, print_dataset, photo_dataset, positive_prob=0.5):
        super().__init__()
        self.print = print_dataset
        self.photo = photo_dataset
        self.positive_prob = positive_prob

        self.h = {}

        print(len(self.print))
        print(len(self.photo))

        for i in range(len(self.photo)):
            photo_address = self.photo.imgs[i][0]
            subject_id = photo_address.split('/')[-2]
            for j in range(len(self.print.imgs)):
                print_address = self.print.imgs[j][0]
                if subject_id in print_address:
                    if i in self.h:
                        self.h[i].append(j)
                    else:
                        self.h[i] = [j]

    def __getitem__(self, index):
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob
        img_0, label_0 = self.photo[index]

        print_samples = self.h[index]
        if same_class:
            rnd_idx = random.randint(0, len(print_samples) - 1)
            index_1 = print_samples[rnd_idx]
            img_1, label_1 = self.print[index_1]
        else:
            while True:
                index_1 = random.randint(0, self.__len__() - 1)
                if index_1 not in self.h[index]:
                    img_1, label_1 = self.print[index_1]
                    break
        # print(same_class, '<<')
        # plot_tensor([img_0, img_1])
        return img_0, img_1, same_class

    def __len__(self):
        return min(len(self.print), len(self.photo))


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
