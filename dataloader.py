import os
import sys
import logging
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop, ToTensor, Compose, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip
import torchvision
import torch


# I wrote image loader for two different scenarios for training
# 1) Custom dataset(I used clic dataset to train)(custom_data=1)
# 2) torchvision datasets (custom_data = 0)
class ImageDataLoader():
    def __init__(self, batch_size, custom_data= 0):
        if (custom_data == 1):
            # for custom data training dataset paths should be defined here
            train_datas = ''
            valid_data = ''

            self.train_dataset = ImageDataset(train_datas,
                                              size=32,
                                              train=True)
            self.test_dataset = ImageDataset_test(valid_data,
                                                size=32,
                                                  train=False)



            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=False)

            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=1,  # 5,
                                          shuffle=False,# num_workers,
                                          pin_memory=False,  # True,
                                          drop_last=False)


        else:
            self.transforms_train = Compose([RandomCrop(size=32), RandomHorizontalFlip(), RandomVerticalFlip(),
                                             ToTensor()])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=self.transforms_train)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)
            self.transforms_test = Compose([ToTensor()])
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=self.transforms_test)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=False, num_workers=2)

# custom dataset, I wrote this in my thesis modified slightly for this task
class ImageDataset(Dataset):
    def __init__(self, root, size=32, train=True):
        self.size = size
        try:
            if isinstance(root, str):
                self.image_files = [os.path.join(root, f) for f in os.listdir(root) if
                                    (f.endswith('.png') or f.endswith('.jpg'))]
            else:
                self.image_files = []
                for i in range(0, len(root)):
                    self.image_files_temp = [os.path.join(root[i], f) for f in os.listdir(root[i]) if
                                             (f.endswith('.png') or f.endswith('.jpg'))]
                    self.image_files = self.image_files + self.image_files_temp
        except:
            logging.getLogger().exception('Dataset could not be found.', exc_info=False)
            sys.exit(1)
        if size == 0:
            # for testing do not apply any preprocessing rather than converting to torch tensor
            self.transforms = Compose([ToTensor()])
        else:
            crop = RandomCrop(size) if train else CenterCrop(size)
            self.transforms = Compose([crop, RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        img = pil_loader(self.image_files[i])
        # check size of image and resize it if width or height less than requested size
        width, height = img.size
        ws, hs = 0, 0
        if width < self.size and self.size > 0:
            ws = 1
        if height < self.size and self.size > 0:
            hs = 1
        if ws == 1 and hs == 1:
            img = ImageOps.fit(img, self.size)
        elif ws == 1 and hs == 0:
            img = ImageOps.fit(img, (self.size, height))
        elif ws == 0 and hs == 1:
            img = ImageOps.fit(img, (width, self.size))
        return self.transforms(img)


class ImageDataset_test(Dataset):
    def __init__(self, root, size, train=False):
        self.size = size
        try:
            if isinstance(root, str):
                self.image_files = [os.path.join(root, f) for f in os.listdir(root) if
                                    (f.endswith('.png') or f.endswith('.jpg'))]
            else:
                self.image_files = []
                for i in range(0, len(root)):
                    self.image_files_temp = [os.path.join(root[i], f) for f in os.listdir(root[i]) if
                                             (f.endswith('.png') or f.endswith('.jpg'))]
                    self.image_files = self.image_files + self.image_files_temp
        except:
            logging.getLogger().exception('Dataset could not be found. Drive might be unmounted.', exc_info=False)
            sys.exit(1)
        if size == 0:
            self.transforms = Compose([ToTensor()])
        else:
            crop = CenterCrop(size)
            self.transforms = Compose([crop, ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        # NOTE check if the output is normalized between 0-1
        img = pil_loader(self.image_files[i])
        # check size of image and resize it if width or height less than requested size
        width, height = img.size
        ws, hs = 0, 0
        if width < self.size and self.size > 0:
            ws = 1
        if height < self.size and self.size > 0:
            hs = 1
        if ws == 1 and hs == 1:
            img = ImageOps.fit(img, self.size)
        elif ws == 1 and hs == 0:
            img = ImageOps.fit(img, (self.size, height))
        elif ws == 0 and hs == 1:
            img = ImageOps.fit(img, (width, self.size))
        return self.transforms(img)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
