import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from .samplers import *

class CIFAR10DataLoader(DataLoader):
    """
    Load CIFAR 10
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if training:
            self.dataset = datasets.CIFAR10(data_dir, train=training, download=True, transform=train_trsfm)
            self.val_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm) # test set
        else:
            self.dataset = datasets.CIFAR10(data_dir, train=training, download=True, transform=test_trsfm)

        num_classes = len(np.unique(self.dataset.targets))
        assert num_classes == 10

        cls_num_list = [0] * num_classes
        for label in self.dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        self.shuffle = shuffle
        self.n_samples = len(self.dataset)
        self.balanced = True

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            "pin_memory": False
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class CIFAR100DataLoader(DataLoader):
    """
    Load CIFAR 100
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if training:
            self.dataset = datasets.CIFAR100(data_dir, train=training, download=True, transform=train_trsfm)
            self.val_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm) # test set
        else:
            self.dataset = datasets.CIFAR100(data_dir, train=training, download=True, transform=test_trsfm)

        num_classes = len(np.unique(self.dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in self.dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        self.shuffle = shuffle
        self.n_samples = len(self.dataset)
        self.balanced = True

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            "pin_memory": False
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class BalancedSampler(Sampler):
    # This balanced sampler is implemented by https://github.com/frank-xwang/RIDE-LongTailRecognition
    # Which is some different with https://github.com/facebookresearch/classifier-balancing
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class ImbalanceCIFAR100DataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_type='exp', imb_factor=0.01):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm) # test set

        if training:
            dataset = IMBALANCECIFAR100(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=imb_factor)
            val_dataset = test_dataset
        else:
            dataset = test_dataset
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.balanced = balanced

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                # buckets = [[] for _ in range(num_classes)]
                # for idx, label in enumerate(dataset.targets):
                #     buckets[label].append(idx)
                # sampler = BalancedSampler(buckets, retain_epoch_size)
                sampler = ClassAwareSampler(dataset, num_samples_cls=1)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            "pin_memory": False
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class ImbalanceCIFAR10DataLoader(DataLoader):
    """
    Imbalance Cifar10 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_factor=0.01):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        if training:
            dataset = IMBALANCECIFAR10(data_dir, imb_factor=imb_factor, train=True, download=True, transform=train_trsfm)
            val_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm) # test set
        else:
            dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm) # test set
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.balanced = balanced

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 10

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                # buckets = [[] for _ in range(num_classes)]
                # for idx, label in enumerate(dataset.targets):
                #     buckets[label].append(idx)
                # sampler = BalancedSampler(buckets, retain_epoch_size)
                sampler = ClassAwareSampler(dataset, num_samples_cls=1)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            "pin_memory": False
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)
