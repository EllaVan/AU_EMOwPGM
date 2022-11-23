import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import os
import numpy as np
import random
import pickle as pkl

from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data

color_jitter = transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0)
SSL_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(),  # with 0.5 probability
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),])
basic_train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def make_dataset(image_list, labelEMO_list, labelAU_list, au_relation=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(), labelEMO_list[i, :], labelAU_list[i, :],au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(), labelEMO_list[i, :], labelAU_list[i, :]) for i in range(len_)]
    return images

class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        weights = dataset.train_weight_EMO
        self.weights = torch.DoubleTensor(list(weights))

    def _get_labels(self, dataset):
        return [dataset.label[i] for i in self.indices]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class BP4D(Dataset):
    def __init__(self, pkl_path, phase='train', fold = 1, transform=None, crop_size = 224):
        # assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        self.crop_size = crop_size
        self.transform = transform
        self.basic_aug = [flip_image, add_g]
        self.pkl_path = pkl_path
        self.phase = phase
        if fold != 0:
            self.test_fold = 'test_fold' + str(fold)
            self.train_fold = 'train_fold' + str(fold)
        else:
            self.test_fold = 'test'
            self.train_fold = 'train'

        with open(self.pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
            
        if self.phase == 'test':
            self.img_path = pkl_file[self.test_fold]['img_path']
            self.labelsEMO = pkl_file[self.test_fold]['labelsEMO']
            self.labelsAU = pkl_file[self.test_fold]['labelsAU']
        elif self.phase == 'train':
            self.img_path = pkl_file[self.train_fold]['img_path']
            self.labelsEMO = pkl_file[self.train_fold]['labelsEMO']
            self.labelsAU = pkl_file[self.train_fold]['labelsAU']
            self.train_weight_AU = pkl_file[self.train_fold]['AU_weight']
            self.train_weight_EMO = pkl_file[self.train_fold]['EMO_weight']
        self.labelsEMO = np.array(self.labelsEMO).reshape(-1, 1)
        self.dataset = make_dataset(self.img_path, self.labelsEMO, self.labelsAU)
        self.priori = pkl_file['priori']
        self.EMO = pkl_file['EMO']
        self.AU = pkl_file['AU']

    def __getitem__(self, index):
        img_path, labelsEMO, labelsAU = self.dataset[index]

        if labelsEMO is not None and labelsAU is not None and os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = img[:, :, ::-1]

            if self.phase == 'train':
                if self.basic_aug and random.uniform(0, 1) > 0.5:
                    img = self.basic_aug[1](img)

            if self.transform is not None:
                img1 = self.transform(img)
                img_SSL = SSL_transforms(img)

            img2 = transforms.RandomHorizontalFlip(p=1)(img1)

            return img1, img_SSL, img2, labelsEMO, labelsAU, index
        else:
            return None, None, None, None, None, None

    def __len__(self):
        return len(self.img_path)


class DISFA(Dataset):
    def __init__(self, pkl_path, phase='train', fold = 1, transform=None, crop_size = 224):
        # assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        self.crop_size = crop_size
        self.transform = transform
        self.basic_aug = [flip_image, add_g]
        self.pkl_path = pkl_path
        self.phase = phase
        if fold != 0:
            self.test_fold = 'test_fold' + str(fold)
            self.train_fold = 'train_fold' + str(fold)
        else:
            self.test_fold = 'test'
            self.train_fold = 'train'

        with open(self.pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
            
        if self.phase == 'test':
            self.img_path = pkl_file[self.test_fold]['img_path']
            self.labelsEMO = pkl_file[self.test_fold]['labelsEMO']
            self.labelsAU = pkl_file[self.test_fold]['labelsAU']
        elif self.phase == 'train':
            self.img_path = pkl_file[self.train_fold]['img_path']
            self.labelsEMO = pkl_file[self.train_fold]['labelsEMO']
            self.labelsAU = pkl_file[self.train_fold]['labelsAU']
            self.train_weight_AU = pkl_file[self.train_fold]['AU_weight']

        self.labelsEMO = np.array(self.labelsEMO).reshape(-1, 1)
        self.dataset = make_dataset(self.img_path, self.labelsEMO, self.labelsAU)
        self.priori = pkl_file['priori']
        self.EMO = pkl_file['EMO']
        self.AU = pkl_file['AU']

    def __getitem__(self, index):
        img_path, labelsEMO, labelsAU = self.dataset[index]

        if labelsEMO is not None and labelsAU is not None and os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = img[:, :, ::-1]

            if self.phase == 'train':
                if self.basic_aug and random.uniform(0, 1) > 0.5:
                    img = self.basic_aug[1](img)

            if self.transform is not None:
                img1 = self.transform(img)
                img_SSL = SSL_transforms(img)

            img2 = transforms.RandomHorizontalFlip(p=1)(img1)

            return img1, img_SSL, img2, labelsEMO, labelsAU, index
        else:
            return None, None, None, None, None, None

    def __len__(self):
        return len(self.img_path)  

class RAF(Dataset):
    def __init__(self, pkl_path, phase='train', fold = 1, transform=None, crop_size = 224):
        # assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        self.crop_size = crop_size
        self.transform = transform
        self.basic_aug = [flip_image, add_g]
        self.pkl_path = pkl_path
        self.phase = phase
        if fold != 0:
            self.test_fold = 'test_fold' + str(fold)
            self.train_fold = 'train_fold' + str(fold)
        else:
            self.test_fold = 'test'
            self.train_fold = 'train'

        with open(self.pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
            
        if self.phase == 'test':
            self.img_path = pkl_file[self.test_fold]['img_path']
            self.labelsEMO = pkl_file[self.test_fold]['labelsEMO']
            self.labelsAU = pkl_file[self.test_fold]['labelsAU']
        elif self.phase == 'train':
            self.img_path = pkl_file[self.train_fold]['img_path']
            self.labelsEMO = pkl_file[self.train_fold]['labelsEMO']
            self.labelsAU = pkl_file[self.train_fold]['labelsAU']
            self.train_weight_AU = pkl_file[self.train_fold]['AU_weight']
            self.train_weight_EMO = pkl_file[self.train_fold]['EMO_weight']
            
        self.labelsEMO = np.array(self.labelsEMO).reshape(-1, 1)
        self.dataset = make_dataset(self.img_path, self.labelsEMO, self.labelsAU)
        self.priori = pkl_file['priori']
        self.EMO = pkl_file['EMO']
        self.AU = pkl_file['AU']

    def __getitem__(self, index):
        img_path, labelsEMO, labelsAU = self.dataset[index]

        if labelsEMO is not None and labelsAU is not None and os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = img[:, :, ::-1]

            if self.phase == 'train':
                if self.basic_aug and random.uniform(0, 1) > 0.5:
                    img = self.basic_aug[1](img)

            if self.transform is not None:
                img1 = self.transform(img)
                img_SSL = SSL_transforms(img)

            img2 = transforms.RandomHorizontalFlip(p=1)(img1)

            return img1, img_SSL, img2, labelsEMO, labelsAU, index
        else:
            return None, None, None, None, None, None

    def __len__(self):
        return len(self.img_path)