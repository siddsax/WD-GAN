import torch
from torch.utils import data
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
import random
from PIL import ImageFile

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize_1, opt.loadSize_2]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop((opt.fineSize_1, opt.fineSize_2 )))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)

    # if opt.isTrain and not opt.no_flip:
    #     print("="*1000)
    #     # exit()
        # transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
                    #    transforms.Normalize((0.5, 0.5, 0.5),
                                            # (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, opt):
        'Initialization'
        self.transform = get_transform(opt)
        self.dataroot = opt.dataroot
        self.AB_paths = os.listdir(opt.dataroot)
        self.train = opt.train
        self.opt = opt

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.AB_paths)

    def __getitem__(self, index):

        AB_path = self.dataroot + '/' + self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        if self.train:
            w, h = AB.size
            w2 = int(w / 2)
            B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize_1, self.opt.loadSize_2), Image.BICUBIC)
        else:
            B = AB

        seed = random.randint(0,2**32)
        random.seed(seed)

        # B = transforms.ToTensor()(B)
        B = self.transform(B)

        w_offset = random.randint(0, max(0, self.opt.loadSize_1 - self.opt.fineSize_1 - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize_2 - self.opt.fineSize_2 - 1))

        B = B[:, h_offset:h_offset + self.opt.fineSize_2, w_offset:w_offset + self.opt.fineSize_1]

        return B, 0

