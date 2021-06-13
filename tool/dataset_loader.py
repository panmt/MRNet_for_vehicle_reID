from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch.nn as nn
from PIL import Image
import os.path as osp
import torch
from torch.utils.data import Dataset
from . import transforms_image
from skimage import io
import numpy as np
import os

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
        self.Rescale = transforms_image.Rescale()
        self.Normalize = transforms_image.Normalize()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img_kp = io.imread(img_path)
        # H, W = img_kp.shape[0], img_kp.shape[1]
        img_kp = img_kp.astype(np.float)
        img_kp = self.Normalize(img_kp)
        image_224, image_56 = self.Rescale(img_kp)
        image_224 = torch.from_numpy(image_224.transpose(2, 0, 1)).float()
        image_56 = torch.from_numpy(image_56.transpose(2, 0, 1)).float()
        return image_224, image_56, pid, camid, img_path



