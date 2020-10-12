from torch.utils.data import Dataset
import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
from PIL import Image
import torch

class BaseDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=""):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix

        if isinstance(self.scale, list):
            assert len(self.scale) == 2, 'Scale list must have 2 elements'
            assert 0 < self.scale[0] and 0 < self.scale[1], 'Scale elements must larger than zero'
        else:
            assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, scale):
        w, h = img.size
        if isinstance(scale, list):
            newW, newH = scale[0], scale[1]
        else:
            newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img = img.resize((newW, newH))

        np_img = np.array(img)
        # expand black image
        if(len(np_img.shape) == 2):
            np_img = np.expand_dims(np_img, axis=2)
        # HWC to CHW
        img_trans = np_img.transpose((2, 0 , 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return torch.from_numpy(img_trans).type(torch.FloatTensor)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(
            os.path.join(self.masks_dir, idx + self.mask_suffix + '.*')
        )
        img_file = glob(
            os.path.join(self.imgs_dir, idx + '.*')
        )

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are img:{img.size} and mask:{mask.size}'
        
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return img, mask