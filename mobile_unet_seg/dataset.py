"""
Dataset class to parse the Supervisely dataset.
Assuming the following dataset folder structure:
    - train
        - images
        - segs
    - val
        - images
        - segs
    - test
        - images
        - segs
The names for the folders are defined through the config.
"""
import os
import itertools
from glob import glob
import cv2
import numpy as np
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)


class Dataset:

    def __init__(self,
                 base_path,
                 im_path,
                 seg_path,
                 batch_size,
                 in_size,
                 out_size,
                 do_aug=False):
        # read all paths
        self.ims = sorted(glob(os.path.join(base_path, im_path, '*')))
        self.segs = sorted(glob(os.path.join(base_path, seg_path, '*')))
        self.do_aug = do_aug
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size

        # validation
        assert len(self.ims) != 0, 'Empty images!'
        assert len(self.segs) != 0, 'Empty segs!'
        for i, s in zip(self.ims, self.segs):
            sname = s.split('/')[-1].split('.')[0]
            iname = i.split('/')[-1].split('.')[0]
            assert iname == sname, f'incoherent names: {i} {s}'

    def __len__(self):
        return len(self.ims)

    def __call__(self):
        zipped = itertools.cycle(zip(self.ims, self.segs))
        while True:
            X = []
            Y = []
            for _ in range(self.batch_size):
                im, seg = next(zipped)
                im, seg = self.read_im(im), self.read_seg(seg)
                if self.do_aug:
                    im, seg = self.augment(im, seg)
                X.append(self.to_tensor_im(im))
                Y.append(self.to_tensor_seg(seg))
            yield np.array(X), np.array(Y)

    def read_im(self, im):
        im = cv2.imread(im)
        im = cv2.resize(im, (self.in_size, self.in_size))
        return im

    def read_seg(self, seg):
        seg = cv2.imread(seg)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        seg = (seg != 0).astype(np.uint8)
        seg = cv2.resize(seg, (self.out_size, self.out_size))
        return seg
    
    def to_tensor_im(self, im):
        im = im.astype(np.float32)
        im = im / 255.0
        return im

    def to_tensor_seg(self, seg):
        seg = seg.reshape(self.out_size * self.out_size, 1)
        seg = seg.astype(np.float32)
        return seg

    def augment(self, image, mask):
        aug = Compose([
                OneOf([
                    RandomSizedCrop(min_max_height=(50, 101),
                                    height=self.out_size,
                                    width=self.out_size, p=0.5),
                    PadIfNeeded(min_height=self.out_size,
                                min_width=self.out_size, p=0.5)
                ], p=1),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                OneOf([
                    ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05,
                                     alpha_affine=120 * 0.03),
                    GridDistortion(p=0.5),
                    OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
                ], p=0.8),
                CLAHE(p=0.8),
                RandomBrightnessContrast(p=0.8),
                RandomGamma(p=0.8)])
        augmented = aug(image=image, mask=mask)
        image_heavy = augmented['image']
        mask_heavy = augmented['mask']
        return image_heavy, mask_heavy


def create_loaders(cfg):
    train_dataset = Dataset(base_path=cfg.TRAIN_PATH,
                            im_path=cfg.IM_PATH,
                            seg_path=cfg.SEG_PATH,
                            batch_size=cfg.BATCH_SIZE,
                            in_size=cfg.IN_SIZE,
                            out_size=cfg.OUT_SIZE,
                            do_aug=cfg.AUG)
    val_dataset = Dataset(base_path=cfg.VAL_PATH,
                          im_path=cfg.IM_PATH,
                          seg_path=cfg.SEG_PATH,
                          batch_size=cfg.BATCH_SIZE,
                          in_size=cfg.IN_SIZE,
                          out_size=cfg.OUT_SIZE,
                          do_aug=False)
    return train_dataset, val_dataset
