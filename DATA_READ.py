import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from os.path import join
import os
import numpy as np
import random
import math
import scipy.io as scio
import csv
import cv2

'''
指定加载图像颜色空间为RGB
'''
def load_img(filepath, C='RGB'):
    with open(filepath, 'rb') as f:
        img = Image.open(f)
        return img.convert(C)


def random_crop_two(img, org, patchsize=256, gap=8):

    # H, W, C = img.shape
    H, W = img.size
    if H-patchsize-1 > 0:  # 如果图像的高度大于裁剪的尺寸，则随机裁剪
        start_x = random.randrange(0, H-patchsize-1, gap)   # 随机生成裁剪的起始点，gap=8即为步长为8
    else:
        start_x = 0
    end_x = start_x + patchsize
    if W-patchsize-1 > 0: # 如果图像的宽度大于裁剪的尺寸，则随机裁剪
        start_y = random.randrange(0, W-patchsize-1, gap)
    else:
        start_y = 0
    end_y = start_y + patchsize
    crop_box = [start_x, start_y, end_x, end_y]
    region_img = img.crop(crop_box)
    region_org = org.crop(crop_box)
    # region_img = img[start_x:end_x, start_y:end_y, :].copy()  # CV2
    # region_org = org[start_x:end_x, start_y:end_y, :].copy()

    return region_img, region_org


def random_crop(img, org, sal, patchsize=256, gap=8):
    H, W = img.size
    if H-patchsize-1 > 0:
        start_x = random.randrange(0, H-patchsize-1, gap)
    else:
        start_x = 0
    end_x = start_x + patchsize
    if W-patchsize-1 > 0:
        start_y = random.randrange(0, W-patchsize-1, gap)
    else:
        start_y = 0
    end_y = start_y + patchsize
    crop_box = [start_x, start_y, end_x, end_y]
    region_img = img.crop(crop_box)
    region_org = org.crop(crop_box)
    # region_sal = sal.crop(crop_box)

    if sal != []:
        region_sal = sal[start_x:end_x, start_y:end_y]  # 把裁剪后的图像放入region_sal中
    else:
        region_sal = []


    return region_img, region_org, region_sal



class augmentation_two(object):

    def __call__(self, img, org, tb_flg=True):
        if random.random() < 0.5:
            # img = img[:, ::-1, :]  # CV2
            # org = org[:, ::-1, :]
            img = img.transpose(Image.FLIP_LEFT_RIGHT) # 对图像进行左右翻转
            org = org.transpose(Image.FLIP_LEFT_RIGHT)
        if tb_flg:
            if random.random() < 0.5:
                # img = img[::-1, :, :]
                # org = org[::-1, :, :]
                img = img.transpose(Image.FLIP_TOP_BOTTOM) # 对图像进行上下翻转
                org = org.transpose(Image.FLIP_TOP_BOTTOM)

        return img, org






def image_file_name(txtfile, image_dir, org_dir):
    image_filenames = []
    org_filenames = []
    image_scores = []
    with open(txtfile, 'r') as f:
        for line in f:
            image_scores.append(float(line.split()[0]))
            image_filenames.append(image_dir+'//'+line.split()[1])
            org_filenames.append(org_dir+'//'+line.split()[1][:-9]+'.bmp')

    return image_filenames, org_filenames, image_scores


class ReadIQAFolder(data.Dataset):
    def __init__(self, matpath, nexp=0, aug=False, random_crop=False, img_transform=None, status='eval'):
        super(ReadIQAFolder).__init__()

        self.D = scio.loadmat(matpath[0])
        data_dir = matpath[1]
        exp = self.D['index'][nexp][:]
        # exp = []
        # for i in range(len(self.D['index'][nexp])):
        #     exp.append(i+1)
        # exp.reverse()
        if status == 'eval':
            colID = exp[int(len(exp)*0.8):]
            print('TEST INDEX:', end=' ')
            print(colID)
        elif status == 'valid':
            colID = exp[int(len(exp)*0.7):int(len(exp)*0.8)]
            print('VALID INDEX:', end=' ')
            print(colID)
        elif status == 'full':
            colID = exp[:]
        else:
            colID = exp[0:int(len(exp)*0.7)]
            # print('TRAIN INDEX:', end=' ')
            # print(colID)


        # self.D['ref_ids'] = []
        # for i in range(len(self.D['index'][nexp])):
        #     self.D['ref_ids'].append(i+1)

        self.im_path = []
        self.ref_path = []
        self.scores = []
        self.scoresV = []
        self.dtype = []
        self.norm = matpath[2]

        for idx, im_name in enumerate(self.D['im_names']):
            if self.D['ref_ids'][idx].item() in colID:
                if status == 'train':
                    rep = 25
                else:
                    rep = 25
                for _ in range(rep):
                    self.im_path.append(join(data_dir, self.D['im_names'][idx][0].item()))
                    self.ref_path.append(join(data_dir, self.D['ref_names'][idx][0].item()))
                    self.scores.append(self.D['subjective_scores'][idx].item() / self.norm)
                    self.scoresV.append(self.D['subjective_scores'][idx].item() / self.norm)
                    # self.dtype.append(self.D['dtype_ids'][idx].item())
                    self.dtype.append(0)
        self.transforms = img_transform
        self.aug = aug
        self.rcrop = random_crop

    def __getitem__(self, item):

        img = load_img(self.im_path[item].replace('\\', '/'))
        org = load_img(self.ref_path[item].replace('\\', '/'))

        if self.aug:
            img, org = augmentation_two()(img, org)
        if self.rcrop:
            img, org = random_crop_two(img, org, patchsize=224)

        mos = self.scores[item]
        mos = np.array(mos).astype(np.float32)
        # mosV = self.scoresV[item]
        # mosV = np.array(mosV).astype(np.float32)
        if self.transforms:
            img = self.transforms(img)
            org = self.transforms(org)

        ids = self.dtype[item]

        return img, org, mos, ids

    def __len__(self):
        return len(self.im_path)




