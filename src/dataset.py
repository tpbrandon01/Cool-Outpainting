import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc.pilutil import imread, imsave
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_extrapolation_mask
np.set_printoptions(threshold=10000000)
import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.loaded_img_size = config.LOADED_IMAGE_SIZE
        
        self.mask = config.MASK
        
        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 7

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        print('loading name...', name)
        return os.path.basename(name)

    def load_item(self, index):

        size = self.loaded_img_size # Assume the input imgs' height and width are the same.

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)
        
        # load mask
        mask, mask_information = self.load_mask(size)
        mask = np.expand_dims(mask, axis=0)
        # mask_information = np.expand_dims(mask_information, axis=0)
        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...].copy()
            mask = mask[:, ::-1, ...].copy()
            mask_information = mask_information.copy() #[1,4]
            temp = mask_information[2] 
            mask_information[2] = size - mask_information[3]
            mask_information[3] = size - temp
            # mask#####
        
        # print('Get dataset item...', torch.from_numpy(unlbl_binary_mask).float().shape, torch.from_numpy(unlbl_binary_mask).float())
        # return self.to_tensor(img), self.to_tensor(img_gray), (torch.from_numpy(semantic)).float(), (torch.from_numpy(unlbl_binary_mask)).float(), (torch.from_numpy(smt_one_hot)).float(), self.to_tensor(edge), (torch.from_numpy(mask)).float()
        return self.to_tensor(img), (torch.from_numpy(mask)).float(), torch.from_numpy(mask_information).int() 
        # [3, 256, 256], [1, 256, 256], [1, 256, 256], [20, 256, 256],  [1, 256, 256]
    

    def onehot_enc(self, smt):#smt.shape = [256, 256] #but if any pixel is unlbled, I set 19-dim vector to 0 vector. 
        _, w, h =  smt.shape
        # print('w,h =',w,h)
        smt_flat = smt.flatten()
        # print(smt_flat)
        voidpart = (smt_flat==self.ignore_index)
        nonvoid_pix = []
        for i in range(len(voidpart)):
            if voidpart[i] == 0:
                nonvoid_pix.append(i)

        # print('smt_flat.shape =',smt_flat.shape)
        onehot = np.zeros((w*h, self.NUM_CLASSES))
        onehot[nonvoid_pix, smt_flat[nonvoid_pix]] = 1
        # print('onehot.shape =',onehot.shape)
        onehot = onehot.reshape((w, h, self.NUM_CLASSES))
        onehot = np.transpose(onehot, (2,0,1)) 
        return onehot 

    def load_mask(self, size):######################################################### 做完去確定image dataset沒問題 就可以下去train了
        # print('load_mask, img.shape =', img.shape)#(256, 256, 3)
        mask_type = self.mask
        if mask_type == 1:
            top = np.random.randint(16, 113)#128-16
            bot = 128+top
            left = np.random.randint(16, 113)#128-16
            right = 128+left
            mask, mask_inf = create_extrapolation_mask(size, size, crop_pos=(top, bot, left, right))
            
        # extrapolation mask (middle white, periphery black) 
        if mask_type == 2:
            mask, mask_inf = create_extrapolation_mask(size, size)
            
        return mask, mask_inf
        
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True, mode=None):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j+side//30:j + 29*side//30, i+side//30:i + 29*side//30, ...]

        img = scipy.misc.imresize(img, [height, width],interp='nearest', mode=mode)

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
