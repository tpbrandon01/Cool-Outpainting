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
from .utils import create_mask, create_extrapolation_mask
np.set_printoptions(threshold=10000000)
import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, semantic_flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        print('initialize edge_data =', self.edge_data)
        self.mask_data = self.load_flist(mask_flist)
        self.semantic_data = self.load_flist(semantic_flist)
        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        # for semantic label map
        self.NUM_CLASSES = config.SEMANTIC_CLASS_NUM
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(0,self.NUM_CLASSES)))
        
        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

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

        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load semantic label map
        _tmp = imread(self.semantic_data[index])
        # _tmp = np.array(Image.open(self.semantic_data[index]), dtype=np.uint8)
        semantic, unlbl_binary_mask = self.encode_segmap(_tmp)
        
        # print('sexy',semantic)
        # time.sleep(50)
        semantic = self.resize(semantic, size, size)
        unlbl_binary_mask = self.resize(unlbl_binary_mask, size, size, mode='L')
        # print("unlbl_mask still numpy:",unlbl_binary_mask.shape,unlbl_binary_mask)
        # time.sleep(50)
        semantic = np.expand_dims(semantic, axis=0)
        unlbl_binary_mask = np.expand_dims(unlbl_binary_mask, axis=0)
        # turn semantic into one-hot (unlbl is index [-1])
        
        
        smt_one_hot = self.onehot_enc(semantic)
        
        # load mask
        mask = self.load_mask(img, index)
        mask = np.expand_dims(mask, axis=0)
        # load edge #not be masked yet
        edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            semantic = semantic[:, ::-1, ...].copy()
            unlbl_binary_mask = unlbl_binary_mask[:, ::-1, ...].copy()
            smt_one_hot = smt_one_hot[:, ::-1, ...].copy()
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...].copy()
            # mask#####
        
        # print('Get dataset item...', torch.from_numpy(unlbl_binary_mask).float().shape, torch.from_numpy(unlbl_binary_mask).float())
        # return self.to_tensor(img), self.to_tensor(img_gray), (torch.from_numpy(semantic)).float(), (torch.from_numpy(unlbl_binary_mask)).float(), (torch.from_numpy(smt_one_hot)).float(), self.to_tensor(edge), (torch.from_numpy(mask)).float()
        return self.to_tensor(img), (torch.from_numpy(semantic)).float(), torch.from_numpy(unlbl_binary_mask).float()/255, (torch.from_numpy(smt_one_hot)).float(), (torch.from_numpy(mask)).float()
        # [3, 256, 256], [1, 256, 256], [1, 256, 256], [20, 256, 256],  [1, 256, 256]
    def encode_segmap(self, mask):
        unlbl_binary_mask = np.zeros(mask.shape)
        # Put all void classes to zero
        for _voidc in self.void_classes:
            unlbl_binary_mask[mask == _voidc] = 1
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask, unlbl_binary_mask

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

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_mask(self, img, index):
        # print('load_mask, img.shape =', img.shape)#(256, 256, 3)
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # # external + random block
        # if mask_type == 4:
        #     mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # # external + random block + half
        # elif mask_type == 5:
        #     mask_type = np.random.randint(1, 4)

        # extrapolation mask (middle white, periphery black) 
        if mask_type == 7:
            mask = create_extrapolation_mask(imgw, imgh)
            return mask
        
        # random block
        elif mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        # elif mask_type == 2:
        #     # randomly choose right or left
        #     return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        elif mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = rgb2gray(mask)######fuck
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8)   # threshold due to interpolation
            return mask

        # test mode: load mask non random
        elif mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) #* 255
            return mask
        
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
