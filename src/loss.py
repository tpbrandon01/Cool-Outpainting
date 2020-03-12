import torch
import torch.nn as nn
import torchvision.models as models
from src.load_incepres import inceptionresnetv2
import numpy as np
import scipy.stats as st

def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    # out_filter = np.repeat(out_filter, [1, 1, inchannels, 1])
    return out_filter

def torch_make_guass_var(size, sigma, inchannels=1, outchannels=1):
    kernel = gauss_kernel(size, sigma, inchannels, outchannels)
    var = torch.tensor(kernel).cuda()
    return var

def relative_spatial_variant_mask(mask, hsize=21, sigma=1.0/40, iters=9, cuda=True):
    eps = 1e-8
    kernel = torch_make_guass_var(hsize, sigma)
    # print('kernel.shape:',kernel.shape, kernel.type())
    init = mask
    mask_priority = None
    mask_priority_pre = None
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=hsize, bias=False, padding = hsize//2)
    with torch.no_grad():
        conv.weight = nn.Parameter(kernel)
    if cuda:
        conv = conv.cuda()
    for i in range(iters):
        with torch.no_grad():
            mask_priority = conv(init)
        mask_priority = mask_priority * (1-mask)
        # print('iter:', i, mask_priority)
        if i == iters-2:
            mask_priority_pre = mask_priority
        init = mask_priority + (mask)
    mask_priority = mask_priority_pre / (mask_priority+eps) # unmasked part are all zero
    # print(mask_priority)
    return mask_priority



class RSV_Loss(nn.Module):
    def __init__(self, hsize=21, sigma=1.0/40, iters=9, cuda=True):
        super(RSV_Loss, self).__init__()
        # self.cuda = cuda
        self.hsize = hsize
        self.sigma = sigma
        self.iters = iters
        self.cuda = cuda
    def __call__(self, x, y, mask):
        rsv_mask = relative_spatial_variant_mask(mask, hsize=self.hsize, sigma=self.sigma, iters=self.iters, cuda=self.cuda)
        diff = torch.abs(x-y)
        summ = torch.sum(diff * rsv_mask)
        # print('summ:',summ)
        _, c, h, w = x.shape
        # print("x.shape:",x.shape,c,h,w)
        loss = summ / (c*h*w)
        # print('loss:',loss)
                
        return loss

class SegmentationLosses(nn.Module):
    def __init__(self, weight=None, reduction='mean', batch_average=True, ignore_index=255, cuda=True, mode = 'ce'):
        super(SegmentationLosses, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.batch_average = batch_average
        self.cuda = cuda
        self.mode = mode

    def __call__(self, logit, target, **kwargs):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss
        

    # def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
    #     n, c, h, w = logit.size()
    #     criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
    #                                     size_average=self.size_average)
    #     if self.cuda:
    #         criterion = criterion.cuda()

    #     logpt = -criterion(logit, target.long())
    #     pt = torch.exp(logpt)
    #     if alpha is not None:
    #         logpt *= alpha
    #     loss = -((1 - pt) ** gamma) * logpt

    #     if self.batch_average:
    #         loss /= n

    #     return loss
    


class EdgePerceptualLoss(nn.Module):
    r"""
    Edge Perceptual loss, Inception_Resnet_v2-based
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(EdgePerceptualLoss, self).__init__()
        self.add_module('incepres', inceptionresnetv2())
        self.incepres.load_state_dict(torch.load('edge_incepres_model/ep_8_14556/incepres_ep_8'))
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        with torch.no_grad():
            x_hid_list, y_hid_list = self.incepres(x), self.incepres(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_hid_list[0], y_hid_list[0])
        content_loss += self.weights[1] * self.criterion(x_hid_list[1], y_hid_list[1])
        content_loss += self.weights[2] * self.criterion(x_hid_list[2], y_hid_list[2])
        content_loss += self.weights[3] * self.criterion(x_hid_list[3], y_hid_list[3])
        content_loss += self.weights[4] * self.criterion(x_hid_list[4], y_hid_list[4])


        return content_loss





class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss(reduction='sum')

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None, mask=None):
        if self.type == 'hinge':
            # not handle
            if is_disc:
                if is_real:
                    outputs = -outputs
                if mask is not None:
                    mask_loss = self.criterion(1 + outputs) * mask
                    return mask_loss.sum() / mask.sum()
                #return self.criterion(1 + outputs).mean()
            else:
                if mask is not None:
                    mask_loss = (-outputs) * mask
                    return mask_loss.sum() / mask.sum()
                # return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            if mask is not None:
                # print(type(mask), mask.shape)
                # print(mask)
                masked_outputs = outputs * mask
                masked_labels = labels * mask
                loss = self.criterion(masked_outputs, masked_labels) / mask.sum()
            else:
                loss = self.criterion(outputs, labels) / outputs.shape[0]
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
