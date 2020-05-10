import torch
import torch.nn as nn
from math import log2
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

# def FEN(self, x, cnum):
#         conv3, conv5, deconv = self.conv3, self.conv5, self._deconv
#         x = conv5(inputs=x, filters=cnum, strides=1, name='conv1')
#         x = conv3(inputs=x, filters=cnum * 2, strides=2, name='conv2_downsample')
#         x = conv3(inputs=x, filters=cnum * 2, strides=1, name='conv3')
#         x = conv3(inputs=x, filters=cnum * 4, strides=2, name='conv4_downsample')
#         x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv5')
#         x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv6')

#         x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=2, name='conv7_atrous')
#         x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=4, name='conv8_atrous')
#         x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=8, name='conv9_atrous')
#         x = conv3(inputs=x, filters=cnum * 4, strides=1, dilation_rate=16, name='conv10_atrous')

#         x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv11')
#         x = conv3(inputs=x, filters=cnum * 4, strides=1, name='conv12')
#         x = deconv(x, filters=cnum * 2, name='conv13_upsample')
#         x = conv3(inputs=x, filters=cnum * 2, strides=1, name='conv14')
#         x = deconv(x, filters=cnum, name='conv15_upsample')
#         return x
class FeatureGenerator(BaseNetwork):
    def __init__(self, _r1, _r2, init_weights=True):
        super(FeatureGenerator, self).__init__()
        self._r1 = _r1
        self._r2 = _r2
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=16, dilation=16),#conv10
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),#, align_corners=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),#, align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True)
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=64//self._r1//self._r2, out_channels=64*self._r1*self._r2, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True)
        )

        if init_weights:
            self.init_weights()
        self.subpixel = nn.PixelShuffle(self._r1)
    def forward(self, x):#x is masked images #the time of enlargement is determined at first. 
        out = self.extractor(x)
        # _bs, _ch, _h, _w = out.shape
        assert x.shape != out.shape
        out_reshaped = self.subpixel(out)# (out, self._r1, self._r2)# (_h+bound_information[0]+bound_information[1])/_h, (_w+bound_information[2]+bound_information[3])/_w)
        # print('out_reshaped size:', out_reshaped.shape)# [4,16,256,256]
        output = self.conv2d(out_reshaped)

        return output



class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, alpha=0.5, input_ch = 260):
        super(InpaintGenerator, self).__init__()
        
        self.alpha = alpha
        
        self.inpainting_gen1 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(True),
            # nn.ELU(True),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=16, dilation=16),#conv10
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True))
            # nn.ELU(True))
        

        self.inpainting_gen2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),#, align_corners=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),#, align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            # nn.ELU(True),

            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1))

        if init_weights:
            self.init_weights()

    def forward(self, x_fe, x_masked_img, mask):#x_fe is the feature
        xcat = torch.cat((x_fe, x_masked_img, mask), 1)

        xcat = self.inpainting_gen1(xcat)
        xcat = context_normalization(xcat, mask, alpha=self.alpha)
        xcat = self.inpainting_gen2(xcat)
        # xcat = torch.clamp(xcat,0,1)
        xcat = (torch.tanh(xcat) + 1) / 2
        return xcat


# def subpixel(x, r1, r2):
#     _bs, _ch0, _h0, _w0 = x.shape
#     _h = _h0 * r1
#     _w = _w0 * r2
    
#     _ch = _ch0 // r1 // r2
#     x_out = x.reshape((_bs, _ch, _h, _w))
#     # x_out = torch.zeros((_bs, _ch, _h, _w)).to(x.device)
#     # for k in range(_ch):
#     #     for i in range(_h):
#     #         for j in range(_w):
#     #             x_out[:,k,i,j] = x[:, k+_ch*r2*(i%r1)+_ch*(j%r2) ,i//r1, j//r2]  
        
#     return x_out


def context_normalization(x, mask, alpha=0.5, eps=1e-8): # known is 1 in my binary mask, but the opposite in Jiaya work. 
    # x.shape = [4,256,64,64]        
    ratio = mask.shape[2]//x.shape[2]
    downsample = nn.Upsample(scale_factor=1/ratio, mode='nearest')
    mask = downsample(mask)
    # print('mask:', mask.shape) # torch.Size([4, 1, 64, 64])   
    x_known_count = torch.sum(mask,[2,3]) #how may known pixels
    # print('x_known_count:', x_known_count, x_known_count.shape) # x_known_count: torch.Size([4, 1])
    x_known_mean = torch.sum(x*mask,[2,3])/x_known_count
    # print('x_known_mean:', x_known_mean.shape) # x_known_mean: torch.Size([4, 256])
    x_known_standard_deviation = (torch.sum((x*mask)**2,[2,3])/x_known_count - x_known_mean**2 + eps) ** (1/2)
    # print('x_known_standard_deviation:', x_known_standard_deviation.shape) #x_known_standard_deviation: torch.Size([4, 256])
    
    x_known_mean = x_known_mean.unsqueeze(2).unsqueeze(3)
    # print('x_known_mean:', x_known_mean.shape)
    x_known_standard_deviation = x_known_standard_deviation.unsqueeze(2).unsqueeze(3)
    # print('x_known_standard_deviation:', x_known_standard_deviation.shape)

    mask_rev = 1 - mask
    x_unknown_count = x.shape[2]*x.shape[3] - x_known_count
    x_unknown_mean = torch.sum(x*mask_rev,[2,3])/x_unknown_count
    x_unknown_standard_deviation = (torch.sum((x*mask_rev)**2,[2,3])/x_unknown_count - x_unknown_mean**2 + eps) ** (1/2)
    x_unknown_mean = x_unknown_mean.unsqueeze(2).unsqueeze(3)
    x_unknown_standard_deviation = x_unknown_standard_deviation.unsqueeze(2).unsqueeze(3)

    x_unknown = (x*mask_rev - x_unknown_mean)/x_unknown_standard_deviation*x_known_standard_deviation + x_known_mean
    x = alpha * x_unknown * mask_rev + (1-alpha) * x * mask_rev + x * mask
    return x #[4,256,64,64]


# def build_wgan_global_discriminator(self, x, config, reuse=False):
#         cnum = config.d_cnum
#         dis_conv = self.d_unit
#         with tf.variable_scope('D_global', reuse=reuse):
#             x = dis_conv(x, cnum, name='conv1')
#             x = dis_conv(x, cnum*2, name='conv2')
#             x = dis_conv(x, cnum*4, name='conv3')
#             x = dis_conv(x, cnum*2, name='conv4')
#             x = flatten(x, name='flatten')
#             return x

#     def build_wgan_discriminator(self, batch_global, config, reuse=False):
#         with tf.variable_scope('discriminator', reuse=reuse):
#             dglobal = self.build_wgan_global_discriminator(
#                 batch_global, config=config, reuse=reuse)
#             dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
#             return dout_global
# self.d_unit = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')


# def build_contextual_wgan_discriminator(self, batch_global, mask, config, reuse=False):
#         with tf.variable_scope('discriminator', reuse=reuse):
#             dglobal = self.build_wgan_global_discriminator(batch_global, config=config, reuse=reuse)
#             dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
#             dout_local, mask_local = self.build_wgan_contextual_discriminator(batch_global, mask,
#                                                                               config=config, reuse=reuse)
#             return dout_local, dout_global, mask_local

# def build_wgan_contextual_discriminator(self, x, mask, config, reuse=False):
#         cnum = config.d_cnum
#         dis_conv = self.d_unit
#         with tf.variable_scope('D_context', reuse=reuse):
#             h, w = x.get_shape().as_list()[1:3]
#             x = dis_conv(x, cnum, name='dconv1')
#             x = dis_conv(x, cnum*2, name='dconv2')
#             x = dis_conv(x, cnum*4, name='dconv3')
#             x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
#                                  name='dconv4')
#             mask = max_downsampling(mask, ratio=8)
#             x = x * mask
#             x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
#             mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
#             return x, mask_local

class GlobalDiscriminator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(GlobalDiscriminator, self).__init__()

        self.glo_discri = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(16, 1)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.glo_discri(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class GlobalLocalDiscriminator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(GlobalLocalDiscriminator, self).__init__()

        self.glo_discri = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(16*16*128, 1)
        self.act = nn.Sigmoid()

        self.loc_discri = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.downsample = nn.Upsample(scale_factor=1/8, mode='nearest')
        self.maxpooling = torch.nn.MaxPool2d(2)
        if init_weights:
            self.init_weights()

    
    def forward(self, x, mask):#mask: unknown part should be 1. 
        glo_x = self.glo_discri(x)
        glo_x = glo_x.view(glo_x.size(0), -1)
        glo_x = self.fc(glo_x) # glo_x is the discri probability output
        glo_x = self.act(glo_x)

        loc_x = self.loc_discri(x)

        mask_local = self.maxpooling_downsampling(mask, ratio=8)

        loc_x = loc_x * mask_local
        
        loc_x = torch.sum(loc_x,dim=[2,3]) / torch.sum(mask_local,dim=[2,3])# glo_x is the discri probability output
        # mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True) # not necessary

        return loc_x, glo_x, mask_local
    
    def maxpooling_downsampling(self, m, ratio=8):
        iters = log2(ratio)
        assert int(iters) == iters
        for _ in range(int(iters)):
            m = self.maxpooling(m)
        return m





class WGAN_GlobalLocalDiscriminator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(WGAN_GlobalLocalDiscriminator, self).__init__()

        self.glo_discri = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(16*16*128, 1)

        self.loc_discri = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1),
        )

        self.downsample = nn.Upsample(scale_factor=1/8, mode='nearest')
        self.maxpooling = torch.nn.MaxPool2d(2)
        if init_weights:
            self.init_weights()

    
    def forward(self, x, mask):#mask: unknown part should be 1. 
        glo_x = self.glo_discri(x)
        glo_x = glo_x.view(glo_x.size(0), -1)
        glo_x = self.fc(glo_x) # glo_x is the discri probability output

        loc_x = self.loc_discri(x)

        mask_local = self.maxpooling_downsampling(mask, ratio=8)

        loc_x = loc_x * mask_local
        # print('mask_local.shape =', mask_local.shape) #[4, 1, 32, 32]

        loc_x = torch.sum(loc_x,dim=[2,3]) / torch.sum(mask_local,dim=[2,3])# glo_x is the discri probability output
        # print('loc_x.shape =', loc_x.shape) #[4, 1]
        # mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True) # not necessary

        return loc_x, glo_x, mask_local
    
    def maxpooling_downsampling(self, m, ratio=8):
        iters = log2(ratio)
        assert int(iters) == iters
        for _ in range(int(iters)):
            m = self.maxpooling(m)
        return m