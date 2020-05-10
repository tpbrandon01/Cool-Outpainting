import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
# from torch.optim import lr_scheduler
from .networks import FeatureGenerator, InpaintGenerator, GlobalLocalDiscriminator, WGAN_GlobalLocalDiscriminator
from .loss import AdversarialLoss, RSV_Loss, IDMRFLoss, PerceptualLoss, StyleLoss, random_interpolates, relative_spatial_variant_mask
from collections import OrderedDict
from .utils import image_cropping

def isNaN(num):
    return num != num

def tensor_isNaN(ts):
    bs, ch, h, w = ts.shape
    for _b in range(bs):
        for _c in range(ch):
            for _h in range(h):
                for _w in range(w):
                    if ts[_b, _c, _h, _w] != ts[_b, _c, _h, _w]:
                        return False
    return True  
    
class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0
        
        os.makedirs(os.path.join(config.PATH,'saved_model', self.name), exist_ok=True)
        self.save_model_path = os.path.join(config.PATH,'saved_model', self.name)
        
    def load(self, load_discri=True):
        #Load generator model
        if len(self.config.GPU) == 1: 
            if torch.cuda.is_available():
                if self.name == 'BoundingModel':
                    pass
                    # data = torch.load(self.config.BOUNDING_MODEL_GENERATOR)
                    # self.feat_generator.load_state_dict(data['feat_generator'])
                    # self.iteration = data['iteration']
                elif self.name == 'InpaintingModel':
                    print('Loading %s generator...' % self.name)
                    data = torch.load(self.config.INPAINTING_MODEL_GENERATOR)
                    self.feat_generator.load_state_dict(data['feat_generator'])
                    self.inp_generator.load_state_dict(data['inp_generator'])
                    self.iteration = data['iteration']
                    print('initial iteration:',self.iteration)
                    # new_state_dict_gen = OrderedDict()
                    # for k, v in data['generator'].items():
                    #     print("loading inp generator ori dict name:", k)
                    #     name = k[7:] # remove `module.`
                    #     print("loading inp generator revised dict name:", name)
                    #     new_state_dict_gen[name] = v
                    # self.generator.load_state_dict(new_state_dict_gen)
                    # self.iteration = data['iteration']
                    if load_discri == True:
                        print('Loading %s discriminator...' % self.name)
                        data = torch.load(self.config.INPAINTING_MODEL_DISCRIMINATOR)
                        self.locglo_discriminator.load_state_dict(data['locglo_discriminator'])
            # self.generator.load_state_dict(data['generator'])
            # self.iteration = data['iteration']
        else:
            print("Multi-GPU is not compatible!")
            pass
            # print("Generator Using Multi-GPU...")
            # if torch.cuda.is_available():
            #     if self.name == 'SemanticModel':
            #         data = torch.load(self.config.SEMANTIC_MODEL_GENERATOR)
            #     elif self.name == 'InpaintingModel':
            #         data = torch.load(self.config.INPAINTING_MODEL_GENERATOR)
            # # lack of 'module.'
            # new_state_dict_gen = OrderedDict()
            # for k, v in data['generator'].items():
            #     print("loading inp generator ori dict name:",k)
            #     name = 'module.'+k # remove `module.`
            #     print("loading inp generator revised dict name:",name)
            #     new_state_dict_gen[name] = v
            # self.generator.load_state_dict(new_state_dict_gen)
            # self.iteration = data['iteration']
        # if smt model, don't load discriminator
        
        # load discriminator only when training
        # if self.config.MODE == 1:
        #     if len(self.config.GPU) == 1: 
        #         if torch.cuda.is_available():
        #             if self.name == 'InpaintingModel' and load_discri==True:
        #                 print('Loading %s discriminator...' % self.name)
        #                 data = torch.load(self.config.INPAINTING_MODEL_DISCRIMINATOR)
        #                 self.locglo_discriminator.load_state_dict(data['locglo_discriminator'])
        #                 # new_state_dict_dis = OrderedDict()
        #                 # for k, v in data['discriminator'].items():
        #                 #     print("loading discriminator ori dict name:",k)
        #                 #     name = k[7:] # remove `module.`
        #                 #     print("loading model revised dict name:",name)
        #                 #     new_state_dict_dis[name] = v
        #                 # self.discriminator.load_state_dict(new_state_dict_dis)#,strict=False)
                    
        #     else:
        #         print('Multi-GPU is not compatible!')
        #         pass
        #         # print("Discriminator Using Multi-GPU...")
        #         # if torch.cuda.is_available():
        #         #     if self.name == 'SemanticModel':
        #         #         data = torch.load(self.config.SEMANTIC_MODEL_DISCRIMINATOR)
        #         #     elif self.name == 'InpaintingModel':
        #         #         data = torch.load(self.config.INPAINTING_MODEL_DISCRIMINATOR)
        #         # lack of 'module.'
        #         new_state_dict_dis = OrderedDict()
        #         for k, v in data['discriminator'].items():
        #             print("loading discriminator ori dict name:",k)
        #             name = 'module.'+k # remove `module.`
        #             print("loading discriminator revised dict name:",name)
        #             new_state_dict_dis[name] = v
        #         self.discriminator.load_state_dict(new_state_dict_dis, strict=False)
    

    def save(self, save_discri=True):
        print('\nsaving %s...\n' % self.name)
        if self.name == 'InpaintingModel':
            os.makedirs(os.path.join(self.config.PATH,'saved_model', self.name, str(self.iteration)), exist_ok=True)
            # print("fuck", self.iteration, str(self.iteration))   #3,3         
            torch.save({
            'iteration': self.iteration,
            'feat_generator': self.feat_generator.state_dict(),
            'inp_generator': self.inp_generator.state_dict(),
            'lr': self.gen_optimizer.param_groups[0]['lr']
            }, os.path.join(self.save_model_path, str(self.iteration), self.name+'_gen_iter_'+str(self.iteration)+'.pth'))
            print('saving...LR =',self.gen_optimizer.param_groups[0]['lr'])
            if save_discri == True:    
                torch.save({
                'locglo_discriminator': self.locglo_discriminator.state_dict()
                }, os.path.join(self.save_model_path, str(self.iteration), self.name+'_locglo_dis_iter_'+str(self.iteration)+'.pth'))
                
                # torch.save({
                # 'global_discriminator': self.global_discriminator.state_dict()
                # }, os.path.join(self.save_model_path, self.iteration, self.name+'_global_dis_iter_'+str(self.iteration)+'.pth'))

        elif self.name == 'BoundingModel':##TODO
            torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
            }, os.path.join(self.save_model_path, self.iteration, self.name+'_gen_iter_'+str(self.iteration)+'.pth'))


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        self.gpu_device = config.DEVICE
        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        feat_generator = FeatureGenerator(_r1=config.HEIGHT_R1,_r2=config.WIDTH_R2)
        inp_generator = InpaintGenerator(input_ch=64*config.HEIGHT_R1*config.WIDTH_R2+4)
        if config.GAN_LOSS == "nsgan":
            print('Using GlobalLocalDiscriminator')
            locglo_discriminator =  GlobalLocalDiscriminator()
        elif config.GAN_LOSS == "wgan":
            print('Using WGAN_GlobalLocalDiscriminator')
            locglo_discriminator = WGAN_GlobalLocalDiscriminator()
        # print('InpaintingModel discriminator:')
        # for i,_ in discriminator.named_parameters():
        #     print(i)
    
        # if len(config.GPU) > 1:
        #     generator = nn.DataParallel(generator, config.GPU)
        #     local_discriminator = nn.DataParallel(local_discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        rsv_loss = RSV_Loss()
        # mrf_loss = IDMRFLoss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        
        self.add_module('feat_generator', feat_generator)
        self.add_module('inp_generator', inp_generator)
        self.add_module('locglo_discriminator', locglo_discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('rsv_loss', rsv_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        # self.add_module('mrf_loss', mrf_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.gen_optimizer = optim.Adam(
            params= list(feat_generator.parameters()) + list(inp_generator.parameters()),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        # self.gen_optimizer = optim.SGD(list(feat_generator.parameters()) + list(inp_generator.parameters()), 
        #                                 lr=0.0001, momentum=0.9)
        # self.gen_scheduler = lr_scheduler.StepLR(self.gen_optimizer, 100, gamma=0.999)
        self.dis_optimizer = optim.Adam(
            params=locglo_discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
        # self.dis_scheduler = lr_scheduler.StepLR(self.dis_optimizer, 100, gamma=0.999) # 0.999^4500=0.01108
        self.mseloss = nn.MSELoss()
    def process_a(self, images, images_masked, masks):# training process 
        #images:256*256, images_masked:128*128, masks:256*256
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        # self.feat_generator.cleargrads()
        # self.inp_generator.cleargrads()
        # process outputs
        outputs = self(images, images_masked, masks)

        gen_loss = 0
        
        gen_unmasked_l1_loss = self.l1_loss(outputs*masks, images*masks) * 4 * self.config.L1_LOSS_ALPHA
        gen_loss += gen_unmasked_l1_loss
        
        gen_masked_rsv_l1_loss = self.rsv_loss(outputs, images, masks) * self.config.RSV_LOSS_ALPHA
        gen_loss += gen_masked_rsv_l1_loss
        
        if isNaN(self.l1_loss(outputs*masks, images*masks)):
            print("NAN Warning!")
            if not tensor_isNaN(outputs):
                print("From outputs!")
            if not tensor_isNaN(images):
                print("From images!")
            if not tensor_isNaN(masks):
                print("From masks!")
        # create logs
        logs = [
            ("masked_rsv_l1", gen_masked_rsv_l1_loss.item()),
            ("unmasked_l1", gen_unmasked_l1_loss.item()),
        ]
        
        return outputs, gen_loss, logs

    def process_b(self, images, images_masked, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, images_masked, masks)

        gen_loss = 0
        dis_loss = 0
        
        # locglo_discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        real_loc_x, real_glo_x, mask_local = self.locglo_discriminator(dis_input_real, 1-masks)                # in: [rgb(3)]
        fake_loc_x, fake_glo_x, _ = self.locglo_discriminator(dis_input_fake, 1-masks)                    # in: [rgb(3)] 

        # locglo_dis_real_loss = self.adversarial_loss(real_loc_x, True, True) + self.adversarial_loss(real_glo_x, True, True)
        # locglo_dis_fake_loss = self.adversarial_loss(fake_loc_x, False, True) + self.adversarial_loss(fake_glo_x, False, True)
        # dis_loss = (locglo_dis_real_loss + locglo_dis_fake_loss) / 4 * self.config.ADV_DISCRI_LOSS_ALPHA

        lg_d_real_l = self.adversarial_loss(real_loc_x, True, True)
        lg_d_real_g = self.adversarial_loss(real_glo_x, True, True)
        lg_d_fake_l = self.adversarial_loss(fake_loc_x, False, True)
        lg_d_fake_g = self.adversarial_loss(fake_glo_x, False, True)
        dis_loss = (lg_d_real_l + lg_d_real_g + lg_d_fake_l + lg_d_fake_g) / 4 * self.config.ADV_DISCRI_LOSS_ALPHA



        # local_generator adversarial loss
        gen_input_fake = outputs
        locglo_gen_loc_fake, locglo_gen_glo_fake, _ = self.locglo_discriminator(gen_input_fake, 1-masks)                    # in: [rgb(3)]
        # locglo_gen_loss = (self.adversarial_loss(locglo_gen_loc_fake, True, False, mask=mask_local) + self.adversarial_loss(locglo_gen_glo_fake, True, False, mask=None)) / 2 * self.config.INPAINT_ADV_LOSS_ALPHA
        locglo_gen_loss = (self.adversarial_loss(locglo_gen_loc_fake, True, False) + self.adversarial_loss(locglo_gen_glo_fake, True, False)) / 2
        gen_loss += locglo_gen_loss * self.config.ADV_GEN_LOSS_ALPHA


        gen_unmasked_l1_loss = self.l1_loss(outputs*masks, images*masks) * 4 * self.config.L1_LOSS_ALPHA
        gen_loss += gen_unmasked_l1_loss * self.config.L1_LOSS_ALPHA
        
        gen_masked_rsv_l1_loss = self.rsv_loss(outputs, images, masks) * self.config.RSV_LOSS_ALPHA
        gen_loss += gen_masked_rsv_l1_loss * self.config.RSV_LOSS_ALPHA


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_ALPHA
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * (1 - masks), images * (1 - masks))
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_ALPHA
        gen_loss += gen_style_loss

        # # mrf-loss
        # gen_mrf_loss = self.mrf_loss(outputs*(1-masks), images) * self.config.MRF_ALPHA
        # gen_loss += gen_mrf_loss * self.config.MRF_ALPHA
        
        
        # flip-L1-loss
        # outputs_flipped = self(torch.flip(images,(3,)), torch.flip(edges,(3,)), torch.flip(masks,(3,)))
        # flip_L1_loss = self.l1_loss(outputs, torch.flip(outputs_flipped,(3,)))
        # gen_loss += flip_L1_loss

        # create logs
        logs = [
            ("lg_d_real_l", lg_d_real_l.item()),
            ("lg_d_real_g", lg_d_real_g.item()),
            ("lg_d_fake_l", lg_d_fake_l.item()),
            ("lg_d_fake_g", lg_d_fake_g.item()),
            ("locglo_inp_l_g2", locglo_gen_loss.item()),
            ("masked_rsv_l1", gen_masked_rsv_l1_loss.item()),
            ("unmasked_l1", gen_unmasked_l1_loss.item()),
            ("per_loss", gen_content_loss.item()),
            ("sty_loss", gen_style_loss.item())#,
            # ("inp_l_mrf", gen_mrf_loss.item())
            # ("l_flip", flip_L1_loss.item())      
        ]
        
        return outputs, gen_loss, dis_loss, logs
    




    def process_c(self, images, images_masked, masks): 
        # wgan
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, images_masked, masks)

        gen_loss = 0
        dis_loss = 0
        
        # locglo_discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()


        real_loc_x, real_glo_x, mask_local = self.locglo_discriminator(dis_input_real, 1-masks)           # in: [rgb(3)] # real_loc_x: [4,1]
        fake_loc_x, fake_glo_x, _ = self.locglo_discriminator(dis_input_fake, 1-masks)                    # in: [rgb(3)] # fake_loc_x: [4,1]

        dis_gan_loss = (-torch.mean(real_loc_x) - torch.mean(real_glo_x) + torch.mean(fake_loc_x) + torch.mean(fake_glo_x)) / 4 * self.config.ADV_DISCRI_LOSS_ALPHA
        dis_loss += dis_gan_loss    


        # local_generator adversarial loss
        gen_input_fake = outputs
        locglo_gen_loc_fake, locglo_gen_glo_fake, _ = self.locglo_discriminator(gen_input_fake, 1-masks)                    # in: [rgb(3)]
        # locglo_gen_loss = (self.adversarial_loss(locglo_gen_loc_fake, True, False, mask=mask_local) + self.adversarial_loss(locglo_gen_glo_fake, True, False, mask=None)) / 2 * self.config.INPAINT_ADV_LOSS_ALPHA
        locglo_gen_loss = (-torch.mean(locglo_gen_loc_fake) - torch.mean(locglo_gen_glo_fake)) / 2
        gen_loss += locglo_gen_loss * self.config.ADV_GEN_LOSS_ALPHA

        ###############

        # wgan with gradient penalty
        # apply penalty
        realfake_itp = random_interpolates(dis_input_real, dis_input_fake) # [4, 3, 256, 256]
        realfake_itp.requires_grad = True
        realfake_loc_x, realfake_glo_x, mask_local = self.locglo_discriminator(realfake_itp, 1-masks) #[4,1], [4,1], [4,1,32,32]
    
        # print("required_grad",dis_input_real.requires_grad, dis_input_fake.requires_grad, real_loc_x.requires_grad, realfake_loc_x.requires_grad)
        
        Mw = relative_spatial_variant_mask(1-masks)

        loc_grad = autograd.grad(outputs=realfake_loc_x, inputs=realfake_itp,
                              grad_outputs=torch.ones(realfake_loc_x.size()).to(self.config.DEVICE),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]#original c_g r_g are all True, loc_grad:[4, 3, 256, 256]
        
        glo_grad = autograd.grad(outputs=realfake_glo_x, inputs=realfake_itp,
                              grad_outputs=torch.ones(realfake_glo_x.size()).to(self.config.DEVICE),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]#original c_g r_g are all True, glo_grad:[4, 3, 256, 256]
        
        loc_grad = loc_grad * Mw
        loc_grad = loc_grad.view(loc_grad.size(0), -1)
        glo_grad = glo_grad.view(glo_grad.size(0), -1)
        # dis_loss += (torch.mean(((torch.mseloss(real_loc_x.grad * Mw))**(1/2))-1)**2) + torch.mean(((torch.mseloss(real_glo_x.grad))**(1/2)-1)**2) / 2 * self.config.WGAN_GP_LAMBDA
        dis_grad_penalty = (((loc_grad.norm(2, dim=1) - 1) ** 2).mean()+ ((glo_grad.norm(2, dim=1) - 1) ** 2).mean()) / 2 * self.config.WGAN_GP_LAMBDA
        dis_loss += dis_grad_penalty

        # l1 loss
        gen_unmasked_l1_loss = self.l1_loss(outputs*masks, images*masks) * 4 * self.config.L1_LOSS_ALPHA
        gen_loss += gen_unmasked_l1_loss * self.config.L1_LOSS_ALPHA
        
        gen_masked_rsv_l1_loss = self.rsv_loss(outputs, images, masks) * self.config.RSV_LOSS_ALPHA
        gen_loss += gen_masked_rsv_l1_loss * self.config.RSV_LOSS_ALPHA


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_ALPHA
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * (1 - masks), images * (1 - masks))
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_ALPHA
        gen_loss += gen_style_loss

        # # mrf-loss
        # gen_mrf_loss = self.mrf_loss(outputs*(1-masks), images) * self.config.MRF_ALPHA
        # gen_loss += gen_mrf_loss * self.config.MRF_ALPHA
        
        
        # flip-L1-loss
        # outputs_flipped = self(torch.flip(images,(3,)), torch.flip(edges,(3,)), torch.flip(masks,(3,)))
        # flip_L1_loss = self.l1_loss(outputs, torch.flip(outputs_flipped,(3,)))
        # gen_loss += flip_L1_loss

        # create logs
        logs = [
            ("dis_gan_loss", dis_gan_loss.item()),
            ("dis_grad_penalty", dis_grad_penalty.item()),
            ("masked_rsv_l1", gen_masked_rsv_l1_loss.item()),
            ("unmasked_l1", gen_unmasked_l1_loss.item()),
            ("per_loss", gen_content_loss.item()),
            ("sty_loss", gen_style_loss.item())#,
            # ("inp_l_mrf", gen_mrf_loss.item())
            # ("l_flip", flip_L1_loss.item())      
        ]
        
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, images_masked, masks):# images_masked:[128,128]
        # use for training, not for testing
        feat_ops = self.feat_generator(images_masked)
        outputs = self.inp_generator(feat_ops, images*masks, masks)
        return outputs
        
    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
            torch.nn.utils.clip_grad_value_(list(self.feat_generator.parameters()) + list(self.inp_generator.parameters()), 2)
            # torch.nn.utils.clip_grad_norm_(list(self.feat_generator.parameters()) + list(self.inp_generator.parameters()), 2, norm_type=2)
            self.dis_optimizer.step()
            # self.dis_scheduler.step()

        if gen_loss is not None:
            gen_loss.backward()
            torch.nn.utils.clip_grad_value_(list(self.feat_generator.parameters()) + list(self.inp_generator.parameters()), 2)
            self.gen_optimizer.step()
            # self.gen_scheduler.step()