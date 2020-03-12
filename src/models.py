import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import FeatureGenerator, InpaintGenerator, GlobalLocalDiscriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, EdgePerceptualLoss, SegmentationLosses, RSV_Loss
from collections import OrderedDict
from .utils import image_cropping

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.iteration = 0
        
        os.makedirs(os.path.join(config.PATH,'saved_model', self.name), exist_ok=True)
        os.makedirs(os.path.join(config.PATH,'saved_model', self.name, self.iteration), exist_ok=True)
        self.load_weights_path = os.path.join(config.PATH,'saved_model', self.name)
        
    def load(self):
        print('Loading %s generator...' % self.name)
        #Load generator model
        if len(self.config.GPU) == 1: 
            if torch.cuda.is_available():
                if self.name == 'BoundingModel':
                    pass
                    # data = torch.load(self.config.BOUNDING_MODEL_GENERATOR)
                    # self.feat_generator.load_state_dict(data['feat_generator'])
                    # self.iteration = data['iteration']
                elif self.name == 'InpaintingModel':
                    data = torch.load(self.config.INPAINTING_MODEL_GENERATOR)
                    self.feat_generator.load_state_dict(data['feat_generator'])
                    self.inp_generator.load_state_dict(data['inp_generator'])
                    self.iteration = data['iteration']
                    
                    # new_state_dict_gen = OrderedDict()
                    # for k, v in data['generator'].items():
                    #     print("loading inp generator ori dict name:", k)
                    #     name = k[7:] # remove `module.`
                    #     print("loading inp generator revised dict name:", name)
                    #     new_state_dict_gen[name] = v
                    # self.generator.load_state_dict(new_state_dict_gen)
                    self.iteration = data['iteration']
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
        elif self.config.MODE == 1:
            
            if len(self.config.GPU) == 1: 
                if torch.cuda.is_available():
                    if self.name == 'InpaintingModel':
                        print('Loading %s discriminator...' % self.name)
                        data = torch.load(self.config.INPAINTING_MODEL_DISCRIMINATOR)
                        self.discriminator.load_state_dict(data['discriminator'])
                        # new_state_dict_dis = OrderedDict()
                        # for k, v in data['discriminator'].items():
                        #     print("loading discriminator ori dict name:",k)
                        #     name = k[7:] # remove `module.`
                        #     print("loading model revised dict name:",name)
                        #     new_state_dict_dis[name] = v
                        # self.discriminator.load_state_dict(new_state_dict_dis)#,strict=False)
                    
            else:
                print('Multi-GPU is not compatible!')
                pass
                # print("Discriminator Using Multi-GPU...")
                # if torch.cuda.is_available():
                #     if self.name == 'SemanticModel':
                #         data = torch.load(self.config.SEMANTIC_MODEL_DISCRIMINATOR)
                #     elif self.name == 'InpaintingModel':
                #         data = torch.load(self.config.INPAINTING_MODEL_DISCRIMINATOR)
                # lack of 'module.'
                new_state_dict_dis = OrderedDict()
                for k, v in data['discriminator'].items():
                    print("loading discriminator ori dict name:",k)
                    name = 'module.'+k # remove `module.`
                    print("loading discriminator revised dict name:",name)
                    new_state_dict_dis[name] = v
                self.discriminator.load_state_dict(new_state_dict_dis, strict=False)
    
    def save(self):
        print('\nsaving %s...\n' % self.name)
        if self.name == 'InpaintingModel':
            torch.save({
            'iteration': self.iteration,
            'feat_generator': self.generator.state_dict()
            }, os.path.join(self.load_weights_path, self.iteration, self.name+'_gen_iter_'+str(self.iteration)+'.pth'))
            
            torch.save({
            'local_discriminator': self.local_discriminator.state_dict()
            }, os.path.join(self.load_weights_path, self.iteration, self.name+'_local_dis_iter_'+str(self.iteration)+'.pth'))
            
            torch.save({
            'global_discriminator': self.global_discriminator.state_dict()
            }, os.path.join(self.load_weights_path, self.iteration, self.name+'_global_dis_iter_'+str(self.iteration)+'.pth'))

        elif self.name == 'BoundingModel':
            torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
            }, os.path.join(self.load_weights_path, self.iteration, self.name+'_gen_iter_'+str(self.iteration)+'.pth'))


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        feat_generator = FeatureGenerator(out_h=config.HEIGHT_OUTPUT_SIZE, out_w=config.WIDTH_OUTPUT_SIZE)
        inp_generator = InpaintGenerator()
        locglo_discriminator =  GlobalLocalDiscriminator()

        # print('InpaintingModel discriminator:')
        # for i,_ in discriminator.named_parameters():
        #     print(i)
    
        # if len(config.GPU) > 1:
        #     generator = nn.DataParallel(generator, config.GPU)
        #     local_discriminator = nn.DataParallel(local_discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        rsv_loss = RSV_Loss()
        
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        
        self.add_module('feat_generator', feat_generator)
        self.add_module('inp_generator', inp_generator)
        self.add_module('locglo_discriminator', locglo_discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('rsv_loss', rsv_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=local_discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process_a(self, images, bounds, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()

        # process outputs
        outputs = self(images, bounds, masks)

        gen_loss = 0
        
        gen_unmasked_l1_loss = self.l1_loss(outputs*masks, images*masks) * 4 * self.config.UNMASKED_L1_LOSS_WEIGHT
        gen_loss += gen_unmasked_l1_loss
        
        gen_masked_rsv_l1_loss = self.rsv_loss(outputs, images, masks) * self.config.MASKED_L1_LOSS_WEIGHT
        gen_loss += gen_masked_rsv_l1_loss
        
        # create logs
        logs = [
            ("masked_rsv_l1", gen_masked_rsv_l1_loss.item()),
            ("unmasked_l1", gen_unmasked_l1_loss.item()),
        ]
        
        return outputs, gen_loss, logs

    def process_b(self, images, bounds, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, bounds, masks)

        gen_loss = 0
        dis_loss = 0
        
        # locglo_discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        real_loc_x, real_glo_x, mask_local = self.locglo_discriminator(dis_input_real, bounds, 1-masks)                # in: [rgb(3)]
        fake_loc_x, fake_glo_x, _ = self.locglo_discriminator(dis_input_fake, 1-masks)                    # in: [rgb(3)] 
        loc_dis_real_loss = self.adversarial_loss(real_loc_x, True, True, mask=mask_local) + self.adversarial_loss(real_glo_x, True, True, mask=None)
        loc_dis_fake_loss = self.adversarial_loss(real_loc_x, False, True, mask=mask_local) + self.adversarial_loss(fake_glo_x, False, True, mask=None)
        loc_dis_loss = (loc_dis_real_loss + loc_dis_fake_loss) / 4 * self.config.DISCRI_LOSS_WEIGHT
        dis_loss += loc_dis_loss
    
        # local_generator adversarial loss
        gen_input_fake = outputs
        gen_loc_fake, gen_glo_fake, _ = self.locglo_discriminator(gen_input_fake, 1-masks)                    # in: [rgb(3)]
        loc_gen_loss = (self.adversarial_loss(gen_loc_fake, True, False, mask=mask_local) + self.adversarial_loss(gen_glo_fake, True, False, mask=None)) / 2 * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += loc_gen_loss 

        gen_unmasked_l1_loss = self.l1_loss(outputs*masks, images*masks) * self.config.UNMASKED_L1_LOSS_WEIGHT
        gen_loss += gen_unmasked_l1_loss
        
        gen_masked_rsv_l1_loss = self.rsv_loss(outputs, images, masks) * self.config.MASKED_L1_LOSS_WEIGHT
        gen_loss += gen_masked_rsv_l1_loss
        
        # mrf-loss
        
        # flip-L1-loss
        # outputs_flipped = self(torch.flip(images,(3,)), torch.flip(edges,(3,)), torch.flip(masks,(3,)))
        # flip_L1_loss = self.l1_loss(outputs, torch.flip(outputs_flipped,(3,)))
        # gen_loss += flip_L1_loss

        # create logs
        logs = [
            ("loc_inp_l_d2", loc_dis_loss.item()),
            ("loc_inp_l_g2", loc_gen_loss.item()),
            ("glo_inp_l_d2", glo_dis_loss.item()),
            ("glo_inp_l_g2", glo_gen_loss.item()),
            ("masked_rsv_l1", gen_masked_rsv_l1_loss.item()),
            ("unmasked_l1", gen_unmasked_l1_loss.item()),
            ("inp_l_per", gen_content_loss.item()),
            ("inp_l_sty", gen_style_loss.item()),
            # ("l_flip", flip_L1_loss.item())      
        ]
        
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, mask_information, masks):
        images_masked = image_cropping(images,mask_information[0],mask_information[1],mask_information[2],mask_information[3])
        feat_ops = self.feat_generator(images_masked)
        outputs = self.inp_generator(feat_ops, images*masks, masks)
        return outputs
        
    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
            self.dis_optimizer.step()
        if gen_loss is not None:
            gen_loss.backward()
            self.gen_optimizer.step()