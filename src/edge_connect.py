import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel#CourseInpaintingModel, RefinedInpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave, image_cropping
from .metrics import PSNR, EdgeAccuracy, SemanticAccuracy
from collections import OrderedDict
from scipy.io import loadmat

# from modeling.sync_batchnorm.replicate import patch_replication_callback
# from modeling.deeplab import DeepLab
class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'boundary-prediction'
        elif config.MODEL == 2:
            model_name = 'wide-exp-only-l1'
        elif config.MODEL == 3:
            model_name = 'wide-exp-all'
        elif config.MODEL == 4:
            model_name = 'all-joint'
        else:
            model_name = 'error'
        self.debug = False
        self.model_name = model_name
    
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        
        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        # if self.config.MODEL == 1:
        #     self.edge_model.load()
        # elif self.config.MODEL == 2:
        #     self.inpaint_model.load()
        if self.config.MODEL == 1:
            pass
        if self.config.MODEL == 2:
            if os.path.isfile(self.config.INPAINTING_MODEL_GENERATOR):
                self.inpaint_model.load(load_discri=False)
        if self.config.MODEL == 3:
            if os.path.isfile(self.config.INPAINTING_MODEL_GENERATOR) \
            and os.path.isfile(self.config.INPAINTING_MODEL_DISCRIMINATOR):
                self.inpaint_model.load()
            elif os.path.isfile(self.config.INPAINTING_MODEL_GENERATOR):
                self.inpaint_model.load(load_discri=False)
            
        elif self.config.MODEL == 4:
            pass
            # if os.path.isfile(self.config.INPAINTING_MODEL_GENERATOR):# and os.path.isfile(self.config.INPAINTING_MODEL_DISCRIMINATOR):
            #     self.inpaint_model.load()
            # else:
            #     print('=> no checkpoint found at {0}'.format(self.config.DEEPLAB_PRETRAINED_MODEL_PATH))
            
            # if os.path.isfile(self.config.SEMANTIC_MODEL_GENERATOR) and os.path.isfile(self.config.SEMANTIC_MODEL_DISCRIMINATOR):
            #     self.semantic_model.load()
            # if os.path.isfile(self.config.INPAINTING_MODEL_GENERATOR) and os.path.isfile(self.config.INPAINTING_MODEL_DISCRIMINATOR):
            #     self.inpaint_model.load()
        else:
            print('Load model error!')

    def save(self):
        if self.config.MODEL == 1:
            pass
            # self.edge_model.save()
        elif self.config.MODEL == 2:
            self.inpaint_model.save(save_discri=False)
        elif self.config.MODEL == 3:
            self.inpaint_model.save()
        elif self.config.MODEL == 4:
            pass
        
    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                
                if self.config.MODEL == 1:
                    pass
                    # self.semantic_model.train()
                elif self.config.MODEL == 2:
                    self.inpaint_model.train()
                elif self.config.MODEL == 3:
                    self.inpaint_model.train()
                elif self.config.MODEL == 4:
                    pass
                images, masks, masks_information = self.cuda(*items)
                
                # print('images.shape:', images.shape)# [4, 3, 256, 256]
                # print('masks.shape:', masks.shape)# [4, 1, 256, 256]
                # print('masks_information.shape:', masks_information.shape)# [4, 4]
                # print(masks_information)
                # edge model
                if model == 1:
                    pass
                    # train
                    # outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metrics
                    # precision, recall = self.edgeacc(edges * (1 - masks), outputs * (1 - masks))
                    # logs.append(('precision', precision.item()))
                    # logs.append(('recall', recall.item()))

                    # backward
                    # self.edge_model.backward(gen_loss, dis_loss)
                    # iteration = self.edge_model.iteration
                # inpainting model
                elif model == 2:
                    # train
                    images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
                    outputs, gen_loss, logs = self.inpaint_model.process_a(images, images_masked, masks)

                    outputs_merged = outputs * (1 - masks) + images * masks
                    
                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))

                    masks_pixel = torch.sum((masks_information[:,1]-masks_information[:,0])*(masks_information[:,3]-masks_information[:,2]))
                    _bs, _ch, _h, _w = images.shape
                    mae = ( torch.sum(torch.abs(images - outputs_merged)) / (_ch*(_bs*_h*_w-masks_pixel)) ).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))


                    # backward
                    self.inpaint_model.backward(gen_loss)
                    iteration = self.inpaint_model.iteration

                # inpainting model
                elif model == 3:
                    # train
                    images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
                    if self.config.GAN_LOSS == "nsgan":
                        outputs, gen_loss, dis_loss, logs = self.inpaint_model.process_b(images, images_masked, masks)
                    elif self.config.GAN_LOSS == "wgan":
                        outputs, gen_loss, dis_loss, logs = self.inpaint_model.process_c(images, images_masked, masks)
                    outputs_merged = outputs * (1 - masks) + images * masks
                    
                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))

                    masks_pixel = torch.sum((masks_information[:,1]-masks_information[:,0])*(masks_information[:,3]-masks_information[:,2]))
                    _bs, _ch, _h, _w = images.shape
                    mae = ( torch.sum(torch.abs(images - outputs_merged)) / (_ch*(_bs*_h*_w-masks_pixel)) ).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))


                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                # inpainting model
                elif model == 4:
                    print('Not done yet')
                    pass

                else:
                    print("Wrong model number")
                    
                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    # print("Sampling Model, iteration =",iteration) # 2
                    self.sample(it=iteration)

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    # print("Saving Model, iteration =",iteration)  #2
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, masks, masks_information = self.cuda(*items)

            if model == 1:
                pass
            # inpainting model
            elif model == 2:
                images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
                with torch.no_grad():
                    outputs, gen_loss, logs = self.inpaint_model.process_a(images, images_masked, masks)
                
                outputs_merged = outputs * (1 - masks) + images * masks
                
                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(1-masks)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('eval_gen_loss',gen_loss.item()))

            elif model == 3:
                images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
                if self.config.GAN_LOSS == "nsgan":
                    outputs, gen_loss, logs = self.inpaint_model.process_b(images, images_masked, masks)
                elif self.config.GAN_LOSS == "wgan":
                    outputs, gen_loss, logs = self.inpaint_model.process_c(images, images_masked, masks)
                outputs_merged = outputs * (1 - masks) + images * masks
                
                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(1-masks)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('eval_gen_loss',gen_loss.item()))
            # joint model
            if model == 4:
                pass

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):# cannot use, need to write testing process
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.BATCH_SIZE
        )

        index = 0
        max_index = self.config.TEST_SAMPLE_NUMBER if self.config.TEST_SAMPLE_NUMBER is not -1 else 100000
        for items in test_loader: 
            name = self.test_dataset.load_name(index)
            images, masks, masks_information = self.cuda(*items)
            index += self.config.BATCH_SIZE
            
            if index > max_index:
                break 
            
        
            if model == 1:
                pass
            elif model == 2:
                images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
                outputs, _, logs = self.inpaint_model.process_a(images, images_masked, masks)
                outputs_merged = outputs * (1 - masks) + images * masks
            elif model == 3:
                images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
                if self.config.GAN_LOSS == "nsgan":
                    outputs, _, _, logs = self.inpaint_model.process_b(images, images_masked, masks)
                if self.config.GAN_LOSS == "wgan":
                    outputs, _, _, logs = self.inpaint_model.process_c(images, images_masked, masks)
                outputs_merged = outputs * (1 - masks) + images * masks
            elif model == 4:
                outputs_merged = outputs * (1 - masks) + images * masks
            else:
                print("test model error")

            for i in range(outputs_merged.shape[0]):
                output = self.smt_postprocess(outputs_merged)[i]
                path = os.path.join(self.results_path, name)
                print(index+i, name)
                imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            print("Validation set is empty...")
            return

        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)

        images, masks, masks_information = self.cuda(*items)
    
        # bound model
        if model == 1:
            pass
        elif model == 2:
            inputs = images * masks
            images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
            outputs, _, _ = self.inpaint_model.process_a(images, images_masked, masks)
            outputs_merged = outputs * (1 - masks) + images * masks
        elif model == 2 or model == 3:
            inputs = images * masks
            images_masked = image_cropping(images, masks_information)# images: 256*256, images_masked:128*128
            if self.config.GAN_LOSS == "nsgan":
                outputs, _, _, _ = self.inpaint_model.process_b(images, images_masked, masks)
            elif self.config.GAN_LOSS == "wgan":
                outputs, _, _, _ = self.inpaint_model.process_c(images, images_masked, masks)
            outputs_merged = outputs * (1 - masks) + images * masks
        elif model == 4:
            outputs = self.semantic_model(images, bounds, masks)
            outputs_merged = outputs * (1 - masks) + images * masks
        else:
            print('sample model error')

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
            
        if model == 2 or model == 3 or model == 4: 
            images = stitch_images(
                self.postprocess(images),

                self.postprocess(inputs),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )
 
        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
    
    def smt_postprocess(self, img):
        # input img = [0, 255]
        img = img.permute(0, 2, 3, 1)
        return img.int()


