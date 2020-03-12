import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, SemanticModel, InpaintingModel#CourseInpaintingModel, RefinedInpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy, SemanticAccuracy
from collections import OrderedDict
from scipy.io import loadmat

# from modeling.sync_batchnorm.replicate import patch_replication_callback
# from modeling.deeplab import DeepLab
import modeling.deeplab as deeplab
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
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_SEMANTIC_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_SEMANTIC_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_SEMANTIC_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
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
            if os.path.isfile(self.config.INPAINTING_MODEL_FEATURE_GENERATOR) \
            and os.path.isfile(self.config.INPAINTING_MODEL_INPAINTING_GENERATOR):
                self.inpaint_model.load()
        if self.config.MODEL == 3:
            if os.path.isfile(self.config.INPAINTING_MODEL_FEATURE_GENERATOR) \
            and os.path.isfile(self.config.INPAINTING_MODEL_INPAINTING_GENERATOR) \
            and os.path.isfile(self.config.INPAINTING_MODEL_INPAINTING_DISCRIMINATOR):
                self.inpaint_model.load()
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
            self.inpaint_model.save()
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
                images, semantics, unlbl_binary_mask, smt_onehs, masks = self.cuda(*items)
                
                # edge model
                if model == 1:
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
                    outputs, gen_loss, _, logs = self.inpaint_model.process_a(images, bound, masks)

                    # metrics
                    IOU = self.semanticacc(smt_onehs * (1 - masks), outputs * (1 - masks))
                    # print('smt_onehs, outputs, masks =',smt_onehs.shape, outputs.shape, masks.shape)
                    logs.append(('IOU', IOU.item()))


                    # backward
                    self.inpaint_model.backward(gen_loss)
                    iteration = self.semantic_model.iteration

                # inpainting model
                elif model == 3:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process_a(images, bound, masks)

                    # print('smt_onehs, outputs, masks =',smt_onehs.shape, outputs.shape, masks.shape)
                    logs.append(('IOU', IOU.item()))


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
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
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
            images, semantics, unlbl_binary_mask, smt_onehs, masks = self.cuda(*items)

            if model == 1:
                pass
            # inpainting model
            elif model == 2 or model == 3:
                # train
                outputs, _, _, logs = self.semantic_model.process(images, bounds, masks)

                # metrics
                IOU = self.semanticacc(semantics * (1 - masks), outputs * (1 - masks))
                logs.append(('IOU', IOU.item()))
           
            # joint model
            if model == 4:
                pass

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
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
            images, semantics, unlbl_binary_mask, smt_onehs, masks = self.cuda(*items)
            index += self.config.BATCH_SIZE
            
            if index > max_index:
                break 
            
            with torch.no_grad():
                elif model == 1:
                    pass
                elif model == 2 or model == 3:
                    outputs = self.inpaint_model(images, bounds, masks)
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
        # images, images_gray, semantics, smt_onehs, edges, masks = self.cuda(*items)
        images, semantics, unlbl_binary_mask, smt_onehs, masks = self.cuda(*items)
        with torch.no_grad():
            # bound model
            if model == 1:
                pass
            elif model == 2 or model == 3:
                outputs = self.semantic_model(images, bounds, masks)
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
                self.smt_postprocess(inputs),
                self.smt_postprocess(outputs),
                self.smt_postprocess(outputs_merged),
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


