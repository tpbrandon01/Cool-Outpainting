import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)#gt get how many right 
        precision = torch.sum(true_positive) / (selected + 1e-8)# predict get how many right

        return precision, recall


class SemanticAccuracy(nn.Module):

    # metrics
    # IOU = self.semanticacc(smt_onehs * (1 - masks), outputs * (1 - masks))
    
    """
    Measures the accuracy of the semantic label map
    """
    def __init__(self):
        super(SemanticAccuracy, self).__init__()

    def __call__(self, gts, outputs): #gts are gt_onehot [bs,class_num, w, h], if the pixel is masked, the whole vector is zero
        #check mask pixels num
        #下面那個或是要是unmasked_pixels??
        masked_pixels, labels = torch.max(gts, dim=1)# values, indices # unmasked pixel are one-hots. #masked pixel are all zero.
        #[bs,w,h]
        bs, w, h = masked_pixels.shape
        masked_pixels = torch.sum(masked_pixels) # masked pixel sum
        unmasked_pixels = bs*w*h - masked_pixels # unmasked pixel sum 
        outputs = torch.argmax(outputs, dim=1)
        corrects = torch.sum(labels==outputs)
        IOU = (corrects-unmasked_pixels) / (masked_pixels + 1e-8)# predict get how many right
        return IOU



class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
