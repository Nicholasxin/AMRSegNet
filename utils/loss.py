import torch
from torch.autograd import Function
import torch.nn.functional as F 
import torch.nn as nn 
from itertools import repeat
import numpy as np
from torch.autograd import Variable
import pdb
import SimpleITK as sitk
import time


class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def forward(self, out, labels):
        labels = labels.float()
        #print('labels.shape: ', labels.shape)
        out = out.float()
        #print('out.shape: ', out.shape)
        smooth = 1 
        cond = labels[:, 0] >= 0 # first element of all samples in a batch 
        #print('cond: ', cond)
        nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
        #print('nnz', nnz)
        nbsup = len(nnz)
        print('nbsup:', nbsup)
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
            masked_labels = labels[cond]
            #print('masked_labels.shape:', masked_labels.shape)

            iflat = masked_labels.float().view(-1)
            tflat = masked_outputs.float().view(-1)
             
            intersection = torch.dot(iflat, tflat)
            dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
            loss = 1. - dice

            #print('IOU: %6f, soft dice: %6f, output.sum:, %f, target.sum: %f, dice_loss: %6f' % 
                  #(intersection, dice, iflat.sum(), tflat.sum(), loss))

            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

class DiceLoss(nn.Module):
    def __init__(self, class_weights):
        super(DiceLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, input_, target):
        output = input_
        output = output.float()
        #pdb.set_trace()
        # tranverse the size for CELoss
        # output = output.view(output.shape[0], 256, 256)
        # output1 = 1 - output
        # output = torch.stack((output, output1), dim=1)
        # target = target.view(output.shape[0], 256, 256)
        
        # CE_loss
        # ce_loss = F.cross_entropy(output, target.long(), weight=self.class_weights)

        # dice loss 
        smooth = 1
        # pdb.set_trace()
        iflat = output.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = torch.dot(iflat, tflat)
        
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
        dice_loss = 1. - dice
        #loss = dice_loss + ce_loss
        #loss = dice

        # L1_loss
        l1 = L1_loss(iflat, tflat)
        
        mix_loss = dice_loss + l1

        # print('contiguous.view: %f s'%(t2-t1))
        # print('torch.dot: %f s'%(t22-t2))
        # print('torch.sum: %f s'%(t3-t22))
        # print('l1_loss torch.mean: %f s'%(t4-t3))

        print('IOU: %6f, soft dice: %6f, dice_loss: %6f, l1_loss: %6f' % 
              (intersection, dice, dice_loss, l1))
        # return mix_loss
        return dice_loss
    
    @staticmethod
    def dice_coeficient(output, target):
        output = output.float()
        target = target.float()
        
        smooth = 1e-20
        iflat = output.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = torch.dot(iflat, tflat)
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
        jaccard = (intersection + smooth) / (iflat.sum() + tflat.sum() - intersection + smooth)

        return dice, jaccard

def L1_loss(output, target):
    # output = output.float()
    # target = target.float()

    # iflat = output.contiguous().view(-1)
    # tflat = target.contiguous().view(-1)

    return torch.mean(torch.abs(output-target))


class SurfaceLoss(nn.Module):
    #surface loss 
    def __init__(self):
        super(SurfaceLoss, self).__init__()

    def forward(self, output, bounds, target):
        output = output.float()
        bounds = bounds.float()
        target = target.float()

        iflat = output.view(-1)
        tflat = bounds.view(-1)

        loss = torch.mean(iflat * tflat)

        return loss
         
class TILoss(nn.Module):
    # Tversky index loss
    def __init__(self):
        super(TILoss, self).__init__()

    def forward(self, output, target):
        beta = 0.5
        alpha = 0.5
        smooth = 1
        output = output.float()
        target = target.float()

        pi = output.view(-1)
        gi = target.view(-1)
        p_ = 1 - pi
        g_ = 1 - gi
        
        intersection = torch.dot(pi, gi)
        inter_alpha = torch.dot(p_, gi)
        inter_beta = torch.dot(g_, pi)
        
        ti = (intersection + smooth) / (intersection + alpha*inter_alpha + beta*inter_beta + smooth)
        print('ti:{}'.format(ti.item()))

        #sigma = 0.5
        #loss = torch.exp(-(ti)**2 / (2*sigma**2))

        loss = (1 - ti)

        #loss = -(1-ti)**2*torch.log(ti+1e-6)
        
        return loss, ti


class DiceCoeff(Function):
    def __init__(self, *args, **kwargs):
        # torch.cuda.set_device(1)
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        # print('input.shape: ', input.shape)
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)

        #epsilon = 0.2
        #output = input[:, 0] - input[:, 1] > epsilon
        #output = output.float()
        #result_ = torch.squeeze(output)
        

        # result_ = input
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        #print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #    union, intersect, target_sum, result_sum, 2*IoU))
        # if IoU.is_cuda:
        #     out = torch.cuda.FloatTensor(1).fill_(2*IoU)
        # else:
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        if input.is_cuda:
            grad_output = grad_output.cuda()
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
    
        grad_input = torch.cat((torch.mul(torch.unsqueeze(dDice, 1), grad_output[0]),
                                torch.mul(torch.unsqueeze(dDice, 1), -grad_output[0])), dim=1)

        return grad_input , None
