import torch.nn as nn 
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def reconstruction_loss(real_image, fake_image):
        reconst = torch.mean(F.l1_loss(real_image, fake_image))
        return Variable(reconst, requires_grad=True)


    def disc_loss(real_logits, fake_logits):
        real_loss = torch.min(0.0, -1 + real_logits)
        real_loss = -1 * torch.mean(real_loss)

        fake_loss = torch.min(0.0, -1 - fake_logits)
        fake_loss = -1 * torch.mean(fake_loss)
        totalloss = real_loss + fake_loss
        return Variable(totalloss, requires_grad=True)


    def generator_loss(fake_logits):
        gen_loss = -1 * torch.mean(fake_logits)
        return Variable(gen_loss, requires_grad=True)
