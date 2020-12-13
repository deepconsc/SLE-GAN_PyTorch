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
        real_loss = torch.min(torch.tensor([0.0]), torch.tensor([-1.0]) + real_logits)
        real_loss = torch.tensor([-1.0]) * torch.mean(real_loss)

        fake_loss = torch.min(torch.tensor([0.0]), torch.tensor([-1.0]) - fake_logits)
        fake_loss = torch.tensor([-1.0]) * torch.mean(fake_loss)
        totalloss = real_loss + fake_loss
        return Variable(totalloss, requires_grad=True)


    def generator_loss(fake_logits):
        gen_loss = torch.tensor([-1.0]) * torch.mean(fake_logits)
        return Variable(gen_loss, requires_grad=True)
