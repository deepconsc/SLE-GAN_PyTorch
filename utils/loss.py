import torch.nn as nn 
import torch
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def reconstruction_loss(real_image, fake_image):
        return torch.mean(F.l1_loss(real_image, fake_image))


    def disc_loss(real_logits, fake_logits):
        real_loss = torch.min(0.0, -1 + real_logits)
        real_loss = -1 * torch.mean(real_loss)

        fake_loss = torch.min(0.0, -1 - fake_logits)
        fake_loss = -1 * torch.mean(fake_loss)

        return real_loss + fake_loss


    def generator_loss(fake_logits):
        return -1 * torch.mean(fake_logits)
