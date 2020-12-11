import torch 
from torch.distributions import normal
from random import randint 
from torch.nn import functional as F

N = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # Initializing Normal distribution sampler

def trainer(generator, discriminator, optim_g, optim_d, losses, trainloader, device, log_interval, resolution, num_samples, save_everything):
    for epoch in range(1, n_epochs+1):
        loader = tqdm.tqdm(trainloader, desc='Loading train data')
        randomspace = [randint(4,12) for i in range(len(trainloader))]  # Generating random numbers for random crop on the fly, instead of overengineering dataloader & discriminator
        for iter, batch in enumerate(loader):
            randn = randomspace[iter] # Random by iter
            img_L, img_D = batch.to(device)
            noise = N.sample([img_L.shape[0], 1, 1, 256]).squeeze(-1).to(torch.device(device)) # Sample normal distribution noise

            # Generator 

            optim_g.zero_grad()
            fake_images = generator(noise)
            fake_L = fake_images.detach() 

            fake_logits, fake_absolute, fake_randcrop = model_d(fake_images, randn) 
            fake_logits_G = fake_logits.detach()

            gen_loss = losses.generator_loss(fake_logits_G)
            gen_loss.backward()
            optim_g.step()


            # Discriminator 
            ratio = resolution/16  # 16 because of decoder module input size is (B, 256, 16, 16) in discriminator

            real_logits, real_absolute, real_randcrop = model_d(img_D, randn)
            reconst_loss_absolute = losses.reconstruction_loss(img_L, real_absolute) # Absolute image reconstruction loss
            xy0, xy1 = int((rand-4)*ratio), int((rand+4)*ratio) # Calculate random crop proportions in prior
            reconst_loss_randcrop = losses.reconstruction_loss(F.interpolate(img_L[:,:,xy0:xy1,xy0:xy1], size=(128,128)), real_randcrop) # Random cropped image reconstruction loss
            disc_loss = losses.disc_loss(real_logits, fake_logits)
            
            disc_total_loss = reconst_loss_absolute + reconst_loss_randcrop + disc_loss

            disc_total_loss.backward()
            optim_d.step()


            if iter % log_interval == 0:
                loader.set_description(f"Epoch: {epoch} | G Loss: {gen_loss.item()} | D Loss: {disc_total_loss.item()} | Step: {iter}")
        
        # Generating samples at the end of the epoch

        with torch.no_grad():
            noise = N.sample([num_samples, 1, 1, 256]).squeeze(-1).to(torch.device(device))
            images = generator(noise).detach().cpu()
            img_array = [img.squeeze(0).transpose(2,0,1).numpy() for img in images]

    if epoch % save_freq == 0:
        if save_everything:
            torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': epoch,
                }, f'logs/{logging_dir}/model_epoch_{epoch}.pth')
        else:
            torch.save({'generator': generator.state_dict()}, f'logs/{logging_dir}/model_epoch_{epoch}.pth')


