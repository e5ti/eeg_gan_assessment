import numpy as np
from torch import from_numpy, full, normal
from torch.nn import BCELoss

def base_training_step(netD, netG, optimizerD, optimizerG, data, device, 
                       d_step_timing = True, noise_dim = 256, real_label=1, fake_label=0,
                        num_epochs = 10, log_step = 32, batch_= 32, emb_size = None): 

    bce_loss = BCELoss().to(device)

    for epoch in range(num_epochs):
    # For each batch in the dataloaders
        np.random.shuffle(data)
        for j in range(0, data.shape[0], batch_):
            netD.zero_grad()
            real_cpu = from_numpy(data[j:j+batch_, ...]).float().to(device)
            batch_size, seq_len = real_cpu.size(0), real_cpu.size(1)
            label = full((batch_size, 1), real_label, device=device)

            latent = normal(mean=0,
                                std=1,
                                size=(batch_size, seq_len if emb_size is None else emb_size, noise_dim)).to(device)

            label.fill_(real_label)
            real_loss = netD(real_cpu)
            d_loss_real = bce_loss(real_loss, label) 
            d_loss_real.backward()
            
            label.fill_(fake_label)
            gen_x = netG(latent)
            fake_loss = netD(gen_x.detach())
            d_loss_fake = bce_loss(fake_loss, label)
            d_loss_fake.backward()

            if epoch in d_step_timing:
                optimizerD.step()
        
            netG.zero_grad()

            label.fill_(real_label)
            output_d_fake = netD(gen_x)
            g_loss = bce_loss(output_d_fake, label)
            g_loss.backward()
            optimizerG.step() 

            if j%log_step == 0:
                print(f"Epoch: {epoch}, iteration: {int(j/batch_size)}/{int(data.shape[0]/batch_size)}")
                print(f"Loss_D_true: {d_loss_real.item():.4f} Loss_D_fake: {d_loss_fake.item():.4f} | Loss_G: {g_loss.item()}")
