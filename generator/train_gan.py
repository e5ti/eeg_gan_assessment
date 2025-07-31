import models
import train_module
from load_wavelet_data import load_data

import os

import numpy as np
np.random.default_rng(seed=0)

import torch
import torch.nn as nn
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

dsample = 1
dir_path = os.path.dirname(os.path.realpath(__file__))

spis_dict_metrox2 = {
    "FS" : 256, 
    "my_channels" : [4,5,6,7], #[0,1,2,3,4] [0, 2, 3, 5, 7, 8, 10, 12, 13]
    "MAT_ARRAY" : ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
    "true_loc": dir_path + "/../data/wavelet_spis/",
    "fixed_len" : 600,
    "dataname" : "spis",
}

save_dir_name = "/../train_gan_results/"
save_dir_name = dir_path + save_dir_name
if not os.path.exists(save_dir_name):
    os.mkdir(save_dir_name)
save_model_name = "generator"

lrD = 1e-5
lrG = 1e-5*5
batch_ = 32
beta1 = 0.5

num_epochs = 20
d_step_timing = np.array([x for x in range(num_epochs)])

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
use_cuda_ = True if torch.cuda.is_available() else False

real_label = 1.
fake_label = 0.

subj_list = [x for x in range(10)]

start_l = 64

for subj in subj_list:

    X_real = load_data(select_subj=subj, **spis_dict_metrox2)
    #X_real = X_real[np.squeeze(np.argwhere(np.max(X_real, axis = (1,2)) != 0)), :, :]
    X_real = X_real[::dsample, :, :]

    n_chans = X_real.shape[2]
    final_l = X_real.shape[1]

    netD = models.CNN_Discriminator(n_channels=n_chans, ngpu=1, ndf = 128).to(device)
    netG = models.timeGAN(start_noise_dim=start_l, end_noise_dim=256, 
                          in_dim = n_chans, out_ch = n_chans,
                          use_cuda=use_cuda_).to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    current_data = X_real.copy()
    log_step = 8192
    print_loss = False

    train_module.base_training_step(netD, netG, optimizerD, optimizerG, X_real, device, noise_dim = n_chans, 
                            real_label=real_label, fake_label=fake_label, num_epochs=num_epochs, log_step=log_step, emb_size=start_l, 
                            batch_=batch_, d_step_timing=d_step_timing)
    torch.save(netG.state_dict(), f"{save_dir_name}/{save_model_name}_{subj}_ds_{dsample}.pt")