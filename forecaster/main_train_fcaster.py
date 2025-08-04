import os
from load_wavelet_data import load_data
from models import timeGAN, AEGRU

import numpy as np
np.random.default_rng(seed=0)

import random
random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

from scipy import signal

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
use_cuda_ = True if torch.cuda.is_available() else False

dsample = 1
is_gen_data = [True, False]
n_synth_samples = 10000
dir_path = os.path.dirname(os.path.realpath(__file__))

save_dir_name = "/../train_fcaster_results/"
save_dir_name = dir_path + save_dir_name
if not os.path.exists(save_dir_name):
    os.mkdir(save_dir_name)
n_future = 32
n_past = 256 - n_future

spis_dict_metrox2 = {
    "FS" : 256, 
    "my_channels" : [4,5,6,7], #[0,1,2,3,4] [0, 2, 3, 5, 7, 8, 10, 12, 13]
    "MAT_ARRAY" : ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
    "true_loc": dir_path + "/../data/wavelet_spis/",
    "fixed_len" : 600,
    "dataname" : "spis",
}

lr = 1e-4
batch_ = 64
beta1 = 0.5

criterion = nn.L1Loss().to(device)

h_size = 64
out_len = n_future
num_epochs = 50

for vidx in range(0, 10):
    for igd in is_gen_data:
        WITH_GEN_DATA = igd
        # set subject to perform eval onto
        val_idx = vidx

        # init train, test data arrays
        data_train, data_test = load_data(select_subj=vidx, **spis_dict_metrox2)
        #X_real = X_real[np.squeeze(np.argwhere(np.max(X_real, axis = (1,2)) != 0)), :, :]
        data_train = data_train[::dsample, :, :]
        data_test = data_test[::dsample, :, :]

        n_chans = data_train.shape[2]
        final_l = data_train.shape[1]

        X_train = data_train[:, :n_past, :]
        y_train = data_train[:, -n_future:, :]

        X_test = data_test[:, :n_past, :]
        y_test = data_test[:, -n_future:, :]

        if WITH_GEN_DATA:
            X_synth = np.empty((0, n_past, n_chans))
            noise_ = torch.randn(n_synth_samples, 64, n_chans, device=device, requires_grad=False).to(device)

            start_l = 64
            final_l = 256
            netG = timeGAN(start_noise_dim=start_l, end_noise_dim=final_l, in_dim = n_chans, out_ch=n_chans, hidden_dim = 64, n_layers = 5, 
                            use_cuda=use_cuda_).to(device)
            b, a = signal.butter(6, 10, 'lowpass', analog = False, fs=256) # filter the generated data output
            model_name = f"generator_{vidx}_ds_{dsample}.pt"
            model_path = dir_path + "/../train_gan_results/" + model_name

            netG.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))

            netG.eval()
            fake = netG(noise_.detach())
            fake = fake.to("cpu").detach().numpy()
            fake = signal.filtfilt(b, a, fake, axis = 1) 
            if np.min(fake) < 0:
                fake = fake + np.abs(np.min(fake))

            X_train = np.concatenate((X_train, fake[:, :-n_future, :]), axis = 0)
            y_train = np.concatenate((y_train, fake[:, n_past:, :]), axis = 0)
            del fake
            del noise_

        netED = AEGRU(input_size=n_chans, hidden_size=h_size, output_size=n_chans, 
                      output_len=out_len, device=device, n_layers=2)
        optimizerED = torch.optim.Adam(netED.parameters(), lr=lr)

        CHANNEL_CHOSEN = [x for x in range(n_chans)]
        idxs = np.arange(X_train.shape[0])

        save_name = f"sub{vidx}_withGenData_{WITH_GEN_DATA}_ds{dsample}"

        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            np.random.shuffle(idxs)
            for j in range(0, len(idxs), batch_):

                netED.zero_grad()
                train_batch = torch.from_numpy(X_train[idxs[j:j+batch_], :, :], ).float().to(device)
                target_batch = torch.from_numpy(y_train[idxs[j:j+batch_], :, :]).float().to(device)
                output, _ = netED(train_batch)

                loss = criterion(output, target_batch) 
                loss.backward() 

                optimizerED.step()
                
            with torch.no_grad():
                    netED.eval()
                    print(f'[{epoch}/{num_epochs}] Loss: {loss.item()}')
                    test_batch = torch.from_numpy(X_test[::5, :, :], ).float().to(device)
                    target_test = torch.from_numpy(y_test[::5, :, :]).float().to(device)
                    print(f'Val loss: {criterion(netED(test_batch)[0], target_test).item()}')

            netED.train(True)

        torch.save(netED.state_dict(), f"{save_dir_name}/{save_name}.pt")


"""         
        netED.eval()
        with torch.no_grad():
            loss = 0
            for i in range(10):
                test_batch = torch.from_numpy(X_test[i::10, :, :], ).float().to(device)
                target_test = torch.from_numpy(y_test[i::10, :, :]).float().to(device).cpu().detach().numpy()
                prediction = netED(test_batch)[0].cpu().detach().numpy()

                print(prediction.shape)
                print(target_test.shape)
                loss += criterion(torch.from_numpy(prediction[:, :, :]), torch.from_numpy(target_test[:, :, :])).item()

            loss = loss / 10
            np.save(f'./BY_SUBJ_SPIS/{save_name}', np.array(loss)) 
            
"""

