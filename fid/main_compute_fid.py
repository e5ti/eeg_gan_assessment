import matplotlib.pyplot as plt
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
import sys, os

from scipy import signal
sys.path.append('/home/ettore/MyCode/EEGForecasting/Z_EEG_wavelets')
sys.path.append("/home/ettore/MyCode/EEG_GAN/data/")

import numpy as np
np.random.default_rng(seed=0)

import pandas as pd

# NN MANIPULATION
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
import torch.optim as optim

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
use_cuda_ = True if torch.cuda.is_available() else False

import scipy

from models import timeGAN, AEGRU
from load_wavelet_data import load_data


def mae(y, y_pred):
   return np.mean(np.abs(y-y_pred))

def smape(y, y_pred):
   return np.mean(np.sum(2*np.abs((y - y_pred)) / (np.abs(y) + np.abs(y_pred)), axis = 1) * 100 / y.shape[1])

def easy_interp(data, final_number, n_to_merge = 3):
    data_intrp = np.zeros((final_number, ) + (data.shape[1:]))
    for fn in range(final_number):
        weights = np.random.uniform(0, 10, size=(n_to_merge))
        weights = np.exp(weights)/np.sum(np.exp(weights), axis = 0)
        
        idxs = np.random.randint(data.shape[0], size=n_to_merge)
        data_intrp[fn, ...] = np.average(data[idxs, ...], axis=0, weights=weights)
    return data_intrp, None

dsample = 1
to_interpolate = True

dir_path = os.path.dirname(os.path.realpath(__file__))
spis_dict_metrox2 = {
    "FS" : 256, 
    "my_channels" : [4,5,6,7], #[0,1,2,3,4] [0, 2, 3, 5, 7, 8, 10, 12, 13]
    "MAT_ARRAY" : ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
    "true_loc": dir_path + "/../data/wavelet_spis/",
    "fixed_len" : 600,
    "dataname" : "spis",
}

n_future = 32
n_past = 256 - n_future

b, a = signal.butter(6, 10, 'lowpass', analog = False, fs=256) # filter the generated data output
lr = 1e-3
batch_ = 64
beta1 = 0.5
h_size = 64
out_len = n_future
num_epochs = 20

list_fid_distance = []
list_synth_mean = []
list_synth_sd = []
list_pure_mean = []
list_pure_sd = []

wgd = True
sub_loc = dir_path + "/../train_fcaster_results/"
def get_sub_name(vidx, wgd, dsample = dsample):
    return f"sub{vidx}_withGenData_{wgd}_ds{dsample}.pt"

for vidx in range(0,10):
    X_train, X_test = load_data(select_subj=vidx, **spis_dict_metrox2)
    n_chans = X_train.shape[2]
    final_l = X_train.shape[1]

    y_train = X_train[:, -n_future:, :]
    y_test = X_test[:, -n_future:, :]

    X_train = X_train[::dsample, :n_past, :]
    X_test = X_test[::dsample, :n_past, :]

    if not to_interpolate:
        X_synth = np.empty((0, n_past, n_chans))
        use_cuda_ = True if torch.cuda.is_available() else False

        start_l = 64
        final_l = 256
        netG = timeGAN(start_noise_dim=start_l, end_noise_dim=final_l, in_dim = n_chans, out_ch=n_chans, 
                       hidden_dim = 64, n_layers = 5, 
                        use_cuda=use_cuda_).to(device)

        save_dir_name = "temp_name" # Saved signals folder name (e.g. generated_vigilance or generated_vigilance)
        model_loc = f"/../train_gan_results/generator_{vidx}_ds_{dsample}.pt"
        model_path = dir_path + model_loc
        noise_ = torch.randn(10000, 64, n_chans, device=device).to(device) # generate a batch of 2000 fake samples

        netG.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
        netG.eval()
        fake = netG(noise_.detach())
        fake = fake.to("cpu").detach()
        fake = signal.filtfilt(b, a, fake, axis = 1) 
        if np.min(fake) < 0:
            fake = fake + np.abs(np.min(fake))

        X_synth = fake[:, :-32, :]
    else:
        synth_data, _ = easy_interp(X_train, final_number=10000, n_to_merge=3)
        X_synth = synth_data
        fake = None

    netED = AEGRU(input_size=n_chans, hidden_size=h_size, output_size=n_chans, output_len=out_len, device=device, n_layers=2)
    optimizerED = optim.Adam(netED.parameters(), lr=lr)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0].detach()
            activation[f'{name}_hidden'] = output[1].detach()
        return hook
    
    netED.load_state_dict(torch.load(sub_loc + get_sub_name(vidx, wgd)))

    model = netED; model.eval()
    model.encoderGRU.register_forward_hook(get_activation('encoderGRU'))
    
    pure_hiddens = np.empty((0, 64))
    for i in range(0, X_train.shape[0], 128):
        x = torch.from_numpy(X_train[i:128+i,:,:]).float().to(device)
        output = model(x)
        output = activation['encoderGRU_hidden'].cpu().detach().numpy()
        pure_hiddens = np.concatenate((pure_hiddens, output[-1, :, :]))
    pure_mean = np.mean(pure_hiddens, axis = 0)
    pure_sd = np.cov(pure_hiddens, rowvar=False)

    synth_hiddens = np.empty((0, 64))
    x = torch.from_numpy(X_synth[:,:,:]).float().to(device)
    output = model(x)
    output = activation['encoderGRU_hidden'].cpu().detach().numpy()
    synth_hiddens = np.concatenate((synth_hiddens, output[-1, :, :]))
    synth_mean = np.mean(synth_hiddens, axis = 0)
    synth_sd = np.cov(synth_hiddens, rowvar=False)

    print("Computing FID...")
    evalues, evectors = np.linalg.eig(np.matmul(pure_sd, synth_sd))
    assert (evalues >= 0).all()
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    fid_distance = (np.sum(np.square(pure_mean -  synth_mean)) + np.trace(synth_sd + pure_sd - 2*sqrt_matrix)) #/ np.sum(np.abs(pure_mean))
    print(fid_distance)

    list_fid_distance.append(fid_distance)
    list_synth_mean.append(synth_mean)
    list_synth_sd.append(synth_sd)
    list_pure_mean.append(pure_mean)
    list_pure_sd.append(pure_sd)

    del fake

    torch.cuda.empty_cache()

    with torch.no_grad():
        loss_pure = 0
        loss_synth = 0
        for i in range(5):
            test_batch = torch.from_numpy(X_test[i::5, :, :], ).float().to(device)
            target_test = torch.from_numpy(y_test[i::5, :, :]).float().to(device).cpu().detach().numpy()

            netED.load_state_dict(torch.load(sub_loc + get_sub_name(vidx, wgd=False)))

            netED.eval()
            prediction_pure = netED(test_batch)[0].cpu().detach().numpy()
            loss_pure += smape(prediction_pure[:, :, :], target_test[:, :, :])

            netED.load_state_dict(torch.load(sub_loc + get_sub_name(vidx, wgd=True)))
            netED.eval()
            prediction_synth = netED(test_batch)[0].cpu().detach().numpy()
            loss_synth += smape(prediction_synth[:, :, :], target_test[:, :, :])

        loss_pure /= 5
        loss_synth /= 5

    save_dir_name = f"./smape/spis_ds{dsample}/"
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)

    np.save(f"{save_dir_name}/sub{vidx}_wgd_False.npy", np.array(loss_pure))
    np.save(f"{save_dir_name}/sub{vidx}_wgd_True.npy", np.array(loss_synth))


save_dir_name = f"./fid/spis/"
if not os.path.exists(save_dir_name):
    os.makedirs(save_dir_name)

np.save(f'{save_dir_name}/fid_{dsample}_wgd_{wgd}.npy', np.array(list_fid_distance))
np.save(f'{save_dir_name}/synth_mean_{dsample}_wgd_{wgd}.npy', np.array(list_synth_mean))
np.save(f'{save_dir_name}/pure_mean_{dsample}_wgd_{wgd}.npy', np.array(list_pure_mean))
np.save(f'{save_dir_name}/synth_sd_{dsample}_wgd_{wgd}.npy', np.array(list_synth_sd))
np.save(f'{save_dir_name}/pure_sd_{dsample}_wgd_{wgd}.npy', np.array(list_pure_sd))