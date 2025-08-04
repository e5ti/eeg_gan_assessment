import torch
import torch.nn as nn

# Discriminator
# Specific for 256 samples long windows
class CNN_Discriminator(nn.Module):
    def __init__(self, n_channels, ngpu, ndf = 128):
        super(CNN_Discriminator, self).__init__()
        self.ndf = ndf
        self.ngpu = ngpu
        self.n_channels = n_channels
        self.main = nn.Sequential(
            # input is ``(nc) x 256``
            nn.Conv1d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 128``
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 64``
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 32``
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 16``
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 8``
            nn.Conv1d(ndf * 16, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4``
            nn.Conv1d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.permute(0,2,1)).permute(0,2,1)[:,:,0]
    

# Generator
class timeGAN(nn.Module):
    def __init__(self, start_noise_dim = 64, end_noise_dim = 256, 
                 in_dim = 6, hidden_dim = 64, n_layers = 5, out_ch = 6, use_cuda = False):
        super(timeGAN, self).__init__()

        self.start_noise_dim = start_noise_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.end_noise_dim = end_noise_dim
        n_upscale = int(end_noise_dim / start_noise_dim)

        self.noise_dec = nn.Sequential()
        for n in range(1, n_upscale):
            self.noise_dec.append(nn.Linear(start_noise_dim * (n), start_noise_dim * (n+1)))
            self.noise_dec.append(nn.LeakyReLU(0.2, inplace=True))

        self.gru_layer = nn.GRU(in_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.out_layer = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, out_ch))

    def forward(self, input):
        batch_size = input.shape[0]
        if self.use_cuda:
            h_0 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).cuda()
        else:
            h_0 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)
        
        noise_lbld = self.noise_dec(input.swapaxes(1, 2)).swapaxes(1, 2)
        recurrent_features, _ = self.gru_layer(noise_lbld, h_0)
        outputs = self.out_layer(recurrent_features)

        return outputs

# Encoder and Decoder architectures
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, device, dropout_p=0.1):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, int(n_layers), batch_first=True, device=device)

    def forward(self, input, input_h):
        if input_h is None:
            input_h = self.init_hidden(input.shape[0]) # Should insert a batch_first option
            output, hidden = self.gru(input, input_h)
        else:
            output, hidden = self.gru(input, input_h)        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if batch_size != 0:
            hidden = torch.squeeze(weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(self.device))
        else:
            hidden = torch.squeeze(weight.new(self.n_layers, self.hidden_size).zero_().to(self.device))
        return hidden

class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, n_layers):
        super(DecoderGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, int(n_layers), batch_first=True, device = device)

        self.out_1 = nn.Linear(hidden_size, int((output_size+hidden_size)/2), device = device)
        self.out_2 = nn.Linear(int((output_size+hidden_size)/2), output_size, device = device)
        self.single_out = nn.Linear(hidden_size, output_size, device = device)
        self.tdd = nn.Linear(hidden_size, output_size, device = device)
        self.device = device

    def forward(self, input, encoder_hidden):
        
        dec_out, dec_hidden = self.gru(input, encoder_hidden)
        dec_out = self.tdd(dec_out)

        return dec_out, dec_hidden

# Encoder-Decoder
class AEGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len, device, dropout_p=0.0, n_layers=2):
        super(AEGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_len = output_len
        self.device = device
        self.encoderGRU = EncoderGRU(input_size, hidden_size, n_layers,device, dropout_p)
        self.decoderGRU = DecoderGRU(hidden_size, hidden_size, output_size, device, n_layers)

    def forward(self, input_gru, input_h = None):

        enc_out, encoded_h = self.encoderGRU(input_gru, input_h = None)
        decoder_input = torch.zeros(encoded_h.shape[1], self.output_len, self.hidden_size).to(self.device)
        dec_o, dec_h = self.decoderGRU(decoder_input, encoded_h)

        return dec_o, dec_h