'''
* @author: EmpyreanMoon
*
* @create: 2025-01-02 15:40
*
* @description: The overview of k2VAE
'''

import torch.nn as nn
import torch
import math
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from probts.model.nn.prob.k2VAE.auxiliarynet import Transformer as AuxiliaryNet
from probts.model.nn.prob.RevIN import RevIN
from einops import rearrange
import torch.nn.functional as F

class latent_transitionerNN(nn.Module):

    def __init__(self, dynamic_dim, hidden_dim=64):
        super(latent_transitionerNN, self).__init__()

        self.fc1 = nn.Linear(dynamic_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dynamic_dim)


    def forward(self, z, pred_len=1):
        B, L, E = z.shape


        z_flat = z.reshape(B * L, E)
        h = torch.relu(self.fc1(z_flat))
        out = self.fc2(h).reshape(B, L, E)

        z_rec = out
        if pred_len <= L:
            z_pred = out[:, -pred_len:, :]
        else:

            last_h = out[:, -1:, :]  # B,1,E
            preds = [last_h]
            for _ in range(pred_len - 1):
                h_tmp = torch.relu(self.fc1(preds[-1].reshape(B, E)))
                next_step = self.fc2(h_tmp).reshape(B, 1, E)
                preds.append(next_step)
            z_pred = torch.cat(preds, dim=1)

        return z_rec, z_pred

class transition_zNN(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        input_dim = state_dim * 3

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, state_dim)
        self.fc_logvar = nn.Linear(hidden_dim, state_dim)

    def forward(self, x, U_t, Y_t):

        B, seq_len, _ = U_t.shape
        predictions = []
        mus, vars_ = [], []

        for t in range(seq_len):
            inp = torch.cat([x, U_t[:, t, :], Y_t[:, t, :]], dim=-1)
            h = torch.relu(self.fc1(inp))

            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            var = torch.exp(logvar) + 1e-6

            predictions.append(mu)
            mus.append(mu)
            vars_.append(var)
            x = mu

        prediction = torch.stack(predictions, dim=1)
        mu_all = torch.stack(mus, dim=1)
        var_all = torch.stack(vars_, dim=1)

        dist = MultivariateNormal(
            loc=mu_all,
            covariance_matrix=torch.diag_embed(var_all)
        )

        return prediction, dist

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
class MLP2(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class estimatenoise(nn.Module):

    def __init__(self,  latent_size,lags=1, num_layers=3, hidden_dim=64, compress_dim=10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        self.gs = nn.ModuleList([MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in
                                 range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP2(input_dim=lags * latent_size + 1, hidden_dim=hidden_dim,
                  output_dim=1, num_layers=num_layers) for _ in range(latent_size)])
        print('input_dim',lags * latent_size + 1)

        self.compress = nn.Linear(lags * latent_size, compress_dim)
        self.compress_dim = compress_dim
        # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)

        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        if x.shape[-1] > 100:
            batch_x_lags = self.compress(batch_x_lags)
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)

            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                    (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)


            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)

        return residuals

class COFEVAE(nn.Module):
    def __init__(self, config):
        super(COFEVAE, self).__init__()
        self.config = config
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.multistep = config.multistep
        self.dynamic_dim = config.dynamic_dim
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars

        self.freq = math.ceil(self.input_len / self.patch_len)
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len
        self.latent_transitioner = latent_transitionerNN(config.dynamic_dim)
        self.transition_z = transition_zNN(state_dim=self.dynamic_dim)
        self.auxiliary = AuxiliaryNet(self.config)
        self.encoder = MLP(f_in=self.patch_len * self.enc_in, f_out=self.dynamic_dim, activation='relu',
                          hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.decoder_mu = MLP(f_in=self.dynamic_dim, f_out=self.patch_len * self.enc_in, activation='relu',
                              hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.decoder_sigma = MLP(f_in=self.dynamic_dim, f_out=self.patch_len * self.enc_in, activation='relu',
                                 hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.transition = MLP(f_in=self.dynamic_dim, f_out=self.patch_len * self.enc_in * 2, activation='relu',
                                 hidden_dim=self.hidden_dim, hidden_layers=1)
        self.compress = MLP(f_in=self.patch_len * self.enc_in * 2, f_out=self.dynamic_dim, activation='relu',
                                 hidden_dim=self.hidden_dim, hidden_layers=1)
        self.estimatenoise = estimatenoise(lags=1,
                                           latent_size=self.enc_in,
                                           num_layers=1,
                                           hidden_dim=self.hidden_dim)
        self.revin = RevIN(self.enc_in)

    def encode(self, x):
        B, L, C = x.shape
        x = self.revin(x, 'norm')

        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)

        res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len

        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC

        res = self.encoder(res)  # B F H

        z_rec, Y = self.latent_transitioner(res, self.step)  # B F H, B S H
        x_residual = res - z_rec

        residual = self.auxiliary(x_residual)
        pred, dist = self.transition_z(res[:, -1, :], residual, Y)
        pred = pred + residual

        z_pred = self.transition(pred)
        z_rec_temp = z_rec
        z_rec = self.transition(z_rec)
        z_pred_temp = z_pred

        B, _, _= z_pred.shape
        z_rec=z_rec.reshape(B, self.freq, self.patch_len, 2*self.enc_in).reshape(B, -1, 2 * self.enc_in)
        z_pred = z_pred.reshape(B, self.step, self.patch_len, 2 * self.enc_in).reshape(B, -1, 2 * self.enc_in)

        Z = torch.cat((z_rec, z_pred), dim=1)
        z_s = Z.reshape(B, -1, 2*self.enc_in)[:, :self.pred_len, :self.enc_in]
        z_f = Z.reshape(B, -1, 2*self.enc_in)[:, :self.pred_len, self.enc_in:]

        residual_z_s = self.estimatenoise(z_s)
        residual_z_f = self.estimatenoise(z_f)
        z_pred_temp = self.compress(z_pred_temp)
        return z_rec_temp, z_pred_temp, dist,residual_z_s,residual_z_f


    def decode(self, x_rec, x_pred):
        B, L, C = x_rec.shape
        x_rec = self.decoder_mu(x_rec)  # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C

        x_pred_mu = self.decoder_mu(x_pred)  # B S PC
        x_pred_mu = x_pred_mu.reshape(B, self.step, self.patch_len, self.enc_in)
        x_pred_mu = x_pred_mu.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C

        x_pred_sigma = self.decoder_sigma(x_pred)  # B S PC
        x_pred_sigma = x_pred_sigma.reshape(B, self.step, self.patch_len, self.enc_in)
        x_pred_sigma = x_pred_sigma.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C

        x_rec = self.revin(x_rec, 'denorm')
        x_pred_mu = self.revin(x_pred_mu, 'denorm')
        x_pred_sigma = self.revin(x_pred_sigma, 'denorm')

        dist = Normal(loc=x_pred_mu, scale=F.softplus(x_pred_sigma) + 1e-6)
        return x_rec, dist

    def forward(self, x):
        x_rec, pred, prior_dist,residual_z_s,residual_z_f = self.encode(x)
        pred = pred + prior_dist.rsample()
        x_rec, post_dist = self.decode(x_rec, pred)

        return x_rec, prior_dist, post_dist,residual_z_s,residual_z_f

    @torch.no_grad()
    def sample(self, x, num_samples):
        x_rec, pred, _,_,_ = self.encode(x)
        dist_list = []
        sample_list = []
        for i in range(self.config.sample_schedule):
            t = pred + torch.randn_like(pred).to(x.device)
            _, post_dist = self.decode(x_rec, t)
            dist_list.append(post_dist)

        cnt = 0
        k = num_samples // self.config.sample_schedule
        for id, dist in enumerate(dist_list):
            sample_num = k if id < self.config.sample_schedule - 1 else num_samples - cnt
            samples = dist.rsample(sample_shape=(sample_num,))
            sample_list.append(samples)
            cnt += sample_num

        if num_samples == self.config.sample_schedule:
            sample_list = [dist.loc.unsqueeze(0) for dist in dist_list]

        return torch.concat(sample_list, dim=0)
