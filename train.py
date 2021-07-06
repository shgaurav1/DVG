import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar,pdb
import numpy as np
import gpytorch
import losses
from matplotlib import pyplot as plt
from models.gp_models import GPRegressionLayer1

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--output_path', default='.', help='output directory path')
parser.add_argument('--data_root', default='path/to/data/', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=601, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=300, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--ft', type = bool, default=True, help='finetune flag')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=15, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=90, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')


opt = parser.parse_args()
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

# ---------------- load the models  ----------------

print(opt)

# ---------------- initialize the new model -------------

import models.dcgan_64 as model
import models.lstm as lstm_models
encoder = model.encoder(opt.g_dim, opt.channels)
decoder = model.decoder(opt.g_dim, opt.channels)
encoder.apply(utils.init_weights)
decoder.apply(utils.init_weights)

frame_predictor = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
frame_predictor.apply(utils.init_weights)


# ---------------- load the trained model -------------

# ---------------- models tranferred to GPU ----------------
encoder.cuda()
decoder.cuda()
frame_predictor.cuda()


# ---------------- optimizers ----------------
frame_predictor_optimizer = torch.optim.Adam(frame_predictor.parameters(), lr = 0.002)
encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr = 0.002)
decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr = 0.002)


# ---------------- GP initialization ----------------------
gp_layer = GPRegressionLayer1(opt.g_dim).cuda()#inputs
likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=opt.g_dim).cuda()

# ---------------- GP optimizer initialization ----------------------
optimizer = torch.optim.Adam([{'params': gp_layer.parameters()}, {'params': likelihood.parameters()},], lr=0.002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)




# Our loss for GP object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, gp_layer, num_data=opt.batch_size, combine_terms=True)



# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
mse_latent_criterion = nn.MSELoss()

mse_criterion.cuda()
mse_latent_criterion.cuda()


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()






# # --------- training funtions ------------------------------------
# -----------finetune GP dynamic network predictor
def train_GP_Frame_predictor(x):
    optimizer.zero_grad()
    with gpytorch.settings.use_toeplitz(False):        
        # initialize the hidden state.
        frame_predictor.hidden = frame_predictor.init_hidden()

        mse_latent = 0
        max_ll = 0
        for i in range(1, opt.n_past+opt.n_future):
            h = encoder(x[i-1])
            h_target = encoder(x[i])[0].detach()

            if opt.last_frame_skip or i < opt.n_past:   
                h, skip = h
            else:
                h = h[0]
                h.detach()

            h_pred = gp_layer(h.transpose(0,1).view(90,50,1))
            max_ll -= mll(h_pred,h_target.transpose(0,1))
            torch.cuda.empty_cache()


        loss = max_ll.sum() 
    loss.backward()
    optimizer.step()
    return max_ll.sum().data.cpu().numpy()/(opt.n_past+opt.n_future)
  
#-----finetune LSTM dynamic network------------------------------------------
def train_frame_predictor(x):
    frame_predictor.zero_grad()
    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()

    mse_latent = 0
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:   
            h, skip = h
        else:
            h = h[0]
        h_pred = frame_predictor(h)

        mse_latent += mse_latent_criterion(h_pred,h_target)
        torch.cuda.empty_cache()

    loss = mse_latent 
    loss.backward()

    frame_predictor_optimizer.step()

    return mse_latent.data.cpu().numpy()/(opt.n_past+opt.n_future)

def train_model(x):
    encoder.zero_grad()
    decoder.zero_grad()
    frame_predictor.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()

    mse = 0
    mse_latent = 0
    mse_gp = 0
    max_ll = 0
    ae_mse = 0
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]

        if opt.last_frame_skip or i < opt.n_past:   
            h, skip = h
        else:
            h = h[0]
        
        h_pred = frame_predictor(h)
        mse_latent += mse_latent_criterion(h_pred,h_target)

        gp_pred = gp_layer(h.transpose(0,1).view(90,opt.batch_size,1))
        max_ll -= mll(gp_pred,h_target.transpose(0,1))
        x_pred = decoder([h_pred, skip])

        x_target_pred = decoder([h_target, skip])
        ae_mse += mse_latent_criterion(x_target_pred,x[i])

        x_pred_gp = decoder([gp_pred.mean.transpose(0,1), skip])
        mse += mse_criterion(x_pred, x[i])
        mse_gp += mse_latent_criterion(x_pred_gp, x[i])
        torch.cuda.empty_cache()



    loss = 1000*ae_mse + 0.001*mse+ 0.01*mse_latent +0.001*mse_gp + 0.0001*max_ll.sum() 
    loss.backward()

    frame_predictor_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    optimizer.step()


    return mse_latent.data.cpu().numpy()/(opt.n_past+opt.n_future),mse_latent.data.cpu().numpy()/(opt.n_past+opt.n_future)

def finetune_temporal_encoders(x):
    temporal_lstm_loss = train_frame_predictor(x)
    temporal_gp_loss = train_GP_Frame_predictor(x)
    return temporal_gp_loss+temporal_lstm_loss

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 5
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]
    
    
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:   
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0].detach()
                frame_predictor(h)
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                h_pred = frame_predictor(h).detach()
                if i == 10:
                    print(str(s) + " started")
                    final_hpred = likelihood(gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)))
                    x_in = decoder([final_hpred.rsample().transpose(0,1), skip]).detach()
                    gen_seq[s].append(x_in)
                    print(str(s) + " completed")
                else:
                    x_in = decoder([h_pred,skip]).detach()
                    gen_seq[s].append(x_in)

    # -------------- creating the GIFs ---------------------------
    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx, 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/sample_%d.png' % (opt.output_path, epoch) 
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/sample_%d.gif' % (opt.output_path, epoch) #add path
    utils.save_gif(fname, gifs)



# --------- training loop ------------------------------------
with gpytorch.settings.max_cg_iterations(45):
    for epoch in range(opt.niter):
        gp_layer.train()
        likelihood.train()
        frame_predictor.train()
        encoder.train()
        decoder.train()
        scheduler.step()

        epoch_mse = 0
        epoch_ls = 0
        temp_loss = 0
        progress = progressbar.ProgressBar(opt.epoch_size).start()
        
        for i in range(opt.epoch_size):
            
            progress.update(i+1)
            x,y = next(training_batch_generator)
            mse_ctrl, indices = train_model(x) 
            
            if opt.ft: 
                temp_loss = finetune_temporal_encoders(x)
            
            epoch_mse += mse_ctrl + temp_loss

        progress.finish()
        utils.clear_progressbar()

        print('[%02d] mse loss: %.5f (%d) %.5f' % (epoch, epoch_mse/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size, indices))
        if epoch % 4 == 0:
        
        # plot some stuff    
            frame_predictor.eval()
            gp_layer.eval()
            likelihood.eval()

            test_x,targets = next(testing_batch_generator)
            plot(test_x, epoch)

        # save the model
            torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'likelihood': likelihood.state_dict(),
                'gp_layer': gp_layer.state_dict(),
                'gp_layer_optimizer': optimizer.state_dict(),
                'opt': opt},
                '%s/model.pth' % (opt.output_path)) #add path

        if epoch % 10 == 0:
            print('log dir: %s' % opt.log_dir)
 
