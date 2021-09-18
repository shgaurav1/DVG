import warnings

from data.satellite import RGB_BANDS
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import argparse
import random
from torch.utils.data import DataLoader
import utils
import progressbar, pdb
import numpy as np
import gpytorch
from models.gp_models import GPRegressionLayer1
from typing import List
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs_final_version_gp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=601, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=300, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=7, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=12, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=90, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--model_path', type=str, default='', help='model pth file with which to resume training')
parser.add_argument('--home_dir', type=str, default='.', help='Where to save gifs, models, etc')
parser.add_argument('--test', type=bool, default=False, help="whether to train or test the model")
parser.add_argument('--run_name', type=str, default='', help='name of run')

opt = parser.parse_args()
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

home_dir = Path(opt.home_dir)

torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt, bands_to_keep=RGB_BANDS)

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

dataset = opt.dataset

if opt.model_path == '':
    if opt.test:
        raise ValueError('Must specify model path if testing')

    # ---------------- initialize the new model -------------

    import models.dcgan_64 as model
    import models.lstm as lstm_models
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

    frame_predictor = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)

else:
    # ---------------- load the trained model -------------

    from models.gp_models import GPRegressionLayer1

    model = torch.load(opt.model_path)
    encoder = model['encoder']
    decoder = model['decoder']
    frame_predictor = model['frame_predictor']

# ---------------- models tranferred to GPU ----------------
encoder.cuda()
decoder.cuda()
frame_predictor.cuda()


# ---------------- optimizers ----------------
frame_predictor_optimizer = torch.optim.Adam(frame_predictor.parameters(), lr = 0.002)
encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr = 0.002)
decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr = 0.002)


# ---------------- GP initialization ----------------------
from models.gp_models import GPRegressionLayer1
gp_layer = GPRegressionLayer1().cuda()#inputs
likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=opt.g_dim).cuda()

if opt.model_path:
    likelihood.load_state_dict(model['likelihood'])
    gp_layer.load_state_dict(model['gp_layer'])


# ---------------- GP optimizer initialization ----------------------
optimizer = torch.optim.Adam([{'params': gp_layer.parameters()}, {'params': likelihood.parameters()},], lr=0.002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)

# Our loss for GP object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, gp_layer, num_data=opt.batch_size, combine_terms=True)

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()#losses.TotalLoss()#
mse_latent_criterion = nn.MSELoss()

mse_criterion.cuda()
mse_latent_criterion.cuda()

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt.dataset, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

# training_batch_generator = torch.nn.DataParallel(training_batch_generator)

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt.dataset, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# # --------- training funtions ------------------------------------
def train(x,epoch, tb_writer):
    encoder.zero_grad()
    decoder.zero_grad()
    frame_predictor.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()

    # pdb.set_trace()
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
        
        h_pred = frame_predictor(h)                         # Target encoding using LSTM
        mse_latent += mse_latent_criterion(h_pred,h_target) # LSTM loss - how well LSTM predicts next encoding

        gp_pred = gp_layer(h.transpose(0,1).view(90,opt.batch_size,1))#likelihood(gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)))#
        max_ll -= mll(gp_pred,h_target.transpose(0,1))      # GP Loss - how well GP predicts next encoding
        x_pred = decoder([h_pred, skip])                    # Decoded LSTM prediction

        x_target_pred = decoder([h_target, skip])           # Decoded target encoding
        ae_mse += mse_latent_criterion(x_target_pred,x[i])  # Encoder loss - how well the encoder encodes

        x_pred_gp = decoder([gp_pred.mean.transpose(0,1), skip])    # Decoded GP prediction
        mse += mse_criterion(x_pred, x[i])                          # Encoder + LSTM loss - how well the encoder+LSTM predicts the next frame
        mse_gp += mse_latent_criterion(x_pred_gp, x[i])             # Encoder + GP loss - how well the encoder+GP predicts the next frame
        torch.cuda.empty_cache()


    encoder_weight = 100
    alpha = 1
    beta = 0.1
    loss = encoder_weight*ae_mse + alpha*mse+ alpha*mse_latent + beta*mse_gp + beta*max_ll.sum()  # + kld*opt.beta

    tb_writer.add_scalar("Loss/Encoder - AE MSE", ae_mse, epoch)
    tb_writer.add_scalar("Loss/Encoder and LSTM - MSE", mse, epoch)
    tb_writer.add_scalar("Loss/LSTM - MSE LATENT", mse_latent, epoch)
    tb_writer.add_scalar("Loss/Encoder and GP loss - MSE GP", mse_gp, epoch)
    tb_writer.add_scalar("Loss/GP loss - Max ll", max_ll.sum(), epoch)
    tb_writer.add_scalar("Loss/Total", loss, epoch)

    loss.backward()

    frame_predictor_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    optimizer.step()

    return mse_latent.data.cpu().numpy()/(opt.n_past+opt.n_future)

# --------- predicting functions ------------------------------------
def predict(x, index_to_use_gp_layer: int = opt.n_past) -> List[torch.Tensor]:
    frame_predictor.hidden = frame_predictor.init_hidden() # Initialize LSTM hidden state
    gen_seq = [x[0]]
    for i in range(1, opt.n_eval):

        # Encode the input time step
        h = encoder(gen_seq[-1])   
        if opt.last_frame_skip or i < opt.n_past:   
            h, skip = h     # Extract encoding and skip-connection?
        else:
            h, _ = h        # Extract encoding only
        h = h.detach()

        # Predict the next time step using the LSTM (needs to be called each time to update the hidden state)
        h_pred = frame_predictor(h).detach()
        
        # Use the GP layer to predict the next time step
        # Previously i%10 was used for longer sequences
        if i == index_to_use_gp_layer: 
            h_pred_from_gp_layer = likelihood(gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)))
            h_pred = h_pred_from_gp_layer.rsample().transpose(0,1)
        

        # Add timestep to generated sequence
        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            decoded_h_pred = decoder([h_pred,skip]).detach()
            gen_seq.append(decoded_h_pred )

    return gen_seq

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 5
    gt_seq = [x[i] for i in range(len(x))]
    gen_seq = [predict(x) for _ in range(nsample)]


    # -------------- creating the GIFs ---------------------------
    to_plot = []
    gifs = [ [] for _ in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [gt_seq[t][i] for t in range(opt.n_eval)] 
        to_plot.append(row)

        # Finds best sequence (lowest loss)
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
        
        # Add to images
        for s in s_list:
            row = [gen_seq[s][t][i] for t in range(opt.n_eval)]
            to_plot.append(row)
        
        # Add to gifs
        for t in range(opt.n_eval):
            row = [gt_seq[t][i]]
            row += [gen_seq[s][t][i] for s in s_list]
            gifs[t].append(row)
    
    if opt.dataset == 'satellite':
        for i, row in enumerate(to_plot):
            for j, img in enumerate(row):
                img_np = img.cpu().numpy()
                img_viewable = train_data.for_viewing(img_np)
                img_tensor = torch.from_numpy(img_viewable)
                to_plot[i][j] = img_tensor
        
        for i, gif in enumerate(gifs):
            for j, row in enumerate(gif):
                for k, img in enumerate(row):
                    img_np = img.cpu().numpy()
                    img_viewable = train_data.for_viewing(img_np)
                    img_tensor = torch.from_numpy(img_viewable)
                    gifs[i][j][k] = img_tensor


    if opt.run_name:
        file_name = opt.run_name
    else:
        file_name = f"end2end_gp_ctrl_sample_{epoch}"

    img_path = home_dir / f'imgs/{dataset}/{file_name}.png'
    img_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_of_images = utils.save_tensors_image(str(img_path), to_plot)
    print(f"Saving image to: {img_path}")

    gif_path = home_dir / f'gifs/{dataset}/{file_name}.gif'
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_gif(str(gif_path), gifs)
    print(f"Saving images as gif: {gif_path}")
    

# --------- testing loop ------------------------------------
def compute_mse(Y_pred: List[np.ndarray], Y_true: List[np.ndarray]) -> np.ndarray:
    return ((np.array(Y_true) - np.array(Y_pred))**2).mean(axis=(0,-2,-1))

if opt.test:
    frame_predictor.eval()
    gp_layer.eval()
    likelihood.eval()

    # Go through all test data
    normalized_mse_list = []
    unnormalized_mse_list = []
    for sequence in tqdm(test_loader):
        x = utils.normalize_data(opt.dataset, dtype, sequence)
        nsample = 5
        gen_seq = [predict(x) for _ in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]
        
        for i in tqdm(range(opt.batch_size), leave=False):
            # Finds best sequence (lowest loss)
            min_mse = None
            for s in range(nsample):
                Y_pred = []
                Y_true = []
                Y_pred_unnormed = []
                Y_true_unnormed = []

                for t in range(opt.n_past, opt.n_eval):
                    y_pred_timestep = gen_seq[s][t][i].data.cpu().numpy()
                    y_true_timestep = gt_seq[t][i].data.cpu().numpy()

                    Y_pred.append(y_pred_timestep)
                    Y_true.append(y_true_timestep)
                    Y_pred_unnormed.append(train_data.unnormalize(y_pred_timestep))
                    Y_true_unnormed.append(train_data.unnormalize(y_true_timestep))

                mse_per_band = compute_mse(Y_true, Y_pred)
                
                if min_mse is None or  min_mse > mse_per_band.sum():
                    min_mse = mse_per_band.sum()
                    lowest_normed_mse_pixel = mse_per_band
                    lowest_unnormed_mse_pixel =  compute_mse(Y_true_unnormed, Y_pred_unnormed)
            
            normalized_mse_list.append(lowest_normed_mse_pixel)
            unnormalized_mse_list.append(lowest_unnormed_mse_pixel)
    mean_mse_per_bands = np.array(normalized_mse_list).mean(axis=0)
    mean_unnormalized_mse_per_bands = np.array(unnormalized_mse_list).mean(axis=0)
    print(f"Normalized_mse {mean_mse_per_bands}")
    print(f"Unnormalized_mse {mean_unnormalized_mse_per_bands}")


# --------- training loop ------------------------------------
else:
    writer = SummaryWriter(log_dir=f"runs/{dataset}/{opt.run_name}" if opt.run_name else None)
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
            progress = progressbar.ProgressBar(opt.epoch_size).start()
            
            for i in range(opt.epoch_size):
                
                progress.update(i+1)
                x = next(training_batch_generator)
                mse_ctrl = train(x, epoch, writer) 
                epoch_mse += mse_ctrl 

            progress.finish()
            utils.clear_progressbar()

            print('[%02d] mse loss: %.5f (%d) %.5f' % (epoch, epoch_mse/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size, mse_ctrl))
            if epoch % 4 == 0:
            
                # plot some stuff
                frame_predictor.eval()
                gp_layer.eval()
                likelihood.eval()

                test_x = next(testing_batch_generator)
                plot(test_x, epoch)

                # save the model
                if opt.run_name:
                    model_name = opt.run_name
                else:
                    model_name = f'e2e_{dataset}_model'
                model_path = home_dir / f'model_dump/{model_name}.pth'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'likelihood': likelihood.state_dict(),
                    'gp_layer': gp_layer.state_dict(),
                    'gp_layer_optimizer': optimizer.state_dict(),
                    'opt': opt},
                    str(model_path))

            if epoch % 10 == 0:
                print('log dir: %s' % opt.log_dir)
        writer.flush()
