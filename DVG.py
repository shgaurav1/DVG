import warnings

from data.satellite import RGB_BANDS, Normalization
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import argparse
import random
from torch.utils.data import DataLoader
import json
import utils
import progressbar, pdb
import numpy as np
import gpytorch
from models.gp_models import GPRegressionLayer1
from pytorch_ssim import SSIM
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
parser.add_argument('--niter', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
#parser.add_argument('--epoch_size', type=int, default=300, help='epoch size')
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
parser.add_argument('--encoder_only', type=bool, default=False, help='whether to train only encoder')
parser.add_argument('--loss', default="mse", help='loss function to use (mse | l1 | ssim)')
parser.add_argument('--normalization', type=str, default="z", help='normalization to use (z | minmax | skip')

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
train_data, test_data = utils.load_dataset(opt, bands_to_keep=RGB_BANDS, normalization=Normalization(opt.normalization))

num_workers = opt.data_threads
if opt.dataset == "satellite":
    print("Satellite dataset only works with num_workers=1")
    num_workers = 1

train_loader = DataLoader(train_data,
                          num_workers=num_workers,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=num_workers,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

# ---------------- load the models  ----------------

print(opt)

dataset = opt.dataset
encoder_only = opt.encoder_only
lr = opt.lr
loss_type = opt.loss

if opt.model_path == '':
    if opt.test:
        raise ValueError('Must specify model path if testing')

    # ---------------- initialize the new model -------------
    from models import dcgan_64, vgg_64
    if opt.model == 'dcgan':
        encoder = dcgan_64.encoder(opt.g_dim, opt.channels)
        decoder = dcgan_64.decoder(opt.g_dim, opt.channels)
    else:
        encoder = vgg_64.encoder(opt.g_dim, opt.channels)
        decoder = vgg_64.decoder(opt.g_dim, opt.channels)

    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

    import models.lstm as lstm_models 
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
frame_predictor_optimizer = torch.optim.Adam(frame_predictor.parameters(), lr = lr)
encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr = lr)
decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr = lr)


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

ssim = SSIM().cuda()
mse = nn.MSELoss().cuda()
l1 = nn.L1Loss().cuda()

if loss_type == "ssim":
    loss_func = lambda pred, gt: 1 - ssim(pred, gt)
elif loss_type == "l1":
    loss_func = l1
elif loss_type == "mse":
    loss_func = mse
elif loss_type.startswith("ssim_mse_point"):
    if loss_type.endswith("point01"):
        mse_weight = 0.01
    elif loss_type.endswith("point1"):
        mse_weight = 0.1
    elif loss_type.endswith("point001"):
        mse_weight = 0.001
    loss_func = lambda pred, gt:  mse_weight*mse(pred, gt) + 1 - ssim(pred, gt)

else:
    raise ValueError(f"Loss type {loss_type} not recognized")

latent_loss_func = nn.MSELoss()

#loss_func.cuda()
latent_loss_func.cuda()

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
def train(x, global_step, tb_writer):
    encoder.zero_grad()
    decoder.zero_grad()

    if encoder_only:
        ae_loss = 0
        for i in range(1, opt.n_past+opt.n_future):
            h_pred = encoder(x[i])[0]
            if opt.last_frame_skip or i < opt.n_past:   
                skip = encoder(x[i-1])[1]

            decoded = decoder([h_pred, skip])
            assert x[i].shape == decoded.shape
            ae_loss += loss_func(decoded, x[i])
            torch.cuda.empty_cache()
        
        ae_loss.backward()
        
        if loss_type == "l1":
            tb_writer.add_scalar(f"Step Loss/Encoder l1", ae_loss, global_step)
        elif loss_type == "ssim":
            tb_writer.add_scalar(f"Step Loss/Encoder ssim", ae_loss, global_step)
        else:
            tb_writer.add_scalar(f"Step Loss/Encoder {loss_type}", ae_loss, global_step)

        encoder_optimizer.step()
        decoder_optimizer.step()

        return ae_loss.data.cpu().numpy()/(opt.n_past+opt.n_future)
    
    else:


        frame_predictor.zero_grad()

        # initialize the hidden state.
        frame_predictor.hidden = frame_predictor.init_hidden()

        # pdb.set_trace()
        lstm_loss = 0
        latent_loss = 0
        gp_loss = 0
        max_ll = 0
        ae_loss = 0
        for i in range(1, opt.n_past+opt.n_future):
            h = encoder(x[i-1])
            h_target = encoder(x[i])[0]

            if opt.last_frame_skip or i < opt.n_past:   
                h, skip = h
            else:
                h = h[0]
            
            h_pred = frame_predictor(h)                         # Target encoding using LSTM
            latent_loss  += latent_loss_func(h_pred,h_target) # LSTM loss - how well LSTM predicts next encoding

            gp_pred = gp_layer(h.transpose(0,1).view(90,opt.batch_size,1))#likelihood(gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)))#
            max_ll -= mll(gp_pred,h_target.transpose(0,1))      # GP Loss - how well GP predicts next encoding
            x_pred = decoder([h_pred, skip])                    # Decoded LSTM prediction

            x_target_pred = decoder([h_target, skip])           # Decoded target encoding
            ae_loss += latent_loss_func(x_target_pred,x[i])  # Encoder loss - how well the encoder encodes

            x_pred_gp = decoder([gp_pred.mean.transpose(0,1), skip])    # Decoded GP prediction
            lstm_loss += loss_func(x_pred, x[i])                          # Encoder + LSTM loss - how well the encoder+LSTM predicts the next frame
            gp_loss += latent_loss_func(x_pred_gp, x[i])             # Encoder + GP loss - how well the encoder+GP predicts the next frame
            torch.cuda.empty_cache()


        encoder_weight = 100
        alpha = 1
        beta = 0.1
        loss = encoder_weight*ae_loss + alpha*lstm_loss+ alpha*latent_loss  + beta*gp_loss + beta*max_ll.sum()  # + kld*opt.beta

        tb_writer.add_scalar("Step Loss/Encoder", ae_loss, global_step)
        tb_writer.add_scalar("Step Loss/Encoder and LSTM", lstm_loss, global_step)
        tb_writer.add_scalar("Step Loss/LSTM", latent_loss , global_step)
        tb_writer.add_scalar("Step Loss/Encoder and GP loss", gp_loss, global_step)
        tb_writer.add_scalar("Step Loss/GP loss", max_ll.sum(), global_step)
        tb_writer.add_scalar("Step Loss/Total", loss, global_step)

        loss.backward()

        frame_predictor_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()
        optimizer.step()

        return loss.data.cpu().numpy()/(opt.n_past+opt.n_future)

# --------- predicting functions ------------------------------------
def predict(x, interval_for_gp_layer: int = 10) -> List[torch.Tensor]:
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
        if i % interval_for_gp_layer == 0: 
            h_pred_from_gp_layer = likelihood(gp_layer(h.transpose(0,1).view(90,opt.batch_size,1)))
            h_pred = h_pred_from_gp_layer.rsample().transpose(0,1)
        

        # Add timestep to generated sequence
        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            decoded_h_pred = decoder([h_pred,skip]).detach()
            gen_seq.append(decoded_h_pred )

    return gen_seq

def predict_decoding(x) -> List[torch.Tensor]:
    gen_seq = [x[0]]
    for i in range(1, opt.n_eval):

        # Encode the input time step
        h = encoder(x[i-1])   
        if opt.last_frame_skip or i < opt.n_past:   
            h, skip = h     # Extract encoding and skip-connection?
        else:
            h, _ = h        # Extract encoding only
        h = h.detach()

        h_target = encoder(x[i])[0].detach()
        decoded_h_target= decoder([h_target,skip]).detach()
        gen_seq.append(decoded_h_target)

    return gen_seq

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 5
    gt_seq = [x[i] for i in range(len(x))]

    if encoder_only:
        gen_seq = [predict_decoding(x)]
    else:
        gen_seq = [predict(x) for _ in range(nsample)]


    # -------------- creating the GIFs ---------------------------
    to_plot = []
    gifs = [ [] for _ in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [gt_seq[t][i] for t in range(opt.n_eval)] 
        to_plot.append(row)

        if encoder_only:
            s_list = [0]
        else:
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

    if encoder_only:
        file_name += "_autoencoders"

    img_path = home_dir / f'imgs/{dataset}/{file_name}.png'
    img_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_of_images = utils.save_tensors_image(str(img_path), to_plot)
    print(f"Saving image to: {img_path}")

    gif_path = home_dir / f'gifs/{dataset}/{file_name}.gif'
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_gif(str(gif_path), gifs)
    print(f"Saving images as gif: {gif_path}")
    

# --------- testing loop ------------------------------------
def compute_metrics(Y_pred: List[np.ndarray], Y_true: List[np.ndarray]) -> np.ndarray:
    metric_per_timestep = {
        "mse": [],
        "l1": [],
        "ssim": [],
    }
    with torch.no_grad():
        Y_pred_tensor = torch.tensor(Y_pred)
        Y_true_tensor = torch.tensor(Y_true)
        for i in range(len(Y_pred)):
            y_true = torch.unsqueeze(Y_true_tensor[i], 0)
            y_pred = torch.unsqueeze(Y_pred_tensor[i], 0)
            
            assert y_true.shape == (1, 3, opt.image_width, opt.image_width)
            assert y_true.shape == y_pred.shape

            mse_for_timestep = mse(y_true, y_pred).item()
            metric_per_timestep["mse"].append(mse_for_timestep)
            assert np.isclose(mse_for_timestep, ((Y_true[i] - Y_pred[i])**2).mean()), f"{mse_for_timestep} and {((Y_true[i] - Y_pred[i])**2).mean()}"
            
            l1_for_timestep = l1(y_true, y_pred).item()
            metric_per_timestep["l1"].append(l1_for_timestep)
            assert np.isclose(l1_for_timestep, (np.abs(Y_true[i] - Y_pred[i])).mean()), f"{l1_for_timestep} and {np.abs(Y_true[i] - Y_pred[i]).mean()}"

            metric_per_timestep["ssim"].append(ssim(y_true, y_pred).item())

    metric_for_example = {}
    for metric_type, metric_values in metric_per_timestep.items():
        assert len(metric_values) == len(Y_pred), f"{metric_type} does not have enough values"
        metric_for_example[metric_type] = np.mean(metric_values)

    return metric_for_example

if opt.test:
    frame_predictor.eval()
    gp_layer.eval()
    likelihood.eval()

    # Go through all test data
    metrics_for_each_example = []

    for sequence in tqdm(test_loader):
        x = utils.normalize_data(opt.dataset, dtype, sequence)
        if encoder_only:
            nsample = 1
            gen_seq = [predict_decoding(x)]
        else:
            nsample = 5
            gen_seq = [predict(x) for _ in range(nsample)]

        gt_seq = [x[i] for i in range(len(x))]

        assert len(gen_seq[0]) == len(gt_seq)
        assert gen_seq[0][0].shape == gt_seq[0].shape
        
        for i in tqdm(range(opt.batch_size), leave=False):
            # Finds best sequence (lowest loss)
            min_mse = None
            for s in range(nsample):
                Y_pred = []
                Y_true = []

                for t in range(opt.n_past, opt.n_eval):
                    y_pred_timestep = gen_seq[s][t][i].data.cpu().numpy()
                    y_true_timestep = gt_seq[t][i].data.cpu().numpy()

                    Y_pred.append(train_data._unnormalize(y_pred_timestep))
                    Y_true.append(train_data._unnormalize(y_true_timestep))

                metrics_for_example = compute_metrics(Y_true, Y_pred)
                
                if min_mse is None or  min_mse > metrics_for_example["mse"]:
                    min_mse = metrics_for_example["mse"]
                    lowest_metrics_for_example = metrics_for_example
            
            metrics_for_each_example.append(lowest_metrics_for_example)

    metrics_for_test_set = {
        metric_type: np.array([m[metric_type] for m in metrics_for_each_example]).mean(axis=0)
        for metric_type in ["mse", "l1", "ssim"]
    }

    for metric_type, metric in metrics_for_test_set.items():
        print(f"{metric_type}: {metric}")

    # Check if metrics json file exists
    metrics_json_path = home_dir / "Results/metrics.json"
    metrics_json_already_exists = metrics_json_path.exists()
    if not metrics_json_already_exists:
        metrics_json_path.touch()

    # Read metrics json file
    with open(metrics_json_path, "r") as f:
        if metrics_json_already_exists:
            metrics_json = json.load(f)
        else:
            metrics_json = {}

        if dataset not in metrics_json:
            metrics_json[dataset] = {}

        metrics_json[dataset][opt.model_path] = metrics_for_test_set

    # Write new metrics to metrics json file
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=4)



# --------- training loop ------------------------------------
else:
    writer = SummaryWriter(log_dir=f"runs/{dataset}/{opt.run_name}" if opt.run_name else None)
    epoch_size = len(train_loader)
    with gpytorch.settings.max_cg_iterations(45):
        for epoch in range(opt.niter):
            gp_layer.train()
            likelihood.train()
            frame_predictor.train()
            encoder.train()
            decoder.train()
            scheduler.step()

            epoch_loss = 0
            progress = progressbar.ProgressBar(epoch_size).start()
            
            for i in range(epoch_size):
                
                progress.update(i+1)
                x = next(training_batch_generator)
                global_step=epoch * epoch_size + i
                batch_loss = train(x, global_step, writer) 
                epoch_loss += batch_loss

            progress.finish()
            utils.clear_progressbar()

            mean_epoch_loss = epoch_loss/epoch_size
            if encoder_only:
                writer.add_scalar(f"Epoch Loss/Encoder {loss_type}", mean_epoch_loss, epoch)
            else:
                writer.add_scalar("Epoch Loss/Total", mean_epoch_loss, epoch)
            
            print('[%02d] mse loss: %.5f (%d) %.5f' % (epoch, mean_epoch_loss, epoch*epoch_size*opt.batch_size, batch_loss))
            
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
                
                if encoder_only:
                    model_name += '_autoencoder'

                model_path = home_dir / f'model_dump/{dataset}/{model_name}.pth'
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
