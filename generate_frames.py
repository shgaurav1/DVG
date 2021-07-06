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
from gp_models import GPRegressionLayer1
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs_gp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')

parser.add_argument('--data_root', default='./data/kth', help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--gp_trigger_flag', default=False, type=bool, help = 'manual_trigger or GP_trigger')
parser.add_argument('--dataset', default='kth', help='dataset to test with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=60, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=90, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')


opt = parser.parse_args()

loaded_model = torch.load('%s/%s.pth'%(opt.model_dir,opt.dataset))
opt = loaded_model['opt']
os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
opt.n_eval = 105
opt.n_future = 100 
opt.batch_size = 50

# -------------------------------------------------------------
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

#--------------------------------------------------------------
encoder = loaded_model['encoder']
decoder = loaded_model['decoder']
frame_predictor = loaded_model['frame_predictor']
#-------- transfer model to gpus-------------------------------
encoder.cuda()
decoder.cuda()
frame_predictor.cuda()

likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=opt.g_dim).cuda()
gp_layer = GPRegressionLayer1(opt.g_dim).cuda()#inputs

#-------- Load GP models---------------------------------------
likelihood.load_state_dict(loaded_model['likelihood'])
gp_layer.load_state_dict(loaded_model['gp_layer'])


encoder.eval()
decoder.eval()
frame_predictor.eval()
gp_layer.eval()
likelihood.eval()

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
mse_latent_criterion = nn.MSELoss()

mse_criterion.cuda()
mse_latent_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()


def make_gifs(x, idx, name, loaded_model):
    

    with torch.no_grad():
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior_gen = []
        posterior_gen.append(x[0])
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
                posterior_gen.append(x_in)
            else:
                h_pred = frame_predictor(h).detach()
                print(str(i) + " started")
                final_hpred = likelihood(gp_layer(h_pred.transpose(0,1).view(90,50,1)))
                x_in = decoder([final_hpred.mean.transpose(0,1), skip]).detach()
                print(str(i) + " completed")
                posterior_gen.append(x_in)
                
      

        nsample = 100#opt.nsample
        ssim = np.zeros((opt.batch_size, nsample, opt.n_eval - opt.n_past))
        psnr = np.zeros((opt.batch_size, nsample, opt.n_eval - opt.n_past))
        progress = progressbar.ProgressBar(nsample).start()
        all_gen = []
        for s in range(nsample):
            progress.update(s+1)
            gen_seq = []
            gt_seq = []
            frame_predictor.hidden = frame_predictor.init_hidden()
            x_in = x[0]
            all_gen.append([])
            all_gen[s].append(x_in)

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
                    all_gen[s].append(x_in)
                else:
                    h_pred = frame_predictor(h).detach()
                    print(str(s) + " started")
                    if i%15 == 0:
                        print(str(s) + " started")
                        final_hpred = likelihood(gp_layer(h.transpose(0,1).view(90,50,1)))
                        x_in = decoder([final_hpred.rsample().transpose(0,1), skip]).detach()
                        print(str(s) + " completed")
                    else:
                        x_in = decoder([h_pred,skip]).detach()
                    gen_seq.append(x_in.data.cpu().numpy())
                    gt_seq.append(x[i].data.cpu().numpy())
                    all_gen[s].append(x_in)
            _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)
            

        progress.finish()
        utils.clear_progressbar()

        ###### ssim ######
        for i in range(opt.batch_size):
            gifs = [ [] for t in range(opt.n_eval) ]
            text = [ [] for t in range(opt.n_eval) ]
            mean_ssim = np.mean(ssim[i], 1)
            ordered = np.argsort(mean_ssim)
            rand_sidx = [np.random.randint(nsample) for s in range(3)]
            for t in range(opt.n_eval):
                # gt 
                gifs[t].append(add_border(x[t][i], 'green'))
                text[t].append('Ground\ntsruth')
                #posterior 
                if t < opt.n_past:
                    color = 'green'
                else:
                    color = 'red'
                gifs[t].append(add_border(posterior_gen[t][i], color))
                text[t].append('Approx.\nposterior')
                # best 
                if t < opt.n_past:
                    color = 'green'
                else:
                    color = 'red'
                sidx = ordered[-1]
                gifs[t].append(add_border(all_gen[sidx][t][i], color))
                text[t].append('Best SSIM')
                # random 3
                for s in range(len(rand_sidx)):
                    gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                    text[t].append('Random\nsample %d' % (s+1))

            fname = '%s/sample_%s_%d.gif' % (opt.log_dir, name, idx+i) 
            utils.save_gif_with_text(fname, gifs, text)
            return ssim, psnr


def generation(x_in,skip):
    h = encoder(x_in)[0].detach()
    h_pred = frame_predictor(h).detach()
    x_out = decoder([h_pred,skip]).detach()
    return x_out


def var_value(x_in,context_array):
    h = encoder(x_in)[0].detach()
    final_pred = likelihood(gp_layer(h.transpose(0,1).view(90,50,1)))
    value = np.linalg.norm(final_pred.variance.cpu().detach().numpy().transpose(),axis = 1)[3]
    context_array = np.concatenate([context_array[1:],[value]])
    return value,context_array


def plot_rec(x, index,frames_generated,depth):

    to_plot = []
    nrow = min(opt.batch_size, 1)
    for i in range(nrow):
        row = []
        for t in range(0,len(x),3):
            row.append(x[t][index])
        to_plot.append(row)
    fname = 'recursive_generation/%d/heuristic_gp_trigger_%d_%d.png' % (index,depth, frames_generated) 
    utils.save_tensors_image(fname, to_plot)



def GPtrigger_gen(x):

    for index in range(opt.batch_size):
        frame_predictor.hidden = frame_predictor.init_hidden()
        context_array = []
        depth = 1
        x_in = x[0]
        gen_seq = []
        frames_generated = 0
        triggers =[]
        values = []
        lstm_values = []
        det_seq = []
        if not os.path.isdir('recursive_generation'):
            os.makedirs('recursive_generation')
        if not os.path.isdir('./recursive_generation/%d'%index):    
            os.makedirs('./recursive_generation/%d'%index)
        for i in range(12):
            h = encoder(x_in)
            if i < 5:
                h,skip = h
            else:
                h, _ = h
            h.detach()
            final_pred = likelihood(gp_layer(h.transpose(0,1).view(90,50,1)))

            value = np.linalg.norm(final_pred.variance.cpu().detach().numpy().transpose(),axis = 1)[index]
            context_array.append(value)
            x_out = generation(x_in,skip)
            values.append(value)
            gen_seq.append(x_out)
            x_in = x_out

        # Creating a threshold using first 12 frames (This includes context frames+ generated frames)
        context_array = np.array(context_array)

        for i in range(12,105):
            # Updating the threshold
            value,context_array = var_value(x_in,context_array)
            threshold = np.mean(context_array) + (2+0.01*depth)* np.std(context_array)
            if value > threshold:
                h = encoder(x_in)[0].detach()
                final_pred = likelihood(gp_layer(h.transpose(0,1).view(90,50,1)))
                x_out = decoder([final_pred.rsample().transpose(0,1),skip]).detach()

            else:
                x_out = generation(x_in,skip)
            values.append(value)
            gen_seq.append(x_out)
            x_in = x_out

        plot_rec(gen_seq,index,frames_generated,depth)
        




def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px




for i in range(5):
    test_x,targets = next(testing_batch_generator)
    if opt.gp_trigger_flag:
        GPtrigger_gen(test_x)
    else:
        ssim_gp_lstm,psnr_gp_lstm = make_gifs(test_x, i, 'lstm',loaded_model)
