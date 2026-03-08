# basic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import time
from scipy.integrate import solve_ivp
from itertools import product
import pandas as pd
import matplotlib.gridspec as gridspec

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import wasserstein_distance

# custom
from _config import model_lorenz
from _wgan import G_model, D_model
from _hyperpinn import TrunkNetwork, HyperNetwork, HyperNetwork_lorenz, compute_total_params
from _deeponet import DeepONet

# save path
path = './save'
if not os.path.isdir(path):
    os.mkdir(path)
for pth in [path+'/figures', path+'/hyperPINN', path+'/DeepONet', path+'/WGAN', path+'/GP', path+'/synthetic_data', path+'/real_data']:
    if not os.path.isdir(pth):
        os.mkdir(pth)

class Trainer:
    def __init__(self, name, prms, device):
        self.name = name
        self.prms_DE, self.prms_PINN, self.prms_GAN = prms
        self.device = device
        # DE
        self.outputs = self.prms_DE['outputs']
        self.param_names = self.prms_DE['param_names']
        self.num_p = self.prms_DE['num_p']
        self.num_eq = self.prms_DE['num_eq']
        self.eq_list = range(self.num_eq)
        self.tmin, self.tmax = self.prms_DE['t_range']
        self.p_range = self.prms_DE['p_range']
        self.Y_init = self.prms_DE['init']
        self.p_comb = list(product(*self.p_range))
        # hyperPINN
        self.width_hyp = self.prms_PINN['width_hyp']
        self.depth_hyp = self.prms_PINN['depth_hyp']
        self.width_tru = self.prms_PINN['width_tru']
        self.num_chunks_in = self.prms_PINN['num_chunks_in']
        self.act = self.prms_PINN['act']
        self.lr = self.prms_PINN['lr']
        self.Nt = self.prms_PINN['Nt']
        self.Np = self.prms_PINN['Np']
        self.batch_size = self.prms_PINN['batch_size']
        self.init_noise_scale = self.prms_PINN['init_noise_scale']
        self.beta = self.prms_PINN['beta']
        self.tol = self.prms_PINN['tol']
        self.layers_config = [(1,self.width_tru)]+[(self.width_tru, self.width_tru)]*3+[(self.width_tru, 1)]
        self.total_params = compute_total_params(self.layers_config)
        self.sc = self.prms_PINN['sc']
        self.deeponet = self.prms_PINN['deeponet']
        self.pscale = torch.FloatTensor(self.p_range).T
        if self.deeponet:
            path_yscale = path+'/DeepONet/'+self.name+'_yscale'
        else:
            path_yscale = path+'/hyperPINN/'+self.name+'_yscale'
        if os.path.isfile(path_yscale):
            self.yscale = torch.load(path_yscale).to(torch.float32).detach().cpu() # if there's a savefile for yscale  
        else:
            self.yscale = None
        # WGAN
        self.num_bins = self.prms_GAN['num_bins']
        self.plow, self.phigh = self.prms_GAN['plow'], self.prms_GAN['phigh']
        self.modes = self.prms_GAN['modes']
        self.num_modes = len(self.modes)
        self.num_noised = self.prms_GAN['num_noised']
        self.num_data = self.num_modes*self.num_noised*self.num_bins # number of all datapoints (t,x)
        self.num_gen = self.prms_GAN['num_gen_ratio']*self.num_data
        self.num_cut = self.prms_GAN['num_cut'] 
        self.noise_dim = self.prms_GAN['noise_dim']
        self.data_noise_scale = self.prms_GAN['data_noise_scale']
        self.init_noise_scale_wgan = self.prms_GAN['init_noise_scale_wgan']
        self.width = self.prms_GAN['width']
        self.depth = self.prms_GAN['depth']
        self.lr_G = self.prms_GAN['lr_G'] 
        self.lr_D = self.prms_GAN['lr_D'] 
        self.num_gen_plot = self.prms_GAN['num_gen_plot'] 
        self.num_gen_traj = self.prms_GAN['num_gen_traj']
        self.ratio_G = self.prms_GAN['ratio_G']
        self.eps = 1e-16
        self.lamb = 10
        if self.sc:
            self.center = torch.FloatTensor([0.5]*self.num_p).view(1,-1).to(self.device)
        else:
            self.center = torch.FloatTensor([sum(self.p_range[i])/2 for i in range(self.num_p)]).view(1,-1).to(self.device)
        # time settings for WGAN
        self.t_grids = torch.linspace(self.tmin, self.tmax, self.num_bins).view(-1,1)
        self.t_numpy = self.t_grids.numpy()
        self.times_all = torch.tensor(self.t_numpy).to(torch.float32).view(-1,1).to(self.device)
        self.onehot_all = F.one_hot(torch.arange(0, self.num_bins), num_classes=self.num_bins).to(torch.float32).to(self.device)
        # filename
        if self.deeponet:
            self.filename = self.name+str(self.num_modes)+'_cut'+str(self.num_cut)+'_deeponet'
        else:
            self.filename = self.name+str(self.num_modes)+'_cut'+str(self.num_cut)
        if self.data_noise_scale>0.0:
            self.filename = self.filename+'_noise='+str(round(self.data_noise_scale,3))
        if self.init_noise_scale_wgan>0.0:
            self.filename = self.filename+'_init='+str(round(self.init_noise_scale_wgan,3))
    
    ############################################## for generating dataset ################################################
    def scale(self, data, vmin=0, vmax=1, backward=False):
        if backward:
            return (vmax-vmin)*data+vmin
        else:
            return (data-vmin)/(vmax-vmin)   
    def scale_y(self, data, backward=False):
        return self.scale(data,vmin=self.yscale[0,:],vmax=self.yscale[1,:], backward=backward)
    def scale_p(self, data, backward=False):
        return self.scale(data,vmin=self.pscale[0,:],vmax=self.pscale[1,:], backward=backward)
    
    # get dataset for hyperPINN
    def get_data_pinn(self, N_p=0, save_sc=False):
        for i in range(self.num_p):
            self.p_range[i][0] *= 0.9 
            self.p_range[i][1] *= 1.1
        if N_p==0:
            N_p = self.Np
        # get time grids
        self.times = torch.linspace(self.tmin,self.tmax,self.Nt).view(1,-1)
        # get unifrom random collocation points of parameter space
        ps = torch.cat([self.p_range[i][0]+(self.p_range[i][1]-self.p_range[i][0])*torch.rand(N_p,1) for i in range(self.num_p)],-1)
        inits = torch.cat([self.Y_init[i] *(1 + self.init_noise_scale*torch.rand(N_p, 1)) for i in self.eq_list],-1)
        sensor = torch.cat([ps.repeat(1,self.Nt).reshape(-1,self.num_p), inits.repeat(1,self.Nt).reshape(-1,self.num_eq)],-1)
        # get true solutions from analytic solution or solver
        y = []
        for i in range(N_p):
            y.append(self.solution(torch.cat([self.times.reshape(-1,1), ps[i,:].repeat(1,self.Nt).reshape(-1,self.num_p)],-1), inits=inits[i:i+1,:].repeat(1,self.Nt).reshape(-1,self.num_eq))) # we need noised initial
        y = torch.concat(y,0)
        # scale : normalize (p_range to [0,1]) and ([y_min,y_max] to [0,1])
        if self.sc and save_sc:
            self.yscale = torch.cat([torch.min(y,0).values.view(1,-1), torch.max(y,0).values.view(1,-1)],0)
            if self.deeponet:
                torch.save(self.yscale, path+'/DeepONet/'+self.name+'_yscale') # save yscale
            else:
                torch.save(self.yscale, path+'/hyperPINN/'+self.name+'_yscale') # save yscale
        if self.sc:
            return torch.cat([self.scale_p(sensor[:,:self.num_p]), self.scale_y(sensor[:,self.num_p:])],-1), self.scale_y(y)
        else:
            return sensor, y
    
    # get train/test split (9:1)
    def get_data_split(self, data, ratio=0.1):
        num_entire = len(data)
        num_test = int(num_entire*ratio)
        num_train = num_entire-num_test
        return data[:num_train], data[num_train:]

    # get dataset for WGAN
    def get_data_gan(self, real_data=None, print_results=True, seed=1234):
        if real_data is None:
            # generate synthetic data
            modes_noisy, y_noisy = self.synthetic_data(seed=seed)
            # save training data
            torch.save(y_noisy, './save/synthetic_data/'+self.filename+'_data')
            # equipments for WGAN : prepare (t,x) in two ways
            # time as float
            X_data = torch.cat([self.t_grids.repeat(self.num_modes*self.num_noised,1), y_noisy],-1)
            # time as one-hot vector
            t_data_onehot = X_data[:,0].clone()
            for t_ind in range(self.num_bins):
                t_data_onehot[X_data[:,0]==self.t_grids[t_ind]] = t_ind
            t_data_onehot = F.one_hot(t_data_onehot.to(torch.int64), num_classes=self.num_bins)
            X_data_onehot = torch.cat([t_data_onehot, X_data[:,1:]],-1)
            # cut time for each mode to make rcs data
            if self.num_cut>0:
                rcs_masks = []
                for i, mode in enumerate(self.modes):
                    X_temp = X_data[i*self.num_noised*self.num_bins:(i+1)*self.num_noised*self.num_bins,:]
                    t_cut = self.t_grids[torch.randperm(self.num_bins-1)[:self.num_cut]+1] # randomly select cut time except initial
                    cuts = torch.cat([(X_temp[:,0]!=t_cut[j]).view(-1,1) for j in range(self.num_cut)],-1)
                    rcs_msk = torch.prod(cuts,-1).bool()
                    rcs_masks.append(rcs_msk)
                    print('[ mode ',mode,' censored at time ',np.sort(t_cut.view(-1).numpy()),' ]')
                rcs_mask_cat = torch.cat(rcs_masks,0)
                X_data = X_data[rcs_mask_cat]
                X_data_onehot = X_data_onehot[rcs_mask_cat]
                self.num_data = len(X_data)
        else:
            self.change_settings_for_real_data(real_data) # there's some change in settings
            y_noisy = None
            modes_noisy = None
            # find dataset file
            data_path = path+'/real_data'+'/'+real_data+'.csv'
            if not os.path.isfile(data_path):
                print('[ '+real_data+'.csv Not Found in ./save/real_data ]')
                return [], [], []
            if real_data in ['Abeta40', 'Abeta42']:
                data_path = path+'/real_data'+'/'+real_data+'.csv'
                print('[ Load '+real_data+' ]')
                df = pd.read_csv(path+'/real_data'+'/'+real_data+'.csv')
                age_value_pairs_42 = df[['Age', 'Value']].values
                max_value_12_months_42 = age_value_pairs_42[age_value_pairs_42[:, 0] == 12, 1].max()
                # scale all values so that the maximum value for 12-month-old mice equals 1.0
                age_value_pairs_42[:, 1] = age_value_pairs_42[:, 1] / max_value_12_months_42
                # filter the data to include only the specified time points
                filtered_data = [age_value_pairs_42[age_value_pairs_42[:, 0] == t] for t in self.t_numpy]
                X_data = torch.FloatTensor(np.concatenate(filtered_data, 0))
            # time as one-hot vector
            t_data_onehot = X_data[:,0].clone()
            for t_ind in range(self.num_bins):
                t_data_onehot[X_data[:,0]==self.t_grids[t_ind]] = t_ind
            t_data_onehot = F.one_hot(t_data_onehot.to(torch.int64), num_classes=self.num_bins)
            X_data_onehot = torch.cat([t_data_onehot, X_data[:,1:]],-1)
            # change settings related to training
            self.num_data = len(X_data)
            self.num_gen = self.prms_GAN['num_gen_ratio']*self.num_data
            self.filename = self.name+'_'+real_data 
        if self.sc:
            if real_data is None:
                y_noisy = self.scale_y(y_noisy)
                modes_noisy = self.scale_p(modes_noisy)
            X_data[:,1:] = self.scale_y(X_data[:,1:])
            X_data_onehot[:,self.num_bins:] = self.scale_y(X_data_onehot[:,self.num_bins:])
        # visualize params and data if needed
        if print_results:
            print('[ total number of true data points : '+str(self.num_data)+' ]')
            self.visualize_data(modes_noisy, X_data, y_noisy=y_noisy)
        return modes_noisy, X_data, X_data_onehot
    
    def change_settings_for_real_data(self, real_data):
        if real_data in ['Abeta40','Abeta42']:
            self.modes = []
            self.num_modes = 0
            self.t_numpy = np.array([4., 8., 12., 18.])
            self.t_grids = torch.FloatTensor(self.t_numpy)
            self.num_bins = len(self.t_numpy) 
            self.times_all = torch.tensor(self.t_numpy).to(torch.float32).view(-1,1).to(self.device)
            self.onehot_all = F.one_hot(torch.arange(0, self.num_bins), num_classes=self.num_bins).to(torch.float32).to(self.device)
    
    def synthetic_data(self, seed=1234):
        if self.name in ['exp', 'log']:
            # generate noised modes
            noise = np.random.uniform(low=self.plow, high=self.phigh, size=[self.num_modes*self.num_noised, self.num_p])
            modes_noisy = np.round(noise*np.reshape(np.tile(self.modes, (1,self.num_noised)), (-1,self.num_p)), 2)
            # generate trajectories from each modes
            y_noisy = []
            for mode in modes_noisy:
                y_values = self.solution(torch.concat([self.t_grids]+[p*torch.ones_like(self.t_grids) for p in mode],-1)).view(-1,1)
                y_values = y_values*(1+self.data_noise_scale*torch.normal(mean=0.0, std=1.0, size=y_values.shape))
                y_noisy.append(y_values)
            y_noisy = torch.concat(y_noisy,0)
            return torch.tensor(modes_noisy).to(torch.float32), y_noisy
        if self.name in ['lorenz']:
            # generate noised modes
            noise = np.random.uniform(low=self.plow, high=self.phigh, size=[self.num_modes*self.num_noised, self.num_p])
            modes_noisy = noise*np.reshape(np.tile(self.modes, (1,self.num_noised)), (-1,self.num_p))
            # generate trajectories from each modes
            y_noisy = []
            self.inits = torch.cat([self.Y_init[i] *(1 + self.init_noise_scale_wgan*torch.rand(len(modes_noisy), 1)) for i in self.eq_list],-1)
            for ii, mode in enumerate(modes_noisy):
                y_values = self.solution(torch.concat([self.t_grids]+[p*torch.ones_like(self.t_grids) for p in mode],-1), inits=self.inits[ii:ii+1,:])
                y_values = y_values*(1+self.data_noise_scale*torch.normal(mean=0.0, std=1.0, size=y_values.shape))
                y_noisy.append(y_values) # we need noised initial
            y_noisy = torch.concat(y_noisy,0)
            return torch.tensor(modes_noisy).to(torch.float32), y_noisy

    # solution of DE (for each single parameter)
    def solution(self, data, inits=None, rtol=1e-10, atol=1e-13):
        if inits==None:
            if self.name in ['exp','log']:
                inits = self.Y_init[0]
            else:
                y0 = self.Y_init
        else:
            if self.name in ['lorenz']:
                y0 = list(inits[0,:].detach().cpu().numpy())
        if self.name == 'exp':
            x, r = data[:,0].view(-1,1), data[:,1].view(-1,1)
            return inits*torch.exp(r * x)
        if self.name == 'log':
            x, r, K = data[:,0].view(-1,1), data[:,1].view(-1,1), data[:,2].view(-1,1)
            return K/(1 + ((K/inits)-1) * torch.exp(-r * x))
        if self.name == 'lorenz':
            t_eval = data[:,0].detach().cpu().numpy()
            ps = data[:,1:].detach().cpu().numpy()
            sigma, rho, beta = ps[0,0], ps[0,1], ps[0,2]
            sol = solve_ivp(model_lorenz, t_span=(t_eval[0], t_eval[-1]), y0=y0, t_eval=t_eval, args=(sigma, rho, beta), method='LSODA', rtol=rtol, atol=atol)
            return torch.FloatTensor(sol.y.T)
    
    # metric of parameter distributions
    def WD(self, p_output_plot, modes_noisy):
        wd = 0
        for i in range(self.num_p):
            wd += wasserstein_distance(p_output_plot[:,i].view(-1).detach().cpu().numpy(), modes_noisy[:,i].view(-1).detach().cpu().numpy())
        return wd
    
    # metric of minimum distance
    def point_min_dist(self, X_data, p_output_plot):
        n_traj = len(p_output_plot)
        print('n_traj:',n_traj)
        t_pred = self.t_grids.clone().to(self.device)
        if self.name in ['exp','log']:
            # first get true trajectories from generated parameters
            y_vals = []
            for i, mode in enumerate(p_output_plot):  # Assuming A_list_ap is your list of parameter sets
                y_vals.append(self.solution(torch.concat([t_pred]+[p*torch.ones_like(t_pred) for p in mode],-1)).view(1,-1))
            y_vals = torch.cat(y_vals, 0)
            d_t = 0.0
            for t_ind in range(self.num_bins):
                y_true = X_data[X_data[:,0]==t_pred[t_ind]]
                if len(y_true)>0:
                    y_dists = []
                    for i in range(y_true.shape[0]):
                        y_dists.append(torch.abs(y_vals[:,t_ind:t_ind+1]-y_true[i,1]))
                    d_t += torch.sum(torch.min(torch.cat(y_dists,-1),-1).values)
            return d_t/n_traj
        if self.name in ['lorenz']:
            # first get true trajectories from generated parameters
            y_vals = []
            for i, mode in enumerate(p_output_plot):  # Assuming A_list_ap is your list of parameter sets
                y_vals.append(self.solution(torch.concat([t_pred]+[p*torch.ones_like(t_pred) for p in mode],-1)))
            y_vals = torch.cat(y_vals, 0)
            d_t = 0.0
            for t_ind in range(self.num_bins):
                y_true = X_data[X_data[:,0]==t_pred[t_ind]]
                if len(y_true)>0:
                    y_dists = []
                    for i in range(y_true.shape[0]):
                        y_dists.append(torch.abs(y_vals[:,t_ind:t_ind+1]-y_true[i,1]))
                    d_t += torch.sum(torch.min(torch.cat(y_dists,-1),-1).values)
            return d_t/n_traj
    
    ############################################## for training networks ################################################
    def load_pinn(self, load_params=True, train_nets=False):
        if self.deeponet:
            branch_features = self.num_p + self.num_eq  #
            trunk_features = 1  
            output_dim = self.num_eq 
            self.deeponets = {}
            for i in self.eq_list:
                self.deeponets[i] = DeepONet(branch_features, trunk_features, output_dim).to(self.device)
            self.path_deeponet = {}
            for i in self.eq_list:
                self.path_deeponet[i] = path+'/DeepONet/deeponets'+str(i)+'_'+self.name+'_init='+str(self.Y_init[0])+'.pth'
                if load_params:
                    self.deeponets[i].load_state_dict(torch.load(self.path_deeponet[i], map_location=self.device))
                    if train_nets:
                        self.deeponets[i].train()
                    else:
                        self.deeponets[i].eval()
            if train_nets:
                self.optimizer = torch.optim.Adam([{'params': self.deeponets[i].parameters(), 'lr':self.lr} for i in self.eq_list])     
        else:
            self.hypernets, self.trunknets = {}, {}
            for i in self.eq_list:
                if self.name in ['lorenz']:
                    self.hypernets[i] = HyperNetwork_lorenz(num_sensors=self.num_p+self.num_eq, total_params=self.total_params, depth=self.depth_hyp, width=self.width_hyp, activation=self.act).to(self.device)
                else:
                    self.hypernets[i] = HyperNetwork(num_sensors=self.num_p+self.num_eq, num_chunks_in=self.num_chunks_in, total_params=self.total_params, depth=self.depth_hyp, width=self.width_hyp, activation=self.act).to(self.device)
                self.trunknets[i] = TrunkNetwork(self.layers_config, nn.Tanh()).to(self.device)
            self.path_hyper = {}
            self.path_trunk = {}
            for i in self.eq_list:
                self.path_hyper[i] = path+'/hyperPINN/hypernets'+str(i)+'_'+self.name+'_init='+str(self.Y_init[0])+'.pth'
                self.path_trunk[i] = path+'/hyperPINN/trunknets'+str(i)+'_'+self.name+'_init='+str(self.Y_init[0])+'.pth'
                if load_params:
                    self.hypernets[i].load_state_dict(torch.load(self.path_hyper[i], map_location=self.device))
                    self.trunknets[i].load_state_dict(torch.load(self.path_trunk[i], map_location=self.device))
                    if train_nets:
                        self.hypernets[i].train()
                        self.trunknets[i].train()
                    else:
                        self.hypernets[i].eval()
                        self.trunknets[i].eval()
            if train_nets:
                params_hyp = [{'params': self.hypernets[i].parameters(), 'lr':self.lr} for i in self.eq_list]
                params_tru = [{'params': self.trunknets[i].parameters(), 'lr':self.lr} for i in self.eq_list]
                self.optimizer = torch.optim.Adam(params_hyp+params_tru)  
    
    def load_gan(self, load_params=True):
        self.g_model = G_model(hidden_dims=[self.noise_dim]+([self.ratio_G*self.width]*self.depth)+[self.num_p]).to(self.device)
        self.d_model = D_model(hidden_dims=[self.num_bins+len(self.eq_list)]+([self.width]*self.depth)+[1]).to(self.device)
        self.path_G = path+'/WGAN/best_G_'+self.filename+'.pth'
        self.path_D = path+'/WGAN/best_D_'+self.filename+'.pth'
        for pth in [self.path_G, self.path_D]:
            if not os.path.isfile(pth): 
                print('[ Saved WGAN Not Found ]')
                load_params=False 
                break
        if load_params:
            self.g_model.load_state_dict(torch.load(self.path_G, map_location=self.device))
            self.d_model.load_state_dict(torch.load(self.path_D, map_location=self.device))
        self.optimizer_G = torch.optim.Adam([{'params': self.g_model.parameters(), 'lr':self.lr_G, 'betas':(0, 0.9)}]) # RMSProp
        self.optimizer_D = torch.optim.Adam([{'params': self.d_model.parameters(), 'lr':self.lr_D, 'betas':(0, 0.9)}]) # RMSProp
        self.input_init = (torch.FloatTensor(self.Y_init)*(1+self.init_noise_scale_wgan*torch.rand([((self.num_gen//self.num_bins)+1)*self.num_bins,self.num_eq]))) # random dist of inits
        if self.num_gen_traj<0:
            self.input_init_plot = (torch.FloatTensor(self.Y_init)*(1+self.init_noise_scale_wgan*torch.rand([((self.num_gen_plot//self.num_bins)+1)*self.num_bins,self.num_eq])))
            self.time_gen_plot = self.times_all.repeat((self.num_gen_plot//self.num_bins)+1,1)
        else:
            self.input_init_plot = (torch.FloatTensor(self.Y_init)*(1+self.init_noise_scale_wgan*torch.rand([self.num_gen_traj*self.num_bins,self.num_eq])))
            self.time_gen_plot = self.times_all.repeat(self.num_gen_traj,1)
        if self.sc:
            self.input_init = self.scale_y(self.input_init)
            self.input_init_plot = self.scale_y(self.input_init_plot)
        self.input_init = self.input_init.to(self.device)
        self.input_init_plot = self.input_init_plot.to(self.device)
    
    def calculate_derivative(self, y, x) :
        return torch.autograd.grad(y, x, create_graph=True, grad_outputs=torch.ones(y.size()).to(self.device))[0]
    
    def hyperpinn_forward(self, sensor, input_times):
        preds = []
        if self.deeponet:
            for i in self.eq_list:
                preds.append(self.deeponets[i](sensor, input_times))
        else:
            for i in self.eq_list:
                params = self.hypernets[i](sensor)
                params = torch.split(params, [(inp * out + out) for inp, out in self.layers_config], dim=1)
                preds.append(self.trunknets[i](input_times, params))
        return preds
    
    def pinn_loss(self, preds, sensor, time, only_p_unsc=False):
        if self.name == 'exp':
            lossr = torch.mean((self.calculate_derivative(preds[0], time)-sensor[:,0:1]*preds[0])**2)
        if self.name == 'log':
            lossr = torch.mean((self.calculate_derivative(preds[0], time)-sensor[:,0:1]*preds[0]*(1-(preds[0]/sensor[:,1:2])))**2)
        if self.name == 'lorenz':
            if self.sc:
                if only_p_unsc:
                    preds_unsc = torch.cat(preds,-1)
                else:
                    preds_unsc = self.scale_y(torch.cat(preds,-1),backward=True)
                sensor_unsc = self.scale_p(sensor[:,:self.num_p],backward=True)
            else:
                preds_unsc = torch.cat(preds,-1)
                sensor_unsc = sensor[:,:self.num_p]
            X, Y, Z = [preds_unsc[:,ii:ii+1] for ii in self.eq_list]
            sigma, rho, beta = [sensor_unsc[:,ii:ii+1] for ii in range(self.num_p)]
            # DE
            term1 = self.calculate_derivative(X, time) - sigma*(Y-X)
            term2 = self.calculate_derivative(Y, time) - (rho-Z)*X + Y
            term3 = self.calculate_derivative(Z, time) - X*Y + beta*Z
            lossr = torch.mean((term1**2)+(term2**2)+(term3**2))
        return lossr
    
    def train_pinn(self, sensor, y, EPOCHS=100000, iter_print=1000, iter_plot=2000, iter_save=10000, ratio=0.5, only_p_unsc=False):
        # Training loop with reduced batch size
        sensor, y = sensor.to(self.device), y.to(self.device)
        times_entire = (self.times.repeat(self.Np,1).reshape(-1,1)).to(self.device)
        # train/test
        times_tr, times_te = self.get_data_split(times_entire)
        sensor_tr, sensor_te = self.get_data_split(sensor)
        y_tr, y_te = self.get_data_split(y)
        print('[ total number of train data points : '+str(len(times_tr))+' ]')
        # training
        num_batches = (len(times_tr) // self.batch_size) + 1
        losses, losses_d, losses_r, test_errors = [],[],[],[]
        time_start = time.time()
        for t in range(1, EPOCHS+1):
            for i in self.eq_list:
                if self.deeponet:
                    self.deeponets[i].train()
                else:
                    self.hypernets[i].train()
                    self.trunknets[i].train()
            loss_d = 0.0
            loss_r = 0.0
            for i_b in range(num_batches):
                b_start, b_end = i_b*self.batch_size, (i_b+1)*self.batch_size
                if i_b == num_batches-1:
                    if b_start==len(times_tr):
                        break
                    sensor_b, times_b, y_b  = sensor_tr[b_start:], times_tr[b_start:].requires_grad_(True), y_tr[b_start:]
                else:
                    sensor_b, times_b, y_b  = sensor_tr[b_start:b_end], times_tr[b_start:b_end].requires_grad_(True), y_tr[b_start:b_end]
                self.optimizer.zero_grad()
                preds = self.hyperpinn_forward(sensor_b, times_b)
                lossd = 0.0
                for i, pred in enumerate(preds):
                    lossd += torch.mean((pred - y_b[:,i:i+1])**2)
                if self.beta==0.0:
                    lossr = torch.zeros_like(lossd)
                else:
                    lossr = self.pinn_loss(preds, sensor_b, times_b, only_p_unsc=only_p_unsc)
                (lossd+self.beta*lossr).backward()
                self.optimizer.step()
                loss_d += lossd.item()
                loss_r += lossr.item()
            
            losses.append(loss_d+self.beta*loss_r)
            losses_d.append(loss_d)
            losses_r.append(loss_r)
            # valid & test error
            for i in self.eq_list:
                if self.deeponet:
                    self.deeponets[i].eval()
                else:
                    self.hypernets[i].eval()
                    self.trunknets[i].eval()
            preds_te = self.hyperpinn_forward(sensor_te, times_te)
            test_error_temp = []
            for i in self.eq_list:
                test_error_temp.append((preds_te[i] - y_te[:,i:i+1])**2)
            test_error = torch.mean(torch.sqrt(torch.sum(torch.cat(test_error_temp,-1),-1)+self.eps))
            test_errors.append(test_error.item())
            # print results
            if t % iter_print == 0:
                print("[ %s/%s | loss: %06.6f | loss_data: %06.6f | loss_physics: %06.6f | test L2 error %06.6f  | training time : %02.2f min ]" % \
                       (t, EPOCHS, loss_d+loss_r, loss_d, loss_r, test_error, (time.time()-time_start)/60))
            # plot results
            if t % iter_plot == 0:
                self.plot_results_pinn()
            # save model
            if t % iter_save == 0 or test_error<self.tol:
                if self.deeponet:
                    self.plot_results_pinn(fig_name=path+'/figures/DeepONet_'+self.name+'_init='+str(self.Y_init[0])+'.png')
                    for i in self.eq_list:
                        torch.save(self.deeponets[i].state_dict(), self.path_deeponet[i])
                else:
                    self.plot_results_pinn(fig_name=path+'/figures/hyperPINN_'+self.name+'_init='+str(self.Y_init[0])+'.png')
                    for i in self.eq_list:
                        torch.save(self.hypernets[i].state_dict(), self.path_hyper[i])
                        torch.save(self.trunknets[i].state_dict(), self.path_trunk[i])
                if test_error<self.tol:
                    break
        return losses, losses_d, losses_r, test_errors
    
    def modify_gen(self, data):
        return data+self.center
        
    def train_G(self, X_data_onehot):
        loss_list, loss_list1, loss_list2 = [], [], []
        self.g_model.train()
        self.d_model.eval()
        self.optimizer_G.zero_grad()
        # sample parameters from generator
        num_gen_traj = (self.num_gen//self.num_bins)+1 # number of generated trajectories (parameters)
        noise = torch.normal(mean=torch.zeros([num_gen_traj, self.noise_dim]), std=1.0).to(self.device)
        p_output = self.modify_gen(self.g_model(noise)).repeat(1,self.num_bins).reshape(-1,self.num_p)
        # emulate the fake data
        sensor = torch.cat([p_output, self.input_init],-1)
        preds = self.hyperpinn_forward(sensor, self.times_all.repeat(num_gen_traj,1))
        X_gen = torch.cat([self.onehot_all.repeat(num_gen_traj,1)] + preds, dim=-1)
        d_output_gen = self.d_model(X_gen)
        # calulate the loss
        loss1 = -torch.mean(d_output_gen)
        loss2 = torch.zeros_like(loss1)
        loss = loss1
        loss.backward()
        self.optimizer_G.step()
        w_dist = torch.mean(self.d_model(X_data_onehot))+loss1.detach() # Wasserstein Distance
        loss_list.append((loss).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2), w_dist.item()
    
    def train_D(self, X_data_onehot):
        loss_list, loss_list1, loss_list2 = [], [], []
        self.g_model.eval()
        self.d_model.train()
        self.optimizer_D.zero_grad()
        # sample parameters from generator
        num_gen_traj = (self.num_gen//self.num_bins)+1
        noise = torch.normal(mean=torch.zeros([num_gen_traj, self.noise_dim]), std=1.0).to(self.device)
        p_output = self.modify_gen(self.g_model(noise)).repeat(1,self.num_bins).reshape(-1,self.num_p)
        # emulate the fake data
        sensor = torch.cat([p_output, self.input_init],-1)
        preds = self.hyperpinn_forward(sensor, self.times_all.repeat(num_gen_traj,1))
        X_gen = torch.cat([self.onehot_all.repeat(num_gen_traj, 1)] + preds, dim=-1)
        # Get Discriminator Outputs
        d_output_data = self.d_model(X_data_onehot)
        # gradient penalty
        if num_gen_traj*self.num_bins == self.num_data:
            X_gen_rand = X_gen
        else:
            X_gen_rand = X_gen[torch.randperm(num_gen_traj*self.num_bins)[:self.num_data], :]
        # data augmentation for calculating gradient penalty
        alpha = torch.rand([self.num_data, 1]).to(self.device)
        X_gen_aug = alpha*(X_data_onehot+ 0.5 * X_data_onehot.std() * torch.rand(X_data_onehot.size()).to(self.device))+(1-alpha)*X_gen_rand
        X_gen_grad = Variable(X_gen_aug, requires_grad=True)
        d_output_gen = self.d_model(X_gen)
        d_output_gen_grad = self.d_model(X_gen_grad)
        gp = self.calculate_derivative(d_output_gen_grad, X_gen_grad) # gradent penalty
        # calulate the loss
        loss1 = -torch.mean(d_output_data)+torch.mean(d_output_gen)
        loss2 = torch.mean((torch.sqrt(self.eps+torch.sum(gp**2, -1))-1)**2)
        loss = loss1+self.lamb*loss2
        loss.backward()
        self.optimizer_D.step()
        loss_list.append((loss).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2)
    
    def mmd(self, x, y, sigma=1.0):
        xx = torch.cdist(x, x) ** 2
        yy = torch.cdist(y, y) ** 2
        xy = torch.cdist(x, y) ** 2
        k_xx = torch.exp(-xx / (2 * sigma**2)).mean()
        k_yy = torch.exp(-yy / (2 * sigma**2)).mean()
        k_xy = torch.exp(-xy / (2 * sigma**2)).mean()
        return k_xx + k_yy - 2 * k_xy
    
    def train_gan(self, modes_noisy, X_data, X_data_onehot, EPOCHS=100000, T=5000, iter_D=5, iter_print=100, iter_plot=1000, iter_save=5000):
        X_data, X_data_onehot = X_data.to(self.device), X_data_onehot.to(self.device)
        time_start = time.time()
        total_lossG, total_lossD, w_dists = [], [], []
        w_dist_min = 100
        t_min = 0
        self.best_G = copy.deepcopy(self.g_model)
        self.best_D = copy.deepcopy(self.d_model)
        # before training, let the generator produce outputs following a uniform distribution
        print('Pre-training Starts')
        N_pre = 1000
        self.optimizer_G.param_groups[0]['lr'] = 10*self.lr_G
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer_G,start_factor=0.1,end_factor=1.0,total_iters=1000)
        for t in range(1000):
            self.g_model.train()
            self.d_model.eval()
            self.optimizer_G.zero_grad()
            noise = torch.normal(mean=torch.zeros([N_pre, self.noise_dim]), std=1.0).to(self.device)
            g_output = self.modify_gen(self.g_model(noise))
            target = (torch.rand(N_pre, self.num_p)*(self.pscale[1:,:]-self.pscale[:1,:])+self.pscale[:1,:]).to(self.device)
            loss = self.mmd(g_output, target)
            loss.backward()
            self.optimizer_G.step()
            scheduler.step()
        print('Pre-training Ends, Final MMD: ', loss.item())
        # reset and start WGAN training
        self.optimizer_G.param_groups[0]['lr'] = self.lr_G
        self.plot_results_gan(modes_noisy, X_data, fig_name=path+'/figures/'+self.filename+'_0'+'.png')
        for t in range(1, EPOCHS+1):
            for j in range(iter_D):
                loss_D, _, loss_gr = self.train_D(X_data_onehot)
            loss_G, _, _, w_dist = self.train_G(X_data_onehot)
            total_lossG.append(loss_G)
            total_lossD.append(loss_D)
            # achieve minimum wasserstein distance
            if w_dist>0:
                w_dists.append(w_dist)
            else:
                if t == 1:
                    w_dists.append(w_dist_min)
                w_dists.append(w_dists[-1])
            if t>T and w_dist < w_dist_min and w_dist>0:
                self.best_G = copy.deepcopy(self.g_model)
                self.best_D = copy.deepcopy(self.d_model)
                w_dist_min = w_dist
                t_min = t
            # print logs
            if t%iter_print == 0 :
                print("[ %s/%s | loss_D: %06.6f | loss_G: %06.6f | gradient penalty loss : %06.6f | w_dist : %06.6f | w_dist_min %06.6f at iter %s | training time : %02.2f min ]" % \
                       (t, EPOCHS, -loss_D, loss_G, loss_gr, w_dist, w_dist_min, t_min, (time.time()-time_start)/60))
            if t%iter_plot == 0:
                self.plot_results_gan(modes_noisy, X_data)
            if t%iter_save == 0:
                # save best models
                torch.save(self.best_G.state_dict(), self.path_G)
                torch.save(self.best_D.state_dict(), self.path_D)
                self.plot_results_gan(modes_noisy, X_data, fig_name=path+'/figures/'+self.filename+'_'+str(t)+'.png')
        return total_lossG, total_lossD, w_dists
    
    ############################################## for plotting results ################################################
    # Visualize both posterior distribution of p & rcs data        
    def visualize_data(self, modes_noisy, X_data, y_noisy=None, p_output_plot=None, X_gen=None, fig_name=None, plot_traj=False):
        font_size = 18
        # show posterior distribution : empty for real data
        if self.name == 'exp':
            fig = plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(1,2,1)
            if p_output_plot is not None:
                cp1 = ax1.hist(p_output_plot[:,0], density=True, bins=100)
            if modes_noisy is not None:
                cp1 = ax1.vlines(x=modes_noisy[:,0], ymin=0, ymax=1, linewidth=0.5, color='r')
            ax1.set_title('Posterior', fontsize=font_size)
            ax1.legend()
            # show trajectories
            ax2 = fig.add_subplot(1,2,2)
            num_traj = self.num_modes*self.num_noised
            if y_noisy is not None:
                for i in range(num_traj):
                    cp1 = ax2.plot(self.t_grids.view(-1), y_noisy.view(-1)[i*self.num_bins:(i+1)*self.num_bins], '--', linewidth=0.5, label='true data before censored' if i==0 else '')
            if X_gen is not None:
                cp1 = ax2.scatter(X_gen[:,0], X_gen[:,1], s=10, c='b', label='fake data', alpha=0.1)
            if p_output_plot is not None and plot_traj:
                t_pred = torch.linspace(self.tmin, self.tmax, 101).view(-1,1) # for trajectories
                for i, mode in enumerate(p_output_plot):  # Assuming A_list_ap is your list of parameter sets
                    y_values = self.solution(torch.concat([t_pred]+[p*torch.ones_like(t_pred) for p in mode],-1)).view(-1,1)
                    ax2.plot(t_pred, y_values.numpy().reshape(-1), linewidth=0.3, color='b', alpha=0.2, label='fake solution' if i==0 else '')
            cp2 = ax2.scatter(X_data[:,0], X_data[:,1], s=20, c='r', label='true data')
            ax2.set_title('Data', fontsize=font_size)
            ax2.set_xlabel('Times', fontsize=font_size)
            ax2.set_ylabel('Y', fontsize=font_size, rotation=0)
            ax2.legend()
            if fig_name is None:
                plt.show()
            else:
                plt.savefig(fig_name)
                plt.close(fig)
        if self.name == 'log':
            fig = plt.figure(figsize=(10,4.5))
            plt.subplots_adjust(wspace=0.3)
            if modes_noisy is None:
                plot_c = 'k'
            else:
                plot_c = 'b'
            # show trajectories
            ax2 = fig.add_subplot(1,2,1)
            num_traj = self.num_modes*self.num_noised
            if y_noisy is not None:
                for i in range(num_traj):
                    cp1 = ax2.plot(self.t_grids.view(-1), y_noisy.view(-1)[i*self.num_bins:(i+1)*self.num_bins], '--', linewidth=0.5, label='true data before censored' if i==0 else '')
            if X_gen is not None:
                if not plot_traj:
                    cp1 = ax2.scatter(X_gen[:,0], X_gen[:,1], s=10, c='b', label='fake data', alpha=0.2)
            if p_output_plot is not None and plot_traj:
                t_pred = torch.linspace(self.tmin, self.tmax, 101).view(-1,1) # for trajectories
                for i, mode in enumerate(p_output_plot):  # Assuming A_list_ap is your list of parameter sets
                    y_values = self.solution(torch.concat([t_pred]+[p*torch.ones_like(t_pred) for p in mode],-1)).view(-1,1)
                    ax2.plot(t_pred, y_values.numpy().reshape(-1), linewidth=0.3, color=plot_c, alpha=0.2, label='fake solution' if i==0 else '')
            cp2 = ax2.scatter(X_data[:,0], X_data[:,1], s=20, c='r', label='true data', zorder=100000)
            ax2.set_xlabel('Time (Month)', fontsize=font_size)
            ax2.set_ylabel('Population', fontsize=font_size)
            # show posterior distribution
            ax1 = fig.add_subplot(1,2,2)
            if p_output_plot is not None:
                cp1 = ax1.scatter(p_output_plot[:,0], p_output_plot[:,1], s=10, c=plot_c, label='Estimated', alpha=0.2)
            if modes_noisy is not None:
                cp1 = ax1.scatter(modes_noisy[:,0], modes_noisy[:,1], s=10, c='r', label='true parameters')
            ax1.set_xlabel('r', fontsize=font_size)
            ax1.set_ylabel('K', fontsize=font_size)
            ax1.set_title('Parameter', fontsize=font_size)
            ax1.legend()
            if fig_name is None:
                plt.show()
            else:
                plt.savefig(fig_name)
                plt.close(fig)
        if self.name in ['lorenz']:
            # show posterior distribution 
            fig = plt.figure(figsize=(10,6))
            fig.suptitle('Posterior', fontsize=font_size)
            axs = {}
            for i in range(self.num_p):
                if self.name == 'lorenz':
                    axs[i] = fig.add_subplot(1,3,i+1)
                if p_output_plot is not None:
                    cp1 = axs[i].hist(p_output_plot[:,i], density=True, bins=100)
                if modes_noisy is not None:
                    cp1 = axs[i].vlines(x=modes_noisy[:,i], ymin=0, ymax=1, linewidth=0.5, color='r')
            if fig_name is None:
                plt.show()
            else:
                plt.savefig(fig_name[:-4]+'_post'+'.png')
                plt.close(fig)
            # show trajectories
            if p_output_plot is not None and plot_traj:
                if self.sc:
                    p_output_unsc = self.scale_p(p_output_plot, backward=True)
                else:
                    p_output_unsc = p_output_plot
                t_pred = torch.linspace(self.tmin, self.tmax, 101).view(-1,1) # for trajectories
                y_vals = []
                for i, mode in enumerate(p_output_unsc):  # Assuming A_list_ap is your list of parameter sets
                    y_values = self.solution(torch.concat([t_pred]+[p*torch.ones_like(t_pred) for p in mode],-1))
                    if self.sc:
                        y_vals.append(self.scale_y(y_values))
                    else:
                        y_vals.append(y_values)
            fig = plt.figure(figsize=(10,10))
            num_traj = self.num_modes*self.num_noised
            axs = {}
            outputs = self.outputs
            # trajectories
            for k in self.eq_list:
                if self.name == 'lorenz':
                    axs[k] = fig.add_subplot(1,3,k+1)
                if y_noisy is not None:
                    for i in range(num_traj):
                        cp1 = axs[k].plot(self.t_grids.view(-1), y_noisy[:,k].view(-1)[i*self.num_bins:(i+1)*self.num_bins], '--', linewidth=0.5, label='true data before censored' if i==0 else '')
                if X_gen is not None:
                    if X_data.shape[-1] == 2:
                        cp2 = axs[k].scatter(X_gen[:,0], X_gen[:,1], s=10, c='b', label='fake data', alpha=0.1)
                    else:
                        cp2 = axs[k].scatter(X_gen[:,0], X_gen[:,k+1], s=10, c='b', label='fake data', alpha=0.1)
                if p_output_plot is not None and plot_traj:
                    for y_values in y_vals:
                        cp3 = axs[k].plot(t_pred, y_values[:,k].numpy().reshape(-1), linewidth=0.3, color='b', alpha=0.2, label='fake solution' if i==0 else '')
                if X_data.shape[-1] == 2:
                    cp3 = axs[k].scatter(X_data[:,0], X_data[:,1], s=20, c='r', label='true data')
                else:
                    cp3 = axs[k].scatter(X_data[:,0], X_data[:,k+1], s=20, c='r', label='true data')
                axs[k].set_title(outputs[k], fontsize=font_size)
                axs[k].set_xlabel('Times', fontsize=font_size)
                axs[k].legend()
            if fig_name is None:
                plt.show()
            else:
                plt.savefig(fig_name[:-4]+'_traj'+'.png')
                plt.close(fig)
    
    def plot_results_pinn(self, fig_name=None):
        for i in self.eq_list:
            if self.deeponet:
                self.deeponets[i].eval()
            else:
                self.hypernets[i].eval()
                self.trunknets[i].eval()
        # prediction of hyperPINN
        sensor_plot, y_plot = self.get_data_pinn(N_p=1)
        preds = self.hyperpinn_forward(sensor_plot.to(self.device), self.times.reshape(-1,1).to(self.device))
        # Plot the results
        title_name = 'prediction for '+', '.join([self.param_names[j]+'='+str(round(sensor_plot[0,j].item(),2)) for j in range(self.num_p)])+', initial condition='+str(round(sensor_plot[0,-1].item(),8))
        if self.name in ['exp','log']:
            fig = plt.figure(figsize=(5, 5))
            plt.plot(self.times.view(-1,1).detach().cpu().numpy(), y_plot[:,i].detach().cpu().numpy(), 'b-', label='True Data')
            plt.plot(self.times.view(-1,1).detach().cpu().numpy(), preds[i].detach().cpu().numpy(), 'r--', label='Predicted Data')
            plt.title(title_name)
            plt.xlabel('Time')
            plt.ylabel('Y', rotation=90)
            plt.legend()
            if fig_name is None:
                plt.show()
            else:
                plt.savefig(fig_name)
                plt.close(fig)
        if self.name in ['lorenz']:
            outputs = self.outputs
            fig = plt.figure(figsize=(10, 8))
            axs = {}
            for i in self.eq_list:
                if self.name == 'lorenz':
                    axs[i] = fig.add_subplot(1,3,i+1)
                axs[i].plot(self.times.view(-1,1).detach().cpu().numpy(), y_plot[:,i].detach().cpu().numpy(), 'b-', label='True Data')
                axs[i].plot(self.times.view(-1,1).detach().cpu().numpy(), preds[i].detach().cpu().numpy(), 'r--', label='Predicted Data')
                axs[i].set_xlabel('Time')
                plt.ylabel(outputs[i], rotation=90)
                plt.legend()
            plt.title(title_name)
            if fig_name is None:
                plt.show()
            else:
                plt.savefig(fig_name)
                plt.close(fig)
    
    def plot_results_gan(self, modes_noisy, X_data, fig_name=None, plot_traj=False, return_gen=False, get_scores=False):
        # real data ; modes_noisy == None
        X_data = X_data.to(self.device)
        if fig_name is None:
            gmodel, dmodel = self.g_model, self.g_model
        else:
            gmodel, dmodel = self.best_G, self.best_D
        gmodel.eval()
        dmodel.eval()
        # sample sarameters from generator
        if self.num_gen_traj < 0:
            num_gen_traj = (self.num_gen_plot//self.num_bins)+1
        else:
            num_gen_traj = self.num_gen_traj
        noise = torch.normal(mean=torch.zeros([num_gen_traj, self.noise_dim]), std=1.0).to(self.device)
        p_output_plot = self.modify_gen(gmodel(noise))
        p_output = p_output_plot.repeat(1,self.num_bins).reshape(-1,self.num_p)
        # metric
        if get_scores:
            dist_params = self.WD(p_output_plot, modes_noisy)
            print('WD: ', dist_params.item())
        # emulate the fake data
        sensor = torch.cat([p_output, self.input_init_plot],-1)
        preds = self.hyperpinn_forward(sensor, self.time_gen_plot)
        X_gen = torch.cat([self.time_gen_plot] + preds, dim=-1)
        self.visualize_data(modes_noisy, X_data.detach().cpu(), p_output_plot=p_output_plot.detach().cpu(), X_gen=X_gen.detach().cpu(), fig_name=fig_name, plot_traj=plot_traj)
        if return_gen:
            return p_output_plot.detach().cpu()
    
    # plot trajectories from estimated parameters
    def plot_results_real_data(self, real_data, X_data, num_g=100, fig_name=None):
        X_data = X_data.to(self.device)
        if fig_name is None:
            gmodel, dmodel = self.g_model, self.g_model
        else:
            gmodel, dmodel = self.best_G, self.best_D
        fig = plt.figure(figsize=(5, 5))
        # generate parameters
        noise = torch.normal(mean=torch.zeros([num_g, self.noise_dim]), std=1.0).to(self.device)
        p_output = self.modify_gen(gmodel(noise))
        t_pred = torch.linspace(0, 20, 101).view(-1,1) # for trajectories
        for i, mode in enumerate(p_output.cpu().detach()):  # Assuming A_list_ap is your list of parameter sets
            y_values = self.solution(torch.concat([t_pred]+[p*torch.ones_like(t_pred) for p in mode],-1)).view(-1,1)
            plt.plot(t_pred, y_values.numpy().reshape(-1), linewidth=0.3, color='red', alpha=0.5)
        plt.scatter(X_data[:,0].detach().cpu().numpy(), X_data[:,1].detach().cpu().numpy(), label='True Data', color='k', s=20)    
        # Set plot properties
        plt.xlim([self.tmin, self.tmax])
        plt.ylim([0, 1.4])
        plt.xticks(self.t_numpy)
        plt.yticks([0, 0.7, 1.4])
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlabel('Time(Month)', fontsize=18)
        plt.ylabel('Population', fontsize=18)
        plt.title(r'Accumulation A$\beta$'+real_data[-2:], fontsize=20)
        if fig_name is None:
            plt.show()
        else:
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close(fig)