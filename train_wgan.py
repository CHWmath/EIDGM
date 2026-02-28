# basic
import numpy as np
import torch
import matplotlib.pyplot as plt
# custom
from _config import get_prms
from _trainer import Trainer

# device
device_num = int(input("[ Select GPU (default:0) ] ") or 0)
device = torch.device("cuda:"+str(device_num) if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# experiments
name_list = ['exp','log','lorenz']
dist_types = ['uni','bi','tri']
real_datas = [None,'Abeta40','Abeta42']

if __name__ == '__main__':
    # problem settings : exponential, logistic, target cell-limited 
    name = name_list[int(input("[ Select Experiments : 1:exp, 2:log, 3:lorenz (default:exp) ] ") or 1)-1] # 'exp', 'log', 'virus'
    dist_type = dist_types[int(input('[ Insert the number of modes : 1=uni, 2=bi, 3=tri (default=tri) ] ') or 3)-1] # 'uni', 'bi', 'tri'

    # hyperparameter settings (prms[0]:DE, prms[1]:hyperPINN, prms[2]:WGAN)
    prms = get_prms(name, dist_type=dist_type) # DE / hyperPINN / WGAN settings
    prms[1]['deeponet'] = bool(input('[ Use deeponet? : 1=True, 0=False (default=False) ] ') or 0)

    # get dataset : if you want real data
    real_data_num = int(input('[ Want to use real data? : 1=None, 2=Abeta40, 3=Abeta42 (default=None) ] ') or 1) # None, 'Abeta40', 'Abeta42'
    real_data = real_datas[real_data_num-1]
    trainer = Trainer(name, prms, device=device)
    modes_noisy, X_data, X_data_onehot = trainer.get_data_gan(real_data=real_data)
    
    # prepare models
    # load pre-trained hyperPINNs
    trainer.load_pinn()

    # load WGANs
    load_params = False # if you want to train WGAN from the begining, then this should be False.
    trainer.load_gan(load_params=load_params)

    # train WGAN
    EPOCHS = 100000
    total_lossG, total_lossD, w_dists = trainer.train_gan(modes_noisy, X_data.to(device), X_data_onehot.to(device), \
                                                          EPOCHS=EPOCHS, iter_plot=100000, iter_save=1000)