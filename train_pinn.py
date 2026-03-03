# basic
import numpy as np
import torch
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

# problem settings : Exponential, logistic, Lorenz
name_list = ['exp','log','lorenz']

# hyperparameter settings (prms[0]:DE, prms[1]:hyperPINN, prms[2]:WGAN)
if __name__ == '__main__':
    name = name_list[int(input("[ Select Experiments : 1:exp, 2:log, 3:lorenz (default:exp) ] ") or 1)-1]
    prms = get_prms(name)
    # deeponet
    prms[1]['deeponet'] = bool(input('[ Train deeponet? : 1=True, 0=False (default=False) ] ') or 0)
    if prms[1]['deeponet']:
        if name == 'lorenz':
            prms[1]['lr'] = 5e-4
        else:
            prms[1]['lr'] = 1e-4

    # get dataset
    trainer = Trainer(name, prms, device=device)
    sensor, y = trainer.get_data_pinn()

    # prepare models
    # load pre-trained hyperPINNs
    load_params = False # if you want to train hyperPINN from the begining, then this should be False.
    trainer.load_pinn(load_params=load_params, train_nets=True)

    # train hyperPINN
    EPOCHS = prms[1]['num_epochs']
    losses, losses_d, losses_r, test_errors = trainer.train_pinn(sensor, y, EPOCHS=EPOCHS, iter_print=10, ratio=0.1, iter_plot=100, iter_save=1000)
