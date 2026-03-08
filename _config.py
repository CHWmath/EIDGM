import numpy as np

# lorenz model
def model_lorenz(t, y, sigma, rho, beta):
    X, Y, Z = y
    dXdt = sigma*(Y - X)
    dYdt = X * (rho - Z) - Y
    dZdt = X*Y - beta*Z
    return [dXdt, dYdt, dZdt]
    
# get hyperparameters 
def get_prms(name, dist_type=[], p_range=[], modes=[]):
    if name == 'exp':
        p_range = [[0.5, 3.5]]
        if dist_type == 'uni':
            modes = np.array([[2]])
        if dist_type == 'bi':
            modes = np.array([[1], [3]])
        if dist_type == 'tri':
            modes = np.array([[1], [2], [3]])
        prms_DE = {'outputs':['Y'], 'param_names':['r'], 'num_p':1, 'num_eq':1, 't_range':[0,1], 'init':[1], 'p_range':p_range}
        prms_PINN = {'width_hyp':64, 'depth_hyp':4, 'width_tru':32, 'num_chunks_in':0, 'act':'tanh', 'lr':5e-5, 'Nt':100, 'Np':100, 'batch_size':10000, 'init_noise_scale':0.2, 'beta':1e-2, 'tol':5e-4, 'sc':False, 'deeponet':False, 'num_epochs':10000}
        prms_GAN = {'num_bins':5, 'plow':0.97, 'phigh':1.03, 'num_noised':12, 'modes':modes, 'num_gen_ratio':1, 'num_gen_plot':1000, 'data_noise_scale':0.0, 'init_noise_scale_wgan':0.0, 'num_cut':0, 'noise_dim':16, 'width':64, 'depth':4,'lr_G':1e-4, 'lr_D':1e-4, 'ratio_G':1, 'num_gen_traj':-1}
        
    if name == 'log':
        p_range = [[1, 5], [0.2, 1.5]]
        if dist_type == 'uni':
            modes = np.array([[2.8, 1.0]])
        if dist_type == 'bi':
            modes = np.array([[1.6,1.4], [4.0,0.6]])
        if dist_type == 'tri':
            modes = np.array([[1.6,0.6], [4.0,0.9], [2.0,1.3]])
        prms_DE = {'outputs':['Y'], 'param_names':['r','K'], 'num_p':2, 'num_eq':1, 't_range':[0,20], 'init':[1e-5], 'p_range':p_range}
        prms_PINN = {'width_hyp':64, 'depth_hyp':4, 'width_tru':32, 'num_chunks_in':0, 'act':'tanh', 'lr':5e-5, 'Nt':100, 'Np':200, 'batch_size':10000, 'init_noise_scale':0.2, 'beta':1e-2, 'tol':5e-4, 'sc':False, 'deeponet':False, 'num_epochs':10000}
        prms_GAN = {'num_bins':5, 'plow':0.97, 'phigh':1.03, 'num_noised':12, 'modes':modes, 'num_gen_ratio':1, 'num_gen_plot':1000, 'data_noise_scale':0.0, 'init_noise_scale_wgan':0.0, 'num_cut':0, 'noise_dim':16, 'width':64, 'depth':4,'lr_G':1e-4, 'lr_D':1e-4, 'ratio_G':1, 'num_gen_traj':-1}
    
    if name == 'lorenz':
        mode0 = np.array([10, 22.5, 5/3])
        multiplier1 = np.array([1.05, 0.8, 3/5])
        multiplier2 = np.array([0.95, 1.2, 1.0])
        multiplier3 = np.array([1.0, 1.1, 7/5])
        p_range = [[9,11], [0,28], [2/3,8/3]]
        if dist_type == 'uni':
            modes = [list(mode0*multiplier2)]
        if dist_type == 'bi':
            modes = [list(mode0*multiplier1), list(mode0*multiplier3)]
        if dist_type == 'tri':
            modes = [list(mode0*multiplier1), list(mode0*multiplier2), list(mode0*multiplier3)]
        prms_DE = {'outputs':['X', 'Y', 'Z'], 'param_names':['sigma','rho','beta'], 'num_p':3, 'num_eq':3, 't_range':[0,1], 'init':[4.67, 5.49, 9.06], 'p_range':p_range}
        prms_PINN = {'width_hyp':64, 'depth_hyp':4, 'width_tru':64, 'num_chunks_in':0, 'act':'tanh', 'lr':5e-5, 'Nt':100, 'Np':1000, 'batch_size':10000, 'init_noise_scale':0.2, 'beta':0.0, 'tol':5e-4, 'sc':False, 'deeponet':False, 'num_epochs':10000}
        prms_GAN = {'num_bins':9, 'plow':0.99, 'phigh':1.01, 'num_noised':12, 'modes':modes, 'num_gen_ratio':3, 'num_gen_plot':500, 'data_noise_scale':0.0, 'init_noise_scale_wgan':0.0, 'num_cut':0, 'noise_dim':32, 'width':128, 'depth':4,'lr_G':1e-4, 'lr_D':1e-4, 'ratio_G':1, 'num_gen_traj':-1}
        
    return [prms_DE, prms_PINN, prms_GAN]