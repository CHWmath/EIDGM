# EIDGM

This repository contains the code for the paper : Estimation of System Parameters Including Repeated Cross-Sectional Data Through Emulator-Informed Deep Generative Model by Hyunwoo Cho, Sung Woong Cho, Hyeontae Jo, and Hyung Ju Hwang. 

This repository provides the implementation of the Emulator-Informed Deep Generative Model (EIDGM) and the scripts required to reproduce the experiments presented in the paper.


## Visualization of trajectory and parameter estimates

For a direct verification of reproducibility for **Figures 2–4**, we provide Jupyter notebooks that visualize the trajectory reconstruction and parameter estimation results obtained by EIDGM.

The following notebooks are included:

- `results_exp.ipynb` — exponential growth model
- `results_log.ipynb` — logistic growth model
- `results_lorenz.ipynb` — Lorenz system  

Each notebook loads the trained models and reproduces the trajectory and parameter distribution plots reported in the paper.


## Setup instructions

To reproduce the experiments, first clone the repository:

```bash
git clone https://github.com/CHWmath/EIDGM.git
cd EIDGM
```

Then install the required Python packages:

```bash
pip install -r requirements.txt
```


## Running EIDGM training

To train EIDGM and generate experimental results, run

```bash
python train_wgan.py
```

You will be prompted to select the following experiment configurations:

### Experiment type
- `exponential`
- `logistic`
- `lorenz`

### The number of modes in the true parameter posterior
- `uni`
- `bi`
- `tri`

### Use DeepONet emulator
Set this option to 'False' to train EIDGM, which uses the emulator as a HyperPINN.
- `True`
- `False` (default)

### Dataset
Set this option to `None` to train the model on synthetic data.
- `None` (default)
- `Abeta40`
- `Abeta42`

During training, intermediate trajectory reconstructions and parameter estimation results are saved in:

```
EIDGM/save/figures
```


## Data source

The real datasets, amyloid-β 40 (Aβ40) and amyloid-β 42 (Aβ42), used in this study were obtained from the AD Knowledge Portal (https://adknowledgeportal.synapse.org/
).
