import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import datetime
import numpy as np
import IPython.display as display
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List
from tqdm.notebook import tqdm
import torch.optim as optim
import dataclasses
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os
import sys
import warnings
import importlib

warnings.filterwarnings("ignore", category=UserWarning)
    
sys.path.append('/gdata/projects/ml_scope/Turbulence/QG_V0001/Src')

# Create an argument parser
parser = argparse.ArgumentParser(description='Parse args')

# Add the run_num argument to specify the run number
parser.add_argument('--run_num', type=int, help='Run number for configuration file', required=True)

# Parse the command-line arguments
args = parser.parse_args()
run_number = args.run_num

## Load config file and print all parameters to log file
config_module = f'Config.Run{run_number:05d}'

from Utils.utils import print_config

try:
    config = importlib.import_module(config_module)
    print(f"Successfully loaded configuration from {config_module}")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print_config(config.params) 
except ModuleNotFoundError:
    print(f"Configuration file for run {run_number} not found.")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    raise

## Set-up grid
from Grid.grid import Grid
grid_DNS=Grid(config.params.grid.Lx,config.params.grid.Ly,config.params.grid.Nx,config.params.grid.Ny)

## Set-up spectral derivatives and operators
from Operators.operators import SpectralDerivatives, LinearOperator, NonlinearOperator
spec_deriv_DNS=SpectralDerivatives(grid_DNS)
linop_DNS = LinearOperator(spec_deriv_DNS,config.params.pde)
nonlinop_DNS = NonlinearOperator(spec_deriv_DNS,config.params.pde)

print(f"Successfully loaded derivatives and operators")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Set-up initial conditions
config.params.ic.option = getattr(config.params.ic, 'option', 1) 
if config.params.ic.option == 1:
    print(f"Using wavenumber ICs")
    from Initial_forcing.ics import init_randn
    init_conds_DNS =  init_randn(config.params.ic.energy, config.params.ic.wavenumbers, grid_DNS, spec_deriv_DNS, config.params.ic.seed)
else:
    raise ValueError("Invalid IC option. Check config.")   

print(f"Successfully created initial conditions")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Set-up forcing
if config.params.forcing.option ==1:
    print(f"Using cos forcing")
    from Initial_forcing.forcing import cos_forcing
    forcing_DNS = cos_forcing
elif config.params.forcing.option ==0:
    forcing_DNS = None
    config.params.forcing=None
else:
    raise ValueError("Invalid forcing option. Check config.")
        
## Run simulation
print(f"Simulation started")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

from Simulation.simulation import Simulation

sim_DNS = Simulation(grid_DNS,config.params.pde,spec_deriv_DNS,linop_DNS,nonlinop_DNS,init_conds_DNS,config.params.time,
                     config.params.forcing,forcing_DNS)

solution_field = sim_DNS.run()

print(f"Simulation completed successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Save numpy files
from Utils.utils import save_file
save_file(grid_DNS,spec_deriv_DNS,solution_field,run_number,config.params.time)

print(f"Simulation np & pt files and plots saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Save spectra
from Utils.utils import save_spectrum_plots
save_spectrum_plots(solution_field,spec_deriv_DNS,run_number,config.params.time)

print(f"Simulation spectrum plots saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
