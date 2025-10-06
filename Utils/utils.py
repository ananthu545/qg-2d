import importlib
import numpy as np
import math
import matplotlib.pyplot as plt
from Plotting.plots import vorticity_plots, spectrum_plot, spectrum
from Operators.spectral_conversion import to_physical, to_spectral, dealias
import os
import torch
import pickle
import shutil

def print_config(obj, indent=0):
    """Recursively prints all attributes of a class or object."""
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):  # If it's a simple value, print it
        print(" " * indent + str(obj))
        return

    for attr_name in dir(obj):
        if attr_name.startswith("__"):  # Skip special attributes
            continue

        attr_value = getattr(obj, attr_name)

        if isinstance(attr_value, type):  # If it's a class, recurse into it
            print(" " * indent + f"{attr_name}:")
            print_config(attr_value, indent + 4)
        elif not callable(attr_value):  # Print regular attributes
            print(" " * indent + f"{attr_name} = {attr_value}")

def save_file(grid,spectral_derivative,solution_field, run_number, time_params):
    base_dir = '/gdata/projects/ml_scope/Turbulence/QG_V0001/Results'
    
    ## Save fields (q,p,u,v)
    save_dir = os.path.join(base_dir, f'Run{run_number:05d}')
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'fields_Run{run_number:05d}.npy'
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, solution_field.cpu().numpy())   
    
    ### Move all code files
    # Move config file to results folder
    source_config_file = f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Src/Config/Run{run_number:05d}.py"
    destination_directory = f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/Run{run_number:05d}/Code"
    os.makedirs(destination_directory, exist_ok=True)

    if os.path.exists(source_config_file):
        destination_config_file = os.path.join(destination_directory, f"Run{run_number:05d}.py")
        shutil.copy(source_config_file, destination_config_file)
    
    # Move remaining code
    source_directory = f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Src"

    # List of folders to ignore
    folders_to_ignore = [
        f"/gdata/projects/ml_scope/Turbulence/QG_V0001/Src/Config"]
    # Move the configuration file to the destination directory
    for root, dirs, files in os.walk(source_directory):
        # Determine the relative path and the destination path
        relative_path = os.path.relpath(root, source_directory)
        destination_path = os.path.join(destination_directory, relative_path)
        # Check if the current directory should be ignored
        if any(ignored_dir in root for ignored_dir in folders_to_ignore):
            continue
        # Create the directory structure in the destination
        os.makedirs(destination_path, exist_ok=True)
        # Copy the files
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_path, file)
            shutil.copy(source_file, destination_file)
    
    
    base_dir = '/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/'
    save_dir = os.path.join(base_dir, f'Run{run_number:05d}', 'Plots')
    os.makedirs(save_dir, exist_ok=True)
    
    for timestep in range(solution_field.shape[2]):
        fig, ax = vorticity_plots(grid,solution_field, timestep, time_params)

        # Save the plot as an image 
        plot_file_name = f'vorticity_Run{run_number:05d}_t_{timestep*time_params.save_int:06d}.png'
        plot_file_path = os.path.join(save_dir, plot_file_name)
        fig.savefig(plot_file_path,bbox_inches='tight',dpi=300)  # Save the plot as an image
        plt.close(fig)  # Close the figure to free memory
    
    
def save_spectrum_plots(solution_field, spectral_derivative, run_number, time_params):
    base_dir = '/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/'
    save_dir = os.path.join(base_dir, f'Run{run_number:05d}', 'Spectrum')
    os.makedirs(save_dir, exist_ok=True)
    
    for timestep in range(solution_field.shape[2]):
        
        qh_sol=to_spectral(solution_field[:,:,timestep,0].squeeze()).cuda() # Extract just vorticity field
        ph_sol=-qh_sol*spectral_derivative.irsq
        uh_sol= -1j * spectral_derivative.ky * ph_sol # Get u velocity
        vh_sol = 1j * spectral_derivative.kr * ph_sol # Get v velocity
        
        z = torch.abs(qh_sol)**2 # Get enstrophy
        e = torch.abs(uh_sol)**2 + torch.abs(vh_sol)**2 # Get kinetic energy
        
        k,[ek,zk]=spectrum([e,z],spectral_derivative)

        fig, ax =  spectrum_plot(k, ek, zk, timestep, time_params)

        # Save the plot as an image 
        plot_file_name = f'spectrum_Run{run_number:05d}_t_{timestep*time_params.save_int:06d}.png'
        plot_file_path = os.path.join(save_dir, plot_file_name)
        fig.savefig(plot_file_path,bbox_inches='tight',dpi=300)  # Save the plot as an image
        plt.close(fig)  # Close the figure to free memory
    