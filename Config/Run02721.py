import math

class grid_params:
    Nx= 512
    Ny= 512
    Lx= 2*math.pi
    Ly= 2*math.pi
    
class time_params:
    dt= 5e-4
    T = 200001 *dt
    save_int=1024 #(Frequency of .np saves and plots)
    
class pde_params:
    mu = 2e-2 #(Linear drag)
    nu = 1.025e-4  #(Viscosity coefficient)
    B = 2.5  #(Beta parameter)
    nv = 1 #(Hyperviscous order)
    
class ic_params:
    option = 1 #(Modify if other initializes are developed)
    energy= 0.01
    wavenumbers= [3.0, 5.0]
    seed= 495
       
class forcing_params:
    option = 1 # 1 is cos forcing
    A=-1/10 # A cos(B x+ Ct) + D (cos E y + Ft)
    B=2
    C=0
    D=1/10
    E=2
    F=0
            
class params:
    grid = grid_params
    time = time_params
    pde = pde_params
    ic = ic_params
    forcing = forcing_params
    run_number = 2721