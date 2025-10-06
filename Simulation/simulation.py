import torch
import numpy as np
import math
from tqdm import tqdm
from Operators.spectral_conversion import to_physical, to_spectral, dealias
from Time_marching.imex_schemes import backward_euler, CN2, AB2


class Simulation:
    def __init__(self, grid, pde_params, spectral_derivative, linear_operator, nonlinear_operator, initial_condition, time_params,
                 forcing_params, forcing=None):
        """
        Initialize the simulation parameters.
        """
        self.grid = grid
        self.device = grid.device
        self.params = pde_params
        self.spectral_derivative = spectral_derivative
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.initial_condition = initial_condition # In spectral space
        self.forcing = forcing
        self.dt = time_params.dt
        self.T = time_params.T
        self.save_interval=time_params.save_int
        self.forcing_params = forcing_params
        self.Nx, self.Ny = grid.Nx, grid.Ny  # Grid resolution
        self.steps = int(self.T / self.dt)  # Number of time steps
        self.q_sol = torch.zeros([self.Nx, self.Ny, int(self.steps/self.save_interval)+1,4]) # Initialize solution array for all save times
        # Assignment of ICs for spectral fields
        qh_temp,ph_temp, uh_temp,vh_temp  = self.initial_condition # Vorticity, Streamfunction, u velocity, v velocity

        # Convert spectral IC fields for physical space and store in solution array
        self.q_sol[:, :, 0,0] = to_physical(qh_temp)
        self.q_sol[:, :, 0,1] = to_physical(ph_temp) 
        self.q_sol[:, :, 0,2] = to_physical(uh_temp) 
        self.q_sol[:, :, 0,3] = to_physical(vh_temp) 
        self.t0=0.0
        
    def time_step(self):
        """Perform the time-stepping loop."""
        dt = self.dt

        for it_count in range(self.steps - 1):
            
            # Handle initial conditions
            if it_count == 0:
                q_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,0].squeeze()).cuda() #Vorticity
                p_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,1].squeeze()).cuda() #Streamfunction
                u_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,2].squeeze()).cuda() # u velocity
                v_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,3].squeeze()).cuda() # v velocity
                
                u_sol_h_IC = u_sol_h_1[0,0]
                v_sol_h_IC = v_sol_h_1[0,0]
                
            else:
                q_sol_h_1 = ans
                p_sol_h_1 = -q_sol_h_1*self.spectral_derivative.irsq
                u_sol_h_1 = -1j*self.spectral_derivative.ky*p_sol_h_1
                v_sol_h_1 = 1j*self.spectral_derivative.kr*p_sol_h_1
                ## Re-set ICs in u and v (background flow)
                u_sol_h_1[0,0] = u_sol_h_IC
                v_sol_h_1[0,0] = v_sol_h_IC
                
            # Initialize source term
            source = q_sol_h_1
              
            # Compute nonlinear operators (only Jacobian)
            nlo_jacobian_1=self.nonlinear_operator.jacobian_pq(q_sol_h_1, p_sol_h_1, u_sol_h_1, v_sol_h_1)

            # Time-stepping schemes for nonlinear terms
            if it_count == 0:
                source_jacobian = backward_euler(nlo_jacobian_1, dt)
                if self.forcing:
                    term_forcing1 = self.forcing(self.grid,self.spectral_derivative,self.forcing_params,self.t0+it_count*dt)
                    source_forcing = backward_euler(term_forcing1, dt)
                    
            else:
                source_jacobian = AB2(nlo_jacobian_1, nlo_jacobian_2, dt)
                if self.forcing:
                    term_forcing1 = self.forcing(self.grid,self.spectral_derivative,self.forcing_params,self.t0+it_count*dt)
                    source_forcing = AB2(term_forcing1,term_forcing2, dt)
                    
            # Compute linear terms
            source_lin, op_lin = CN2(self.linear_operator,q_sol_h_1,dt)         

            # Update the source term
            source = source + source_lin + source_jacobian
            
            if self.forcing:
                source = source + source_forcing

            # Apply the linear operator inversion
            operator = (1 - op_lin).cuda()
            ans = source / operator
            
            # Store term 2 for AB and CN for next timestep
            nlo_jacobian_2=nlo_jacobian_1
            if self.forcing:
                term_forcing2 = term_forcing1

            # Convert back to physical space and store the result for every save interval
            if (it_count+1) % self.save_interval == 0:
                save_index = (it_count + 1) // self.save_interval
                qh_temp = ans
                ph_temp = -qh_temp*self.spectral_derivative.irsq
                uh_temp = -1j*self.spectral_derivative.ky*ph_temp
                vh_temp = 1j*self.spectral_derivative.kr*ph_temp
                ## Re-set ICs in u and v (background flow)
                uh_temp[0,0] = u_sol_h_IC
                vh_temp[0,0] = v_sol_h_IC
                self.q_sol[:, :, save_index,0] = to_physical(qh_temp)
                self.q_sol[:, :, save_index,1] = to_physical(ph_temp)
                self.q_sol[:, :, save_index,2] = to_physical(uh_temp)
                self.q_sol[:, :, save_index,3] = to_physical(vh_temp)
                

    def run(self):
        """Run the full simulation."""
        self.time_step()
        return self.q_sol  # Return the solution array
