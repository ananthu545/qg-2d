import torch
import numpy as np
import math

from Operators.spectral_conversion import to_physical, to_spectral, dealias

### Set up spectral derivatives (first and second derivatives)
# Conventions: first derivative: + 1j*k, second derivative: -k**2

class SpectralDerivatives:
    def __init__(self, grid):
        self.grid = grid
        self.device = grid.device

        # Number of wavenumber components (half of real grid in x-direction)
        self.dk = int(grid.Nx / 2 + 1)

        ## Compute wavenumbers for first derivatives
        # Derivative in y
        self.ky = torch.reshape((torch.fft.fftfreq(grid.Ny, grid.Ly / (grid.Ny * 2 * math.pi))), 
            (grid.Ny, 1)
        ).to(self.device) 
        
        # Derivative in x
        self.kr = torch.reshape((torch.fft.rfftfreq(grid.Nx, grid.Lx / (grid.Nx * 2 * math.pi))), 
            (1, self.dk)
        ).to(self.device)

        # Squared wavenumbers (for second derivatives)
        self.krsq = self.kr**2 + self.ky**2  

        # Inverse squared wavenumbers (and handling zero division)
        self.irsq = 1.0/self.krsq
        self.irsq[0,0] = 0.0 #

    def to(self, device):
        """ Move spectral operator tensors to another device. """
        self.device = device
        self.ky = self.ky.to(device)
        self.kr = self.kr.to(device)
        self.krsq = self.krsq.to(device)
        self.irsq = self.irsq.to(device)

    def __repr__(self):
        return (f"SpectralDerivatives(Nx={self.grid.Nx}, Ny={self.grid.Ny}, dk={self.dk}, "
                f"Lx={self.grid.Lx:.4f}, Ly={self.grid.Ly:.4f}, device={self.device})")

### Set up linear operator
class LinearOperator:
    def __init__(self, spectral_derivative, params):
        self.spectral_derivative = spectral_derivative
        self.params = params
        self.device = spectral_derivative.device
        
        # Precompute the linear term and store it
        self.Lc = self.linear_term()

    def linear_term(self):
        # Extracting parameters from the params object
        nu = self.params.nu
        mu = self.params.mu
        B = self.params.B
        
        # Calculate the linear term 
        # first term is diffusion: nu del^2 omega
        # then bottom drag: - mu omega
        # then Coriolis with beta term: - beta d psi/ dx (where omega = del^2 psi)
        Lc = -nu * self.spectral_derivative.krsq - mu + 1j * torch.tensor(B).to(self.device) * self.spectral_derivative.kr * self.spectral_derivative.irsq
        return Lc

    def apply(self, input_field):
        """
        Multiplies the linear oeprator with the input_tensor.
        Assumes the input_field is on the same device as the LinearOperator.
        """
        # Ensure input_tensor is on the same device
        input_field = input_field.to(self.device)

        # Multiply the linear term by input_tensor
        out = self.Lc * input_field
        
        return out

    def __repr__(self):
        return (f"LinearOperator(nu={self.params.nu}, mu={self.params.mu}, "
                f"B={self.params.B}, device={self.device})")

### Set up nonlinear operator
class NonlinearOperator:
    def __init__(self, spectral_derivative, params):
        self.spectral_derivative = spectral_derivative
        self.params = params
        self.device = spectral_derivative.device

    def jacobian_pq(self, input_field_q,input_field_p,input_field_u,input_field_v):
        """
        Computes the Jacobian of q (vorticity) and p (streamfunction) in spectral space (h).
        """
        # In spectral space
        qh= input_field_q.clone()
        ph= input_field_p.clone()
        uh= input_field_u.clone() 
        vh= input_field_v.clone() 

        # In physical space
        q=to_physical(qh)
        u=to_physical(uh)
        v=to_physical(vh)

        # Calculate jacobian
        uq = u*q
        vq = v*q

        uqh=to_spectral(uq)
        vqh=to_spectral(vq)
        
        #[-d/dx (u*q) - d/dy (v*q)]
        out = -1j*self.spectral_derivative.kr*uqh - 1j*self.spectral_derivative.ky*vqh

        return dealias(out,self.spectral_derivative,1/3)    
        
    def __repr__(self):
        return (f"NonlinearOperator(penalty_coeff={self.params.penalty_coeff}, "
            f"device={self.device})")
    