
import numpy as np

class ModelParameters:
    """
    A container for all the fundamental constants and user-definable
    parameters of a given physical model.
    """
    def __init__(self):
        self.M_pl = 2.435e18          # Planck mass [GeV]
        self.fermi_constant = 1.166e-5   # Fermi constant [GeV^-2]
        self.sin_theta_w_sq = 0.2312     # Sine-squared of the Weinberg angle

        # HNL Model Parameters
        self.m_N = 0.1 # HNL Mass in GeV (default)
        
        #ALP Model Parameters
        # Default values taken from Deppisch et al 2024 paper's benchmarks, these are changed on the fly if needed
        self.m_a = 1e-6 # ALP mass in GeV (1 keV)
        self.f_a = 1e3  # ALP decay constant in GeV (1 TeV)

        # CKM Matrix
        self.V_ij = np.array([[0.974, 0.225, 0.004],
                              [0.225, 0.973, 0.041],
                              [0.009, 0.040, 0.999]])
        
        # PMNS-like mixing matrix for HNLs, adjust these as required for the chosen simulation. 
        self.U_matrix = np.array([[1e-10, 0, 0],
                                  [0,0, 0],
                                  [0,0, 0]])
        
        
