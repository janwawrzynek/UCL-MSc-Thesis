
import numpy as np
from scipy.integrate import quad
from scipy.special import zeta, kn
from typing import TYPE_CHECKING, List

from .Model_Parameters import ModelParameters
from .Rel_degrees_of_freedom import RelDegreesOfFreedom
from .Particles import (
    Particle, SterileNeutrino, Electron, Muon, Tau, LightNeutrino, Quark, ALP,
    PiPlus,KPlus,DPlus,DStrangePlus,BPlus,BCharm,PiZero,Eta,EtaPrime,EtaCharmed,RhoPlus,DStarPlus,DstarstrangePlus,RhoZero,Omega,Phi,JPsi 
)

# Use TYPE_CHECKING to avoid circular imports at runtime, fixes an error that I had. 
if TYPE_CHECKING:
    from .Particles import SterileNeutrino, Charged_Lepton

pi = np.pi
riemann_zeta_3 = 1.20205 # # Riemann zeta function at 3, used in equilibrium density calculations


z_init = 0.3675    # from t≈1e-5 s (see Deppisch et al 2024)
z_final = 116.3    # from t=1 s (see Deppisch et al 2024)
z_range = np.logspace(np.log10(z_init), np.log10(z_final), 1000)  # z = m_N / T

class Model:
    """
    Contains the physical model, including all parameters,
    constants, fundamental state functions, and particle creation.
    """
    def __init__(self, params: ModelParameters, g_star_csv_path: str):
        self.params = params
        self.dof_calculator = RelDegreesOfFreedom(
            csv_path=g_star_csv_path, 
            m_N=self.params.m_N
        )

    # Creating instances of each particle species. 
    def create_sterile_neutrino(self, flavor: str) -> SterileNeutrino:
        return SterileNeutrino(flavor, mass=self.params.m_N)
    
    def create_electron(self) -> Electron: return Electron()
    def create_muon(self) -> Muon: return Muon()
    def create_tau(self) -> Tau: return Tau()
    
    def create_quark(self, flavor: str) -> Quark: return Quark(flavor)
    
    def create_alp(self) -> ALP:
        
        return ALP(mass=self.params.m_a,
                decay_const=self.params.f_a)

    # Creating instances of all of the particles 
    # Charged Pseudoscalar Mesons
    def create_pi_plus(self) -> PiPlus: return PiPlus()
    def create_k_plus(self) -> KPlus: return KPlus()
    def create_d_plus(self) -> DPlus: return DPlus()
    def create_d_strange_plus(self) -> DStrangePlus: return DStrangePlus()
    def create_b_plus(self) -> BPlus: return BPlus()
    def create_b_charm(self) -> BCharm: return BCharm()
    # Neutral Pseudoscalar Mesons
    def create_pi_zero(self) -> PiZero: return PiZero()
    def create_eta(self) -> Eta: return Eta()
    def create_eta_prime(self) -> EtaPrime: return EtaPrime()
    def create_eta_charmed(self) -> EtaCharmed: return EtaCharmed()
    # Charged Vector Mesons
    def create_rho_plus(self) -> RhoPlus: return RhoPlus()
    def create_d_star_plus(self) -> DStarPlus: return DStarPlus()
    def create_d_star_strange_plus(self) -> DstarstrangePlus: return DstarstrangePlus()
    # Neutral Vector Mesons
    def create_rho_zero(self) -> RhoZero: return RhoZero()
    def create_omega(self) -> Omega: return Omega()
    def create_phi(self) -> Phi: return Phi()
    def create_jpsi(self) -> JPsi: return JPsi()

    #QCD Correction implementation
    def alpha_s(self, Q):
        Lambda_QCD = 0.23
        nf = 3 
        if Q > 1.3: nf = 4 #charm threshold
        if Q > 4.5: nf = 5 #bottom threshold
        beta0 = (33 -2*nf)/(12* pi)
        if Q <= Lambda_QCD: return np.inf
        return 1 / (beta0 * np.log(Q**2 / Lambda_QCD**2))

    def get_qcd_correction(self, M_N):
        a_s = self.alpha_s(M_N)
        return (a_s/pi) + 5.2*(a_s/pi)**2 + 26.4*(a_s/pi)**3

    # Cosmology functions
    def entropy_density(self, z: float) -> float:
        g_star = self.get_g_star(z)
        m_N = self.params.m_N
        return ((2 * (pi**2)) / 45) * g_star * ((m_N / z)**3)
    
    def hubble(self, z: float) -> float:
        M_pl = self.params.M_pl
        m_N = self.params.m_N
        g_star = self.get_g_star(z)
        T = m_N / z
        rho_rad = (pi**2 / 30) * g_star * T**4
        H_squared = (8 * pi / (3 * M_pl**2)) * rho_rad
        return np.sqrt(H_squared)

    def get_g_star(self, z: float) -> float:
        return self.dof_calculator.g_star(z)
    #Kinematic check for particle decays
    def check_kinematics(self, incoming_particle: Particle, outgoing_particles: list[Particle]):
        m_incoming = incoming_particle.mass
        m_outgoing_total = sum(p.mass for p in outgoing_particles)
        if m_outgoing_total >= m_incoming:
            raise ValueError(
                f"Kinematic condition not satisfied: Sum of outgoing masses "
                f"({m_outgoing_total:.4f} GeV) must be less than the incoming mass "
                f"({m_incoming:.4f} GeV)."
            )

    def get_U_alpha(self, sterile_neutrino: 'SterileNeutrino', charged_lepton: 'Charged_Lepton') -> float:
        flavor_mapping = {'n1': 0, 'n2': 1, 'n3': 2, 'electron': 0, 'muon': 1, 'tau': 2}
        sterile_index = flavor_mapping[sterile_neutrino.flavor]  #It takes the incoming sterile_neutrino object and gets its flavor string (e.g., "n1"). It then uses this string as a key to look up the corresponding numerical index from the flavor_mapping.
        lepton_name = charged_lepton.__class__.__name__.lower()  #gets the name of that class as a string (e.g., "Electron")
        if lepton_name not in flavor_mapping:
            raise KeyError(f"Charged lepton '{lepton_name}' not found in flavor mapping.")
        charged_lepton_index = flavor_mapping[lepton_name]  #uses the lowercase lepton name (e.g., "electron") to look up the corresponding numerical index from the flavor_mapping.
        return self.params.U_matrix[sterile_index][charged_lepton_index]

    def V_ij_for_quark(self, i: Quark, j: Quark) -> float:
        up_type = {'up': 0, 'charm': 1, 'top': 2}
        down_type = {'down': 0, 'strange': 1, 'bottom': 2}
        i_flavor, j_flavor = i.flavor.lower(), j.flavor.lower()
        if i_flavor in up_type and j_flavor in down_type:
            i_idx, j_idx = up_type[i_flavor], down_type[j_flavor]
        elif i_flavor in down_type and j_flavor in up_type:
            i_idx, j_idx = down_type[i_flavor], up_type[j_flavor]
        else:
            raise ValueError("One quark must be up-type (u,c,t) and one must be down-type (d,s,b)")
        return abs(self.params.V_ij[i_idx, j_idx])

    def create_light_neutrino(self,
                            sterile_flavor: str,
                            light_flavor:   str
                            ) -> LightNeutrino:
        N = self.create_sterile_neutrino(sterile_flavor)
        lep = {'nu_e': Electron(),
            'nu_mu': Muon(),
            'nu_tau': Tau()}[light_flavor]
        U = self.get_U_alpha(N, lep)
        ν = LightNeutrino(flavor=light_flavor,
                        model=self,
                        sterile=N)
        ν.mass = abs(U)**2  * N.mass
        return ν

#Below I will build up the cosmology sections of this Model class. First I will be simulating the thermally averaged decays of the sterile neutrinos over z (mass temperature ratio)
    def equilibrium_species_density(self, g: float, mass: float, z, is_fermion: bool = False):
        """
        Equilibrium number density n_eq for species of mass `mass` and degeneracy `g`,
        using relativistic formula for z <~ 1 and Maxwell-Boltzmann for z > 1.
        z = m / T.
        """
        z_arr = np.asarray(z)
        T = mass / z_arr  # temperature

        # Relativistic limit (z small): 
        prefactor_rel = ((3/4 if is_fermion else 1.0) * g * zeta(3) / (np.pi**2))
        n_rel = prefactor_rel * T**3

        # Nonrelativistic Maxwell-Boltzmann
        n_nonrel = g * (mass * T / (2 * np.pi))**1.5 * np.exp(-z_arr)

        # For scalar or array: choose nonrel when z > 1, relativistic when z <= 1
        if np.isscalar(z):
            return n_nonrel if z > 1 else n_rel
        else:
            return np.where(z_arr > 1, n_nonrel, n_rel)
        
        # n_N_eq = equilibrium_species_density(2,0.1,z_init)    
        # n_a_eq = equilibrium_species_density(1,m_a,z_init)
        # Y_N_eq = n_N_eq / entropy_density(z_init)
        # Y_a_eq = n_a_eq/entropy_density(z_init)


        

