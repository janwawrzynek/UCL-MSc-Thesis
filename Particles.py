
import numpy as np
from typing import List, TYPE_CHECKING
from scipy.integrate import quad
import itertools

if TYPE_CHECKING:
    from OOP.Model import Model

pi = np.pi


# Base Particle Class
# This is the foundational class for all particles in the simulation.
# It ensures every particle has a 'mass' attribute.
class Particle:
    def __init__(self, mass):
        self.mass = mass
    def __str__(self):
        return f"{self.__class__.__name__}({self.mass:.3f} GeV)"

# Fundamental & Composite Particle Base Classes
# These classes provide a hierarchical structure for different types of particles,
class Lepton(Particle): pass
class Quark(Particle):
    def __init__(self, flavor):
        mass_map = {'up':0.005,'down':0.003,'charm':1.3,'strange':0.1,'top':174,'bottom':4.5}
        super().__init__(mass_map[flavor])
        self.flavor = flavor
        self.is_up_type = flavor in ['up', 'charm', 'top']

class PseudoScalar(Particle): pass
class Hadron(Particle): pass
class Meson(Hadron):
    def __init__(self, mass, quark_content=(), charge=0):
        super().__init__(mass)
        self.quark_content = quark_content  
        self.charge = int(charge)           
        self._validate_charge_and_quarks()
    def is_charged(self) :   #charge helper function to be used later in the code
        return self.charge != 0
    def _validate_charge_and_quarks(self):
        up_types = {"up", "charm", "top"}
        down_types = {"down", "strange", "bottom"}
        if self.charge == 0:
            return
        if not self.quark_content or len(self.quark_content) != 2:
            raise ValueError(f"{self.__class__.__name__}: charged mesons require quark_content=(U, D) type.")
        qU, qD = self.quark_content
        if qU not in up_types or qD not in down_types:
            raise ValueError(f"{self.__class__.__name__}: quark_content must be (up-type, down-type), got {self.quark_content}.")


class PseudoscalarMeson(Meson):
    __slots__ = ("f_decay_const",)
    def __init__(self, mass, f_decay_const, quark_content=(), charge=0):
        super().__init__(mass, quark_content=quark_content, charge=charge)
        self.f_decay_const = f_decay_const

class ChargedPseudoscalarMeson(PseudoscalarMeson):
    def __init__(self, mass, f_decay_const, quark_content):
        super().__init__(mass, f_decay_const, quark_content=quark_content, charge=+1)
        
class NeutralPseudoscalarMeson(PseudoscalarMeson):
    def __init__(self, mass, f_decay_const, quark_content=()):
        super().__init__(mass, f_decay_const, quark_content=quark_content, charge=0)
        # neutrals may leave quark_content empty or same-flavor (e.g. ('charm','charm'))


class VectorMeson(Meson): pass
class ChargedVectorMeson(VectorMeson):
    def __init__(self, mass, g_decay_const, quark_content):
        super().__init__(mass, quark_content=quark_content, charge=+1)
        self.g_decay_const = g_decay_const

class NeutralVectorMeson(VectorMeson):
    def __init__(self, mass, g_decay_const, kappa_factor, quark_content=()):
        super().__init__(mass, quark_content=quark_content, charge=0)
        self.g_decay_const = g_decay_const
        self.kappa_factor = kappa_factor

# Specific Particle Implementations
# These are the concrete particle classes with their specific masses and properties.
# The meson decay constants and quark content are taken from Appendix C of the phenomenology paper.
class Charged_Lepton(Lepton): pass
class Neutrino(Lepton): pass
class ALP(Particle):
    _total_width_cache = {}
    def __init__(self, mass= 1e-6, decay_const=1e3, sterile = None):
        super().__init__(mass)
        self.decay_const = decay_const
        self.sterile = sterile
    def __str__(self):
        return f"ALP(mass={self.mass:.3f} GeV, decay_const={self.decay_const:.3f} GeV)"
    
    def ALP_channels(self, model: 'Model'):
        channels = []
        for flavour in ('nu_e', 'nu_mu', 'nu_tau'):
            v1 = LightNeutrino(flavour, model, self.sterile)
            v2 = LightNeutrino(flavour, model, self.sterile)
            channels.append([v1,v2])
        return channels
    def calculate_decay_width(self, outgoing_particles: List[Particle], model: 'Model'):
        model.check_kinematics(self, outgoing_particles)
        if not (len(outgoing_particles) == 2 and 
                isinstance(outgoing_particles[0], LightNeutrino) and 
                isinstance (outgoing_particles[1], LightNeutrino)):
            raise TypeError("ALP decay must be into two light neutrinos.")
        nu1,nu2  = outgoing_particles
        if nu1.flavor != nu2.flavor:
            raise TypeError("ALP decay requires neutrinos of the same flavor.")
        m_N = model.params.m_N
        m_a = self.mass
        f_a = self.decay_const
        primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu1.flavor)
        U_alpha = model.get_U_alpha(self.sterile, primary_lepton)

        width = (m_N**2 * m_a * abs(U_alpha)**4) / (2 * pi * f_a**2)
        return width
    
    def get_total_decay_width(self, model: 'Model'):
        cache_key = (self.mass, model.params.m_N, self.decay_const)
        if cache_key in ALP._total_width_cache:
            return ALP._total_width_cache[cache_key]

        total = 0.0
        for channel in self.ALP_channels(model):
            try:
                total += self.calculate_decay_width(channel, model)
            except (TypeError, ValueError):
                 continue
            
        ALP._total_width_cache[cache_key] = total
        return total

class Electron(Charged_Lepton):
    def __init__(self): super().__init__(mass=0.000511)
class Muon(Charged_Lepton):
    def __init__(self): super().__init__(mass=0.105658)
class Tau(Charged_Lepton):
    def __init__(self): super().__init__(mass=1.77686)
# Charged pseudoscalars  ALL Pseudosalars Implemented
class PiPlus(ChargedPseudoscalarMeson):
    def __init__(self): super().__init__(mass=0.1396, f_decay_const=0.1302, quark_content=('up', 'down')) # u d̄ → V_ud 

class KPlus(ChargedPseudoscalarMeson):
    def __init__(self): super().__init__(mass=0.4937, f_decay_const=0.1556, quark_content=('up', 'strange'))  # u s̄ → V_us

class DPlus(ChargedPseudoscalarMeson):
    def __init__(self): super().__init__(mass=1.8694, f_decay_const=0.212, quark_content=('charm', 'down')) # c d̄ → V_cd
class DStrangePlus(ChargedPseudoscalarMeson):
    def __init__(self): super().__init__(mass=1.9683, f_decay_const=0.249, quark_content=('charm', 'strange')) # c s̄ → V_cs
class BPlus(ChargedPseudoscalarMeson):
    def __init__(self): super().__init__(mass=5.279, f_decay_const=0.187, quark_content=('up', 'bottom')) # u b̄ → V_ub

class BCharm(ChargedPseudoscalarMeson):
    def __init__(self): super().__init__(mass=6.2747, f_decay_const=0.434, quark_content=('charm', 'bottom'))  # c b̄ → V_cb


#neutral pseudoscalar mesons
class PiZero(NeutralPseudoscalarMeson):
    def __init__(self): super().__init__(mass=0.1350, f_decay_const=0.130)
class Eta(NeutralPseudoscalarMeson):
    def __init__(self): super().__init__(mass=0.5478, f_decay_const=0.0817)
class EtaPrime(NeutralPseudoscalarMeson):
    def __init__(self): super().__init__(mass=0.9578, f_decay_const= -0.0947)
class EtaCharmed(NeutralPseudoscalarMeson):
    def __init__(self): super().__init__(mass=2.9796, f_decay_const=0.237, quark_content=('charm', 'charm'))
# class KZero(PseudoscalarMeson):
#     def __init__(self): super().__init__(mass=0.4976, f_decay_const=0.1556)   # K0 not included in HNL decays
#Charged Vector Mesons
    
class RhoPlus(ChargedVectorMeson):
    def __init__(self): super().__init__(mass=0.775, g_decay_const=0.162, quark_content=('up', 'down'))  # u d̄ → V_ud
class DStarPlus(ChargedVectorMeson):
    def __init__(self): super().__init__(mass=2.010, g_decay_const=0.535, quark_content=('charm', 'down'))  # c d̄ → V_cd
class DstarstrangePlus(ChargedVectorMeson):
    def __init__(self): super().__init__(mass=2.1121, g_decay_const=0.650, quark_content=('charm', 'strange'))  # c s̄ → V_cs



# Neutral Vector Mesons
class RhoZero(NeutralVectorMeson):
    def __init__(self): super().__init__(mass=0.775, g_decay_const=0.162, kappa_factor=(1 - 2 * 0.2312))
class Omega(NeutralVectorMeson):
     def __init__(self): super().__init__(mass=0.782, g_decay_const=0.153, kappa_factor=(4/3)*(0.2312))
class Phi(NeutralVectorMeson):
    def __init__(self): super().__init__(mass=1.019456, g_decay_const=0.234, kappa_factor=((4/3)*(0.2312) -1), quark_content=('strange', 'strange'))  # s s̄

class JPsi(NeutralVectorMeson):
    def __init__(self): super().__init__(mass=3.096916, g_decay_const=1.29, kappa_factor=(1 - (8/3)*0.2312), quark_content=('charm', 'charm')) # c c̄

# Sterile Neutrino Class
# This class contains all the physics logic for calculating the decays of the Heavy Neutral Lepton (HNL).
class SterileNeutrino(Neutrino):
    def __init__(self, flavor, mass=0.1):
        super().__init__(mass)
        if flavor not in ['n1', 'n2', 'n3']: raise ValueError("Invalid sterile neutrino flavor.")
        self.flavor = flavor
        self._total_width_cache = {}

    def __str__(self):
        return f"SterileNeutrino({self.flavor}, {self.mass:.3f} GeV)"

    # Physics Helper Functions

    @staticmethod
    def lam(a,b,c):
        # Implements the Källén function lambda(a,b,c), a standard kinematic function used in relativistic decay calculations.
        # This corresponds to Equation (2.10) in the Bondarenko paper.
        return (a**2)+(b**2)+(c**2)-(2*a*b)-(2*a*c)-(2*b*c)

    def _integral_cc(self, x_u, x_d, x_l):
        # Calculates the integral for the 3-body charged-current decay width.
        # This corresponds to the function I(x_u, x_d, x_l) defined in Equation (3.2) of the Bondarenko paper.
        lower,upper = (x_d+x_l)**2, (1-x_u)**2
        def integrand(x):
            sqrt_arg = self.lam(x,x_l**2,x_d**2)*self.lam(1,x,x_u**2)
            if sqrt_arg < 0: return 0.0
            return (1/x)*(x-x_l**2-x_d**2)*(1+x_u**2-x)*np.sqrt(sqrt_arg)
        integral,_ = quad(integrand, lower, upper)
        return 12*integral

    def _get_nc_coeffs(self, f, is_int, m):
        # Implements Table 3 from the Bondarenko phenomenology paper.
        # It returns the C1 and C2 coefficients needed for neutral current decays based on the final state fermion type.
        s2w = m.params.sin_theta_w_sq
        if isinstance(f,Quark):
            if f.is_up_type: C1,C2 = 0.25*(1-(8/3)*s2w+(32/9)*s2w**2), (1/3)*s2w*((4/3)*s2w-1) # row 1 of the table 3, interference case
            else: C1,C2 = 0.25*(1-(4/3)*s2w+(8/9)*s2w**2), (1/6)*s2w*((2/3)*s2w-1) # row 2 of table 3
        elif isinstance(f,Charged_Lepton):
            if is_int: C1,C2 = 0.25*(1+4*s2w+8*s2w**2), 0.5*s2w*(2*s2w+1) #row 4 of the table 3, interference case
            else: C1,C2 = 0.25*(1-4*s2w+8*s2w**2), 0.5*s2w*(2*s2w-1)  #row 3 of table 3
        else: raise TypeError(f"No NC coeffs for {f.__class__.__name__}")
        return C1,C2
    
    @staticmethod
    def _L_function(x):
        # Implements the L(x) function from the phenomenology paper, used in 3-body neutral current decays.
        # This corresponds to the function defined in the text around Equation (3.4) of the Bondarenko paper.
        if x>=0.5 or x<=0: return 0
        sqrt_val = np.sqrt(1-4*x**2)
        num,den = 1-3*x**2-(1-x**2)*sqrt_val, x**2*(1+sqrt_val)
        return np.log(num/den) if num>0 and den>0 else 0

    def calculate_decay_width(self, outgoing_particles: List[Particle], model: 'Model'):
        """
        This is the central calculation engine. It takes a list of final-state particles
        and determines which physical process is occurring, then applies the correct formula
        from the phenomenology paper to calculate the partial decay width.
        """
        model.check_kinematics(self, outgoing_particles)
        m_N, G_F = model.params.m_N, model.params.fermi_constant

        # The code first categorizes the final state particles to create a unique "signature".
        charged_leptons = [p for p in outgoing_particles if isinstance(p, Charged_Lepton)]
        light_neutrinos = [p for p in outgoing_particles if isinstance(p, LightNeutrino)]
        quarks = [p for p in outgoing_particles if isinstance(p, Quark)]
        mesons = [p for p in outgoing_particles if isinstance(p, Meson)]
        alps = [p for p in outgoing_particles if isinstance(p, ALP)]
        
        signature = (len(charged_leptons), len(light_neutrinos), len(quarks), len(mesons), len(alps))
       
        width = 0.0 # Initialize width

        # The match/case statement acts as a router, selecting the correct physics formula based on the signature.
        # Note the choice to require precise user inputs of the form (x,x,x,x,x) specific decay calculations
        # was made, as despite the unideal hardcoded nature of this specific order implementation over an order-independent "set object" form,
        # as we required a method that could take multiple inputs of of the same object type, e.g. two of the same neutrinos which is forbidden by sets.
        # Hence I decided to use these ordered lists. There are some downsides, such as having to hard-code changes to  all the case switches if a new exotic particle is introduced.
        # However other than in the case of such an extension, this implementation only affects the evaluation of single decay widths , as described in the simulation.py module.
        # While for the rest of the tasks which make up the bulk of the use cases of this engine, the choice of using these hard coded lists have no impact on the usability of the code.  
        #the same particle species. 


        match signature:
            # This case handles all 2-body decays into a lepton and a meson.
            case (1, 0, 0, 1, 0) | (0, 1, 0, 1, 0):
                meson, lepton = mesons[0], (charged_leptons + light_neutrinos)[0]
                x_h, x_l = meson.mass / m_N, lepton.mass / m_N
                
                # A nested match statement further divides the logic based on the specific particle types.
                match (lepton, meson):
                    # N -> l- + P+ (Charged Pseudoscalar Meson). Implements Eq. (3.6).
                    case (Charged_Lepton() as lep, ChargedPseudoscalarMeson() as mes):
                        U_alpha = model.get_U_alpha(self, lep)
                        q1, q2 = Quark(mes.quark_content[0]), Quark(mes.quark_content[1])
                        V_ud = model.V_ij_for_quark(q1, q2)
                        term = ((1 - x_l**2)**2 - x_h**2 * (1 + x_l**2))
                        width = (G_F**2 * mes.f_decay_const**2 * abs(V_ud)**2 * abs(U_alpha)**2 * m_N**3)/(16*pi) * term * np.sqrt(self.lam(1,x_h**2,x_l**2))
                    
                    
                    # N -> nu + P0 (Neutral Pseudoscalar Meson). Implements Eq. (3.7).
                    case (LightNeutrino() as nu, NeutralPseudoscalarMeson() as mes):
                        primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu.flavor)
                        U_alpha = model.get_U_alpha(self, primary_lepton)
                        width = ((G_F**2 * mes.f_decay_const**2  * m_N**3)/(32*pi) )* abs(U_alpha)**2 * (1-x_h**2)**2

                    # N -> l- + V+ (Charged Vector Meson). Implements Eq. (3.8).
                    case (Charged_Lepton() as lep, ChargedVectorMeson() as mes):
                        U_alpha = model.get_U_alpha(self, lep)
                        q1, q2 = Quark(mes.quark_content[0]), Quark(mes.quark_content[1])
                        V_ud = model.V_ij_for_quark(q1, q2)
                        term = ((1-x_l**2)**2 + x_h**2*(1+x_l**2) - 2*x_h**4)
                        width = (G_F**2 * mes.g_decay_const**2 * abs(V_ud)**2 * abs(U_alpha)**2 * m_N**3)/(16*pi*mes.mass**2) * term * np.sqrt(self.lam(1,x_h**2,x_l**2))


                    # N -> nu + V0 (Neutral Vector Meson). Implements Eq. (3.9).
                    case (LightNeutrino() as nu, NeutralVectorMeson() as mes):
                        primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu.flavor)
                        U_alpha = model.get_U_alpha(self, primary_lepton)
                        width = (G_F**2 * mes.kappa_factor**2 * mes.g_decay_const**2 * abs(U_alpha)**2 * m_N**3)/(32*pi*mes.mass**2) * (1+2*x_h**2)*(1-x_h**2)**2
                    
                    case _:
                        raise TypeError("Unsupported or charge-mismatched lepton-meson decay combination.")

            # This case handles 3-body decays into a lepton and two quarks (Charged Current). Implements Eq. (3.1).
            case (1, 0, 2, 0, 0): # CC: N -> l + u + d
                l_alpha, u, d = charged_leptons[0], quarks[0], quarks[1]
                U_alpha = model.get_U_alpha(self, l_alpha)
                x_l, x_u, x_d = l_alpha.mass/m_N, u.mass/m_N, d.mass/m_N
                N_w = 3 * abs(model.V_ij_for_quark(u, d))**2
                width = N_w * (G_F**2*m_N**5)/(192*pi**3) * (U_alpha**2) * self._integral_cc(x_u, x_d, x_l)
            
            # This case handles 3-body decays into a neutrino and two quarks (Neutral Current). Implements Eq. (3.4).
            case (0, 1, 2, 0, 0): # NC: N -> nu + q + q_bar
                nu_alpha, q1, q2 = light_neutrinos[0], quarks[0], quarks[1]
                if q1.__class__ != q2.__class__: raise TypeError("NC hadronic decay requires a quark-antiquark pair.")
                primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu_alpha.flavor)
                U_alpha = model.get_U_alpha(self, primary_lepton)
                x = q1.mass/m_N
                if x >= 0.5: return 0.0
                sqrt_term, L_val = np.sqrt(1-4*x**2), self._L_function(x)
                C1, C2 = self._get_nc_coeffs(q1, False, model)
                term1 = C1*((1-14*x**2-2*x**4-12*x**6)*sqrt_term + 12*x**4*(x**4-1)*L_val)
                term2 = 4*C2*(x**2*(2+10*x**2-12*x**4)*sqrt_term + 6*x**4*(1-2*x**2+2*x**4)*L_val)
                width = 3 * (G_F**2*m_N**5)/(192*pi**3) * abs(U_alpha)**2 * (term1+term2)

            # This case handles all 3-body leptonic decays. It implements Eq. (3.1) for CC and Eq. (3.4) for NC/Interference.
            case (2, 1, 0, 0, 0):
                nu, l1, l2 = light_neutrinos[0], charged_leptons[0], charged_leptons[1]
                if l1.__class__ != l2.__class__: # CC
                    lepton_doublets = {'nu_e': Electron, 'nu_mu': Muon, 'nu_tau': Tau}
                    if isinstance(l1, lepton_doublets.get(nu.flavor)): l_alpha, l_beta = l2, l1
                    elif isinstance(l2, lepton_doublets.get(nu.flavor)): l_alpha, l_beta = l1, l2
                    else: raise TypeError("Lepton family violation in CC decay.")
                    U_alpha = model.get_U_alpha(self, l_alpha)
                    x_l, x_u, x_d = l_alpha.mass/m_N, nu.mass/m_N, l_beta.mass/m_N
                    width = (G_F**2*m_N**5)/(192*pi**3) * (U_alpha**2) * self._integral_cc(x_u, x_d, x_l)
                else: # NC / Interference
                    primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu.flavor)
                    U_alpha = model.get_U_alpha(self, primary_lepton)
                    x = l1.mass/m_N
                    if x >= 0.5: return 0.0
                    sqrt_term, L_val = np.sqrt(1-4*x**2), self._L_function(x)
                    is_interference = l1.__class__ == primary_lepton.__class__
                    C1, C2 = self._get_nc_coeffs(l1, is_interference, model)
                    term1 = C1*((1-14*x**2-2*x**4-12*x**6)*sqrt_term + 12*x**4*(x**4-1)*L_val)
                    term2 = 4*C2*(x**2*(2+10*x**2-12*x**4)*sqrt_term + 6*x**4*(1-2*x**2+2*x**4)*L_val)
                    width = (G_F**2*m_N**5)/(192*pi**3) * abs(U_alpha)**2 * (term1+term2)
            
            # This case handles the "invisible" decay into three neutrinos. It implements Eq. (3.5).
            case (0, 3, 0, 0, 0): # Purely neutrinic NC decay
                nu_alpha, nu_beta_1, nu_beta_2 = light_neutrinos[0], light_neutrinos[1], light_neutrinos[2]
                
                if nu_beta_1.flavor != nu_beta_2.flavor:
                    raise TypeError("Invisible decay requires a neutrino-antineutrino pair of the same flavor.")

                primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu_alpha.flavor)
                U_alpha = model.get_U_alpha(self, primary_lepton)
                
                delta_ab = 1.0 if nu_alpha.flavor == nu_beta_1.flavor else 0.0
                
                width = (1+delta_ab) *( (G_F**2*m_N**5)/(768*pi**3)) * abs(U_alpha)**2

            #This incorporates the dominant ALP decay channel N -> a + nu.
            case (0, 1, 0, 0, 1): # N -> nu + ALP
                nu_alpha, alp = light_neutrinos[0], alps[0]
                primary_lepton = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}.get(nu_alpha.flavor)
                U_alpha = model.get_U_alpha(self, primary_lepton)
                x_a = alp.mass / m_N
                width =  ((abs(U_alpha)**2 * m_N**3 )/(4*np.pi* alp.decay_const**2 ))* np.sqrt(1+ x_a**2) * (1-x_a**2)**1.5 #equation 2.21 of Deppisch et al 2024




            case _:
                raise TypeError(f"Could not classify the provided decay channel signature: {signature}")
        
        return width

    #The below get_x_channels functions work to generate the possible decay channels for the HNL.
    # They return these lists as inputs for the above calculate_decay_width function.
    # This combination is how the decay width is calculated in the later decay width functions.

    def _get_invisible_channels(self, model: 'Model'):
        # Generates the 9 possible decay channels for the invisible decay N -> nu_alpha nu_beta nu_beta_bar.
        channels = []
        light_neutrinos = [LightNeutrino('nu_e', model, self), LightNeutrino('nu_mu', model, self), LightNeutrino('nu_tau', model, self)]
        for nu_alpha in light_neutrinos:
            for nu_beta in light_neutrinos:
                 channels.append([nu_alpha, nu_beta, nu_beta])
        return channels

    def _get_charged_leptonic_channels(self, model: 'Model'):
        # Generates all possible 3-body decays that include charged leptons in the final state.
        channels = []
        leptons = [Electron(), Muon(), Tau()]
        light_neutrinos = [LightNeutrino('nu_e', model, self), LightNeutrino('nu_mu', model, self), LightNeutrino('nu_tau', model, self)]
        
        # 1. Neutral Current / Interference: N -> nu_alpha l_beta+ l_beta-
        for nu_alpha in light_neutrinos:
            for l_beta in leptons:
                channels.append([nu_alpha, l_beta, l_beta])

        # 2. Charged Current: N -> l_alpha- nu_beta l_beta+
        lepton_doublets = {'nu_e': Electron(), 'nu_mu': Muon(), 'nu_tau': Tau()}
        for nu_beta in light_neutrinos:
            l_beta = lepton_doublets[nu_beta.flavor]
            for l_alpha in leptons:
                if l_alpha.__class__ != l_beta.__class__:
                    channels.append([l_alpha, nu_beta, l_beta])
        return channels
        
    def _get_exclusive_hadronic_channels(self, model: 'Model'):
        # Generates all possible 2-body decays into a lepton and a single meson.
        channels = []
        leptons = [Electron(), Muon(), Tau()]
        light_neutrinos = [LightNeutrino('nu_e', model, self), LightNeutrino('nu_mu', model, self), LightNeutrino('nu_tau', model, self)]
        charged_ps = [PiPlus(), KPlus(), DPlus(), DStrangePlus(), BPlus(), BCharm()] # 
        neutral_ps = [PiZero(), Eta(), EtaPrime(), EtaCharmed()] 
        charged_vs = [RhoPlus(), DStarPlus(), DstarstrangePlus()] # Charged Vector Mesons
        neutral_vs = [RhoZero(), Omega(), Phi(), JPsi()] # Neutral Vector Mesons
        #charged current l + charged meson
        for meson in (charged_ps +charged_vs):
            for lepton in leptons: channels.append([lepton, meson])
        #Neutral current \nu + neutral meson
        for meson in (neutral_ps + neutral_vs):
            for nu in light_neutrinos: channels.append([nu, meson])
        return channels
        
    def _get_inclusive_quark_channels(self, model: 'Model'):
        # Generates all possible 3-body decays into a lepton and a quark-antiquark pair.
        channels = []
        leptons = [Electron(), Muon(), Tau()]
        up_quarks, down_quarks = [Quark('up'), Quark('charm'), Quark('top')], [Quark('down'), Quark('strange'),Quark('bottom')]
        light_neutrinos = [LightNeutrino('nu_e', model, self), LightNeutrino('nu_mu', model, self), LightNeutrino('nu_tau', model, self)]
        for l_alpha, u, d in itertools.product(leptons, up_quarks, down_quarks): #itertools.product generates all combinations of the three lists, alternative to nested loops.
            channels.append([l_alpha, u, d])
        for nu_alpha in light_neutrinos:
            for quark in (up_quarks + down_quarks):
                channels.append([nu_alpha, quark, quark])
        return channels
    
    
    # Here we actually calculate the decay widths the channels taken from the get_x_channels functions defined right above
    #these channels are then passed through the calculate_decay_width function where each channel is compared to the switches to calculate the relevant decay width.
    def get_invisible_width(self, model: 'Model'):
        # Calculates the total width for the 'invisible' category for the right-hand plot.
        width = 0.0
        for ch in self._get_invisible_channels(model):
            try:
                width += self.calculate_decay_width(ch, model)
            except (ValueError, TypeError):
                continue
        return width

    def get_charged_leptonic_width(self, model: 'Model'):
        # Calculates the total width for the 'leptons' category.
        width = 0.0
        for ch in self._get_charged_leptonic_channels(model):
            try:
                width += self.calculate_decay_width(ch, model)
            except (ValueError, TypeError):
                continue
        return width

    def get_hadronic_width(self, model: 'Model'):
        # the total probability of decaying into all possible hadronic states is very close to the probability of decaying into quarks.
        #Use QCD Correction to account for the difference between hadronic and quark-level decays.
        # Calculates the total width for the 'quarks' category for the right-hand plot.
        # It  switches between exclusive meson and inclusive quark calculations at mN = 2 GeV.
        width = 0.0
        if model.params.m_N < 2.0: # Use exclusive sum for low mass 
            hadronic_channels = self._get_exclusive_hadronic_channels(model)
            for channel in hadronic_channels:
                try:
                    width += self.calculate_decay_width(channel, model)
                except (ValueError, TypeError):
                    continue
        else: # Use inclusive quark calculation for high mass 
            quark_channels = self._get_inclusive_quark_channels(model)
            width_hadronic_quark_level = 0
            for channel in quark_channels:
                 try:
                    width_hadronic_quark_level += self.calculate_decay_width(channel, model)
                 except (ValueError, TypeError):
                    continue
            qcd_correction = model.get_qcd_correction(model.params.m_N)
            width = width_hadronic_quark_level * (1 + qcd_correction)
        return width
    #Here we introduce the ALP extension from Deppisch et al 2024.
    def get_ALP_width(self, model: 'Model'):
        # Calculates the total width for the ALP decay channel.
        
        width = 0.0
        for flavor in ('nu_e', 'nu_mu', 'nu_tau'):
            nu = LightNeutrino(flavor, model, self)
            alp = model.create_alp()
            try:
                width += self.calculate_decay_width([nu, alp], model)
            except (ValueError, TypeError):
                continue
        return width
    
    #The following function combines the above categories of decay widths to calculate the total decay width.

    def get_total_decay_width(self, model: 'Model'):
        # Calculates the total decay width by summing the three main categories.
        cache_key = (self.mass,)
        if cache_key in self._total_width_cache: return self._total_width_cache[cache_key]
        #Multiplying by 2 to account for the Majorana nature of the HNL.
        total_width = 2* (self.get_charged_leptonic_width(model) + 
                       self.get_invisible_width(model) + 
                       self.get_hadronic_width(model) +
                       self.get_ALP_width(model))
                       
        
        self._total_width_cache[cache_key] = total_width
        return total_width
    
    # The following function calculates the branching ratio for a specific user provided decay channel. 
    # Usage instructions are specified in the Simulation.py module. 

    def get_branching_ratio(self, outgoing_particles: List[Particle], model: 'Model'):
        # Calculates the branching ratio for a single, specific decay channel.
        try:
            partial_width = self.calculate_decay_width(outgoing_particles, model)
        except (ValueError, TypeError):
            return 0.0
        
        total_width = self.get_total_decay_width(model)
        
        return partial_width / total_width if total_width > 0 else 0.0


class LightNeutrino(Neutrino):
    # This class represents the light, Standard Model neutrinos.
    # Its mass is not a fixed constant but is determined by the parent sterile neutrino's mass
    # and the mixing angle between them, as described by the seesaw mechanism section in the corresponding thesis paper. 
    def __init__(self,
                 flavor: str,
                 model: 'Model',
                 sterile: 'SterileNeutrino'):
        U = model.get_U_alpha(sterile, 
                              {'nu_e': Electron(),
                               'nu_mu': Muon(),
                               'nu_tau': Tau()}[flavor])
        super().__init__(mass=abs(U)**2  * sterile.mass)
        #super().__init__(mass=0.0)

        if flavor not in ('nu_e','nu_mu','nu_tau'):
            raise ValueError(f"Invalid flavor {flavor!r}")
        self.flavor = flavor
