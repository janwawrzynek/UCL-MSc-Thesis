
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Union
import re 
from scipy.special import kn, zeta

from .Model import Model
from .Model_Parameters import ModelParameters

from .Particles import (
    Particle, SterileNeutrino, Electron, Muon, Tau, LightNeutrino, Quark, ALP,
    PiPlus,KPlus,DPlus,DStrangePlus,BPlus,BCharm,PiZero,Eta,EtaPrime,EtaCharmed,RhoPlus,DStarPlus,DstarstrangePlus,RhoZero,Omega,Phi,JPsi 
)
z_init = 0.3675    # from t≈1e-5 s (see Deppisch et al 2024)
z_final = 116.3    # from t=1 s
z_range = np.logspace(np.log10(z_init), np.log10(z_final), 1000)  # z = m_N / T
HBAR = 6.582e-25
class Plotting:
    """
    Handles the creation of plots for HNL phenomenology.
    """
    def __init__(self, model_params: ModelParameters, g_star_csv_path: str):
        """
        Initializes the plotter with base model parameters.
        """
        self.base_params = model_params
        self.g_star_csv_path = g_star_csv_path

        # factory map bound to this plotter's models
        self.factory_map = {
            Electron: lambda model, sterile_parent: model.create_electron(),
            Muon:     lambda model, sterile_parent: model.create_muon(),
            Tau:      lambda model, sterile_parent: model.create_tau(),
            ALP:      lambda model, sterile_parent: model.create_alp(),
            PiPlus:   lambda model, sterile_parent: model.create_pi_plus(),
            KPlus:    lambda model, sterile_parent: model.create_k_plus(),
            DPlus:   lambda model, sterile_parent: model.create_D_plus(),
            DStrangePlus: lambda model, sterile_parent: model.create_D_strange_plus(),
            BPlus:   lambda model, sterile_parent: model.create_B_plus(),
            BCharm:   lambda model, sterile_parent: model.create_B_charm(),
            PiZero:   lambda model, sterile_parent: model.create_pi_zero(),
            Eta:      lambda model, sterile_parent: model.create_eta(),
            EtaPrime: lambda model, sterile_parent: model.create_eta_prime(),
            EtaCharmed: lambda model, sterile_parent: model.create_eta_charmed(),
            RhoPlus:  lambda model, sterile_parent: model.create_rho_plus(),
            DStarPlus: lambda model, sterile_parent: model.create_D_star_plus(),
            DstarstrangePlus: lambda model, sterile_parent: model.create_D_star_strange_plus(),
            RhoZero:  lambda model, sterile_parent: model.create_rho_zero(),
            Omega:    lambda model, sterile_parent: model.create_omega(),
            Phi:      lambda model, sterile_parent: model.create_phi(),
            JPsi:     lambda model, sterile_parent: model.create_JPsi(),

            
            
        }

        print("Plotting engine initialized.")

    
    def plot_branching_ratios(self, plot_spec: Dict[str, list], mass_range: np.ndarray):
        """
        Calculates and plots branching ratios for exclusive hadronic channels .
        """
        print(f"\nGenerating plot data for {len(mass_range)} mass points...")
        #initializes a dictionary to store the results of the calculations. 
        # For each label in plot_spec (like 'invis.', 'lept.'), it creates an empty list
        #As the code loops through the masses, it will calculate the branching ratio for each group and append it to the corresponding list.
        results = {label: [] for label in plot_spec}
        #Defined inside plot branching ratios so that it has access to the newest masses, sterile neutrino, and model
        def _create_channel_from_recipe(recipe: list, model: Model, sterile_parent: SterileNeutrino):
            channel = []
            for item in recipe:
                if isinstance(item, tuple):
                    _, flavor = item
                    channel.append(LightNeutrino(flavor, model, sterile_parent))
                else:
                    try:
                        factory = self.factory_map[item]
                    except KeyError:
                        raise RuntimeError(f"No factory defined for particle class {item.__name__}")
                    channel.append(factory(model, sterile_parent))
            return channel

        for i, mass in enumerate(mass_range):
            current_params = copy.copy(self.base_params)
            current_params.m_N = mass
            current_model = Model(params=current_params, g_star_csv_path=self.g_star_csv_path)
            sterile_neutrino = current_model.create_sterile_neutrino('n1')
            
            total_width = sterile_neutrino.get_total_decay_width(current_model)

            for label, recipes in plot_spec.items():
                partial_width = 0.0
                if total_width > 0:
                    for recipe in recipes:
                        try:
                            real_channel = _create_channel_from_recipe(recipe, current_model, sterile_neutrino)
                            partial_width += sterile_neutrino.calculate_decay_width(real_channel, current_model)
                        except (ValueError, TypeError, AttributeError):
                            continue
                
                br = partial_width / total_width if total_width > 0 else 0.0
                results[label].append(br)
            
            if (i + 1) % 10 == 0 or i == len(mass_range) - 1:
                print(f"  ...processed point {i+1}/{len(mass_range)} (Mass = {mass:.2f} GeV)")  #Progress update every 10 mass steps

        # Plotting
        print("\nPlotting results")
        plt.figure(figsize=(12, 8))
        
        for label, br_values in results.items():
            if any(br > 0 for br in br_values):
                plt.plot(mass_range, br_values, label=label, lw=2.5, alpha=0.8)

        plt.xlabel("Sterile Neutrino Mass $M_N$ [GeV]", fontsize=20)
        plt.ylabel("Branching Ratio (BR)", fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        #plt.title("HNL Branching Ratios vs. Mass (Exclusive Hadrons)", fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=12)
        plt.ylim(1e-10, 1.1)
        plt.xlim(mass_range[0], mass_range[-1])
        
        plot_filename = "branching_ratios_hadrons.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.show()

    def plot_quark_level_branching_ratios(self, mass_range: np.ndarray):
        """
        Calculates and plots branching ratios for inclusive categories (Fig. 13, right panel).
        """
    
        
        results = {'quarks': [], 'leptons': [], 'invisible': [], 
                   'ALPs': []
                   }

        for i, mass in enumerate(mass_range):
            current_params = self.base_params
            current_params = copy.copy(self.base_params)
            current_params.m_N = mass
            current_model = Model(params=current_params, g_star_csv_path=self.g_star_csv_path)
            sterile_neutrino = current_model.create_sterile_neutrino('n1')
            total_width = sterile_neutrino.get_total_decay_width(current_model)

            if total_width > 0:
                hadronic_width = sterile_neutrino.get_hadronic_width(current_model)
                leptonic_width = sterile_neutrino.get_charged_leptonic_width(current_model)
                invisible_width = sterile_neutrino.get_invisible_width(current_model)
                #adding the ALP decays to compare
                alp_width = sterile_neutrino.get_ALP_width(current_model)
                results['quarks'].append(hadronic_width / total_width)
                results['leptons'].append(leptonic_width / total_width)
                results['invisible'].append(invisible_width / total_width)
                results['ALPs'].append(alp_width / total_width)
            else:
                results['quarks'].append(0)
                results['leptons'].append(0)
                results['invisible'].append(0)
                results['ALPs'].append(0)

            if (i + 1) % 10 == 0 or i == len(mass_range) - 1:
                print(f"  ...processed point {i+1}/{len(mass_range)} (Mass = {mass:.2f} GeV)")

        # Plotting
        print("\nPlotting results...")
        print(results['quarks'][:5])  # Print first 5 values for debugging
        print(results['leptons'][:5])  # Print first 5 values for debugging
        print(results['invisible'][:5])  # Print first 5 values for debugging
        print(results['ALPs'][:5])  # Print first 5 values for debugging
        plt.figure(figsize=(12, 8))
        
        for label, br_values in results.items():
            plt.plot(mass_range, br_values, label=label, lw=2.5, alpha=0.8)

        plt.xlabel("Sterile Neutrino Mass $M_N$ [GeV]", fontsize=20)
        plt.ylabel("Branching Ratio (BR)", fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        #plt.title("HNL Branching Ratios vs. Mass (Inclusive Quarks)", fontsize=16)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.ylim(1e-8, 1.1)
        plt.xlim(mass_range[0], mass_range[-1])
        
        plot_filename = "branching_ratios_quarks.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.show()

    #below I am going to plot the QCD correction term from get_qcd_correction over a range of masses
    def plot_qcd_correction(self, mass_range:np.ndarray, save_filename="qcd_correction.png"):
        """
        Uses the existing model.get_qcd_correction to plot the QCD correction vs mass.
        """
        corrections = []
        for i, mass in enumerate(mass_range):
            #using the copy module to avoid modifying the base parameters and only changing the masses
            current_params = copy.copy(self.base_params)
            current_params.m_N = mass
            current_model = Model(params=current_params, g_star_csv_path=self.g_star_csv_path)
            # use the existing method instead of redefining it
            corr = current_model.get_qcd_correction(mass)
            corrections.append(corr)

            if (i + 1) % 10 == 0 or i == len(mass_range) - 1:
                print(f"  ...computed QCD correction for point {i+1}/{len(mass_range)} (Mass = {mass:.2f} GeV)")
        corrections = np.array(corrections)
        valid = np.isfinite(corrections) & (mass_range >= 1.0)
        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        plt.plot(mass_range[valid], corrections[valid], label=r"QCD correction ", lw=2, color= 'black')
        #plt.xscale("log")
        #plt.yscale("log")
        plt.xlabel(r"HNL Mass $M_N$ (GeV)", fontsize=25)
        plt.ylabel(r" $\Delta_{QCD}$ Correction", fontsize=25)
        #plt.title("QCD Correction vs. Mass", fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        #plt.legend(fontsize=12)
        plt.xlim(mass_range[0], mass_range[-1])
        plt.ylim(0, 0.35)
        plt.tight_layout()
        plt.savefig(save_filename)
        print(f"QCD correction plot saved as {save_filename}")
        plt.show()


#Below I will plot the evolution of the thermally averaged decay rates following the example in Deppisch 2024 Figure 4
# First I will plot the degrees of freedom over the relevent temperature range
    def plot_g_star(self, model: Model, z_range: np.ndarray):
        g_star = model.get_g_star(z_range)

        print(g_star[:5]) # Print first 5 values for debugging
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, g_star, label=r"$g_*$", color='blue', lw=2)
        #plt.xscale('log')
        
        plt.xlabel(r"$z =m_N/T$", fontsize=20)
        plt.ylabel(r"$g_*$", fontsize=20)
        #plt.title("Effective Number of Relativistic Degrees of Freedom", fontsize=16)
    
        plt.grid(True, which="both", ls="--", alpha=0.5)
        #plt.legend(fontsize=12)
        plt.xlim(z_range[0], z_range[-1])
        plt.xlim(0.5,1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=20)
        plt.xscale('log')
        plt.show()


    def plot_thermally_averaged_decay_rates(self, m_N: float, z_range: np.ndarray):

    
        """
        Plots thermally averaged decay rates for a sterile neutrino of mass m_N,
        comparing the SM part and the N -> a ν channel, summed over all sterile flavors.
        """
        # Build model with the desired sterile neutrino mass
        params = copy.copy(self.base_params)
        params.m_N = m_N
        m_a = params.m_a
        model = Model(params=params, g_star_csv_path=self.g_star_csv_path)

        # Sum over sterile neutrino flavors n1, n2, n3
        #sterile_flavors = ["n1", "n2", "n3"]
        #consider each flavour separately, as in Deppisch 2024
        sterile_flavors = ["n2"]
        Gamma_N_SM_total = 0.0
        Gamma_N_to_a_total = 0.0
        for fl in sterile_flavors:
            sterile = model.create_sterile_neutrino(fl)
            Gamma_N_SM_total += (
                sterile.get_charged_leptonic_width(model)
                + sterile.get_invisible_width(model)
                + sterile.get_hadronic_width(model)
            )
            Gamma_N_to_a_total += sterile.get_ALP_width(model)  # N -> a ν

        # Adjusted z for ALP mass (for its own equilibrium)
        z_range_alp = (m_a / m_N) * z_range

        # Equilibrium number densities
        g_N = 2.0  # three Majorana steriles, each with 2 spin states
        g_a = 1.0      # ALP scalar
        n_N_eq = model.equilibrium_species_density(g_N, m_N, z_range, is_fermion=True)
        n_a_eq = model.equilibrium_species_density(g_a, m_a, z_range_alp, is_fermion=False)

        # Thermal averaging factors
        thermal_factor_N = kn(1, z_range) / kn(2, z_range)
        thermal_factor_a = kn(1, z_range_alp) / kn(2, z_range_alp)  # kept for potential a->νν use

        # Thermally averaged reaction densities
        thermally_averaged_N_SM = n_N_eq*thermal_factor_N * Gamma_N_SM_total
        print(thermally_averaged_N_SM[:5], "thermall averaged decay rates")  # Print first 5 values for debugging
        thermally_averaged_N_to_a = n_N_eq*thermal_factor_N * Gamma_N_to_a_total  # N -> a nu

        # Entropy density 
        try:
            s = model.entropy_density(z_range)
        except Exception:
            s = np.vectorize(model.entropy_density)(z_range)

        # Equilibrium yields (not currently plotted)
        Y_N_eq = n_N_eq / s
        Y_a_eq = n_a_eq / s

        # Plotting the thermally averaged rates
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, thermally_averaged_N_SM, label="SM ", lw=2)
        plt.plot(z_range, thermally_averaged_N_to_a, label="N → a ν", lw=2, linestyle="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$z = m_N / T$", fontsize=25)
        plt.ylabel("Thermally Averaged Decay Width Density", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.title(f"Thermally Averaged Decay Rates for the N1 Flavour, $m_N$ = {m_N:.2f} GeV", fontsize=16)
        plt.grid(which="both", ls="--", alpha=0.5)
        plt.legend(prop={'size': 20})
        plt.tight_layout()
        plt.show()


    def plot_lifetime_iso_lines(self,
                                mN_min: float,
                                mN_max: float,
                                n_masses: int,
                                levels: list[float] = [0.1,1.0,10.0]):
    
    

        # ℏ in GeV·s
        HBAR = 6.582e-25

        # 1) Prepare a grid of m_N values
        m_vals = np.logspace(np.log10(mN_min), np.log10(mN_max), n_masses)

        # 2) Compute Γ_tot(m_N; |U|^2=1) once per m_N
        Gamma_unit = np.zeros_like(m_vals)
        for i, mN in enumerate(m_vals):
            params = copy.copy(self.base_params)
            params.m_N = mN
            # force |U_{n1e}|^2 = 1, zero out other mixings
            params.U_matrix[:, :] = 0.0
            #change the below to [0,0] = 1.0 for the first sterile neutrino
            #change the below to [1,1] = 1.0 for the second sterile neutrino
            # change the below to [2,2] = 1.0 for the third sterile neutrino
            params.U_matrix[0, 0] = 1.0
            model = Model(params=params, g_star_csv_path=self.g_star_csv_path)
            #change sterile neutrino to n1,n2,n3 as needed
            sterile = model.create_sterile_neutrino('n1')
            Gamma_unit[i] = sterile.get_total_decay_width(model)

        # 3) For each lifetime level, solve U2 = ħ/(τ*Γ_unit)
        plt.figure(figsize=(10,6))
        for tau in levels:
            U2 = HBAR / (tau * Gamma_unit)
            plt.plot(m_vals, U2, lw=2, label=f"sterile neutrino Lifetime = {tau}s")

        # 4) Style the plot
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim(mN_min, mN_max)
        #plt.xlim(1.7,mN_max)
        plt.xlabel(r"Sterile mass $m_{N_1}$ (GeV)", fontsize=20)
        #adjust the label to reflect the mixing term
        plt.ylabel(r"Mixing $|U_{e \ N_1}|^2$", fontsize=20)
        #plt.title("Iso–Lifetime Curves for N1 HNL")
        plt.legend(prop = {'size': 20})
        plt.grid(which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()



    def plot_combined_branching(self, mass_range: np.ndarray):
        """
        For each m_N in mass_range, sum over n1,n2,n3:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import copy
        from .Model import Model

        BR_SM  = []
        BR_ALP = []

        for m in mass_range:
            # build a fresh model at mass m
            params = copy.copy(self.base_params)
            params.m_N = m
            params.f_a =1e6###############################
            model = Model(params=params, g_star_csv_path=self.g_star_csv_path)
            Γ_SM_tot  = 0.0
            Γ_ALP_tot = 0.0
            for flav in ("n1","n2","n3"):
                sterile = model.create_sterile_neutrino(flav)
                # sum all SM widths
                Γ_SM  = (sterile.get_charged_leptonic_width(model)
                    + sterile.get_invisible_width(model)
                    + sterile.get_hadronic_width(model))
                Γ_ALP = sterile.get_ALP_width(model)
                Γ_SM_tot  += Γ_SM
                Γ_ALP_tot += Γ_ALP

            Γ_tot = Γ_SM_tot + Γ_ALP_tot
            if Γ_tot > 0:
                BR_SM.append (Γ_SM_tot  / Γ_tot)
                BR_ALP.append(Γ_ALP_tot / Γ_tot)
            else:
                BR_SM.append (0.0)
                BR_ALP.append(0.0)

        # now plot
        plt.figure(figsize=(8,6))
        plt.plot(mass_range, BR_SM,  label="BR(N→SM)",  lw=2)
        plt.plot(mass_range, BR_ALP, label="BR(N→aν)", lw=2, ls="--")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(mass_range[0], mass_range[-1])
        plt.xlabel(r"Sterile Neutrino Mass $m_N$ (GeV)", fontsize =20)
        plt.ylabel("Branching Ratio", fontsize=20)
        #plt.title("HNL Branching Ratios into SM and ALP Channels $f_a = 10^7$ GeV")
        plt.grid(which="both", ls="--", alpha=0.5)
        plt.legend(prop = {'size': 20})
        plt.tight_layout()
        plt.show()




    def plot_fermi_theory_error(self,
                                mN_min: float,
                                mN_max: float,
                                n_masses: int):
        import copy
        from .Model import Model
        m_W = 80.4  # W boson mass in GeV
        m_vals = np.logspace(np.log10(mN_min), np.log10(mN_max), n_masses)
        #Fermi contact theory error in comparison to use of the full propagator
        fermi_error_values = ((m_vals/m_W)**2)
        plt.figure( figsize=(8, 6))
        plt.plot(m_vals, fermi_error_values, lw=2, label="Relative Error")
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r"HNL Mass $m_N$ GeV)", fontsize=25)
        plt.ylabel("Relative Error %", fontsize=25)
        #plt.title("Relative Error between the Full Propagator and the Fermi Approximation ", fontsize=14)
        plt.grid(which="both", ls="--", alpha=0.5)
        #plt.legend(fontsize=12)
        plt.xlim(mN_min, mN_max)
        plt.tight_layout()
        plt.show()
        

    def plot_total_sterile_width(self, mass_range: np.ndarray):
        '''
        plotting the total decay width for the sterile neutrino over a mass range. 
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        import copy
        from .Model import Model

        Gamma_SM_list  = []
        Gamma_ALP_list = []
        Gamma_total_list = []

        for m in mass_range:
            # build a fresh model at mass m
            params = copy.copy(self.base_params)
            params.m_N = m
            params.f_a =1e6
            model = Model(params=params, g_star_csv_path=self.g_star_csv_path)
            Gamma_SM_tot  = 0.0
            Gamma_ALP_tot = 0.0
            
            sterile = model.create_sterile_neutrino('n1')
            # sum all SM widths
            gamma_SM  = (sterile.get_charged_leptonic_width(model)
                + sterile.get_invisible_width(model)
                + sterile.get_hadronic_width(model))
            gamma_ALP = sterile.get_ALP_width(model)
            Gamma_SM_tot  += gamma_SM
            Gamma_ALP_tot += gamma_ALP
            Gamma_total = Gamma_SM_tot + Gamma_ALP_tot
            Gamma_total_list.append(Gamma_total)


        # now plot
        plt.figure(figsize=(8,6))
        plt.plot(mass_range, Gamma_total_list,  label="Γ_tot",  lw=2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(mass_range[0], mass_range[-1])
        plt.xlabel(r"Sterile Neutrino Mass $m_N$ (GeV)", fontsize =20)
        plt.ylabel("Decay Width (Γ)", fontsize=20)
        #plt.title("HNL Branching Ratios into SM and ALP Channels $f_a = 10^7$ GeV")
        plt.grid(which="both", ls="--", alpha=0.5)
        #plt.legend(prop = {'size': 20})
        plt.tight_layout()
        plt.show()


    def plot_total_lifetime(self, mass_range: np.ndarray):
        '''
        plotting the total decay width for the sterile neutrino over a mass range. 
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        import copy
        from .Model import Model

        Gamma_SM_list  = []
        Gamma_ALP_list = []
        lifetime_total_list = []

        for m in mass_range:
            # build a fresh model at mass m
            params = copy.copy(self.base_params)
            params.m_N = m
            params.f_a =1e3
            model = Model(params=params, g_star_csv_path=self.g_star_csv_path)
            Gamma_SM_tot  = 0.0
            Gamma_ALP_tot = 0.0
            
            sterile = model.create_sterile_neutrino('n1')
            # sum all SM widths
            gamma_SM  = (sterile.get_charged_leptonic_width(model)
                + sterile.get_invisible_width(model)
                + sterile.get_hadronic_width(model))
            gamma_ALP = sterile.get_ALP_width(model)
            Gamma_SM_tot  += gamma_SM
            Gamma_ALP_tot += gamma_ALP
            Gamma_total = Gamma_SM_tot + Gamma_ALP_tot
            lifetime_value = HBAR / Gamma_total if Gamma_total > 0 else np.inf
            lifetime_total_list.append(lifetime_value)
    


        # now plot
        plt.figure(figsize=(8,6))
        plt.plot(mass_range, lifetime_total_list)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(mass_range[0], mass_range[-1])
        plt.xlabel(r"Sterile Neutrino Mass $m_N$ (GeV)", fontsize =20)
        plt.ylabel("Lifetime (s)", fontsize=20)
        #plt.title("HNL Branching Ratios into SM and ALP Channels $f_a = 10^7$ GeV")
        plt.grid(which="both", ls="--", alpha=0.5)
        #plt.legend(prop = {'size': 20})
        plt.tight_layout()
        plt.show()


    #def plot_sterile_lifetime



