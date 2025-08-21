import numpy as np
from pathlib import Path

from .Model import Model
from .Model_Parameters import ModelParameters
from .Particles import (
    Particle, SterileNeutrino, Electron, Tau, LightNeutrino, Muon, Quark, ALP,
    PiPlus, KPlus, PiZero, Eta, RhoPlus, RhoZero, Omega
)
from .Plotting import Plotting
from .Plotting import z_init, z_final, z_range

class Simulation:
    """
    The simulation engine. It takes a physical model and can be configured
    to run different types of decay calculations or generate plots.
    """
    def __init__(self, model: Model):
        print("Initialising the simulation engine...")
        self.model = model

    def calculate_partial_width(self, incoming_particle: Particle, outgoing_particles: list[Particle]):
        channel_str = f"{incoming_particle} -> {' + '.join(p.__class__.__name__ for p in outgoing_particles)}"
        print(f"\n[TASK] Calculating partial decay width for: {channel_str}")
        try:
            width = incoming_particle.calculate_decay_width(outgoing_particles, self.model)
            print(f"  > Partial Width: {width:.6e} GeV")
            return width
        #Checks if the calculation fails due to invalid parameters or types (ex mass mismatch)
        except (ValueError, TypeError) as ex:
            print(f"  > Calculation failed: {ex}")
            return 0.0

    def calculate_total_width(self, incoming_particle: Particle):
        print(f"\n[TASK] Calculating total decay width for: {incoming_particle}")
        #get_total_decay_width method from the SterileNeutrino class.
        # This method is responsible for looping through all possible leptonic and hadronic channels and summing up their individual widths.
        try:
            total_width = incoming_particle.get_total_decay_width(self.model)
            print(f"  > Total Width : {total_width:.6e} GeV")
            if total_width > 0:
                # Calculate the lifetime 
                lifetime = 6.582e-25 / total_width
                print(f"  > Lifetime: {lifetime:.6e} s")
            return total_width
        except Exception as ex:
            print(f"  > Calculation failed: {ex}")
            return 0.0
    #Combination of the above
    #Determines branching ratio by dividing the partial width by the total width       
    def run_full_channel_analysis(self, incoming_particle: Particle, outgoing_particles: list[Particle]):
        channel_str = f"{incoming_particle} -> {' + '.join(p.__class__.__name__ for p in outgoing_particles)}"
        print(f"\n[TASK] Running full analysis for channel: {channel_str}")
        try:
            partial_width = incoming_particle.calculate_decay_width(outgoing_particles, self.model)
            print(f"  > Partial Width): {partial_width:.6e} GeV")
            total_width = incoming_particle.get_total_decay_width(self.model)
            print(f"  > Total Width :   {total_width:.6e} GeV")
            if total_width > 0:
                branching_ratio = partial_width / total_width
                print(f"  > Branching Ratio (BR):      {branching_ratio:.10f} ({branching_ratio:.10%})")
            else:
                print("  > Branching Ratio: N/A (Total width is zero)")
        except (ValueError, TypeError) as ex:
            print(f"  > Analysis failed: {ex}")




# Main Execution Block
# creates a block of code that runs only if this script is executed directly (by running python -m OOP.Simulation)
if __name__ == "__main__":
    
    base_params = ModelParameters() # Creates instance of ModelParameters class with default values
    base_params.f_a = 1e3  # Set ALP decay constant to 1 TeV 
    
    try:
        script_dir = Path(__file__).parent #This gets the directory where the Simulation.py script itself is located.
        csv_path = script_dir.parent / "rel_degree_of_Freedom.csv" # looks for the CSV file in the parent directory
        if not csv_path.is_file(): # If it can't find the file there, it checks in the same directory as the script
             csv_path = script_dir / "rel_degree_of_Freedom.csv"
    except NameError: #. If the script is run in an environment where __file__ is not defined (like some interactive notebooks), it will simply assume the CSV file is in the current working directory.
        csv_path = "rel_degree_of_Freedom.csv"

    ###  CHOOSE WHICH PLOT TO RUN ###
    #  "SINGLE_CALCULATION" Runs a single-point calculation for a given HNL mass
    # "PLOT_BRANCHING_RATIOS_HADRONS
    # "PLOT_BRANCHING_RATIOS_QUARKS" 
    # "PLOT_QCD_CORRECTION_VS_MASS"
    # "PLOT_REL_DEGREES_OF_FREEDOM_VS_Z"
    # "PLOT_THERMALLY_AVERAGED_WIDTHS"
    # "LIFETIME_CONTOUR_PLOT"
    # "PLOT_COMBINED_BRANCHING_RATIO_SM_ALP"
    # "Fermi_theory_error" ->  Calculates the Fermi theory error for HNL decays
    TASK_TO_RUN = "PLOT_REL_DEGREES_OF_FREEDOM_VS_Z"  # Change this to switch tasks
    ####################################################################
    match TASK_TO_RUN: 
        case "SINGLE_CALCULATION":
            base_params.m_N = 0.5
            print(f"--- Running single-point calculation for HNL mass of {base_params.m_N} GeV ---")
            # Create a model instance with the base parameters and CSV path
            model = Model(params=base_params, g_star_csv_path=str(csv_path))
            # Create a Sterile Neutrino instance with the specified mass
            sterile_neutrino = model.create_sterile_neutrino('n1')  # Example mass of 0.5 GeV
            # Define the decay channel: HNL -> Electron + Muon + Light Neutrino
            decay_channel = [Electron(), Muon(), LightNeutrino(flavor='nu_e', model=model, sterile=sterile_neutrino)]
            # Run the full channel analysis
            simulation = Simulation(model)
            simulation.run_full_channel_analysis(sterile_neutrino, decay_channel)
            print(f"--- Single-point calculation completed for HNL mass of {base_params.m_N} GeV ---")
        
            print("test")
        

        case "PLOT_BRANCHING_RATIOS_HADRONS":
            print(f"--- Preparing to plot branching ratios for HNL decays into HADRONS ---")
            plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
            #Groups of decay channels, each group corresponds to a different particle type
            #Which is then used to plot the branching ratios for each particle type
            #Using a dictionary to specify the decay channels
            # Done with Plotting.py
            plot_specification = {
                #invis. is the label that will appear in the plot
                'invis.': [
                    [('LightNeutrino', 'nu_e'), ('LightNeutrino', 'nu_e'), ('LightNeutrino', 'nu_e')],
                    [('LightNeutrino', 'nu_e'), ('LightNeutrino', 'nu_mu'), ('LightNeutrino', 'nu_mu')],
                    [('LightNeutrino', 'nu_e'), ('LightNeutrino', 'nu_tau'), ('LightNeutrino', 'nu_tau')],
                    [('LightNeutrino', 'nu_mu'), ('LightNeutrino', 'nu_e'), ('LightNeutrino', 'nu_e')],
                    [('LightNeutrino', 'nu_mu'), ('LightNeutrino', 'nu_mu'), ('LightNeutrino', 'nu_mu')],
                    [('LightNeutrino', 'nu_mu'), ('LightNeutrino', 'nu_tau'), ('LightNeutrino', 'nu_tau')],
                    [('LightNeutrino', 'nu_tau'), ('LightNeutrino', 'nu_e'), ('LightNeutrino', 'nu_e')],
                    [('LightNeutrino', 'nu_tau'), ('LightNeutrino', 'nu_mu'), ('LightNeutrino', 'nu_mu')],
                    [('LightNeutrino', 'nu_tau'), ('LightNeutrino', 'nu_tau'), ('LightNeutrino', 'nu_tau')],
                ],
                'lept.': [
                    [('LightNeutrino', 'nu_e'), Electron, Electron],
                    [('LightNeutrino', 'nu_e'), Muon, Muon],
                    [('LightNeutrino', 'nu_mu'), Electron, Electron],
                    [('LightNeutrino', 'nu_mu'), Muon, Muon],
                    [Electron, ('LightNeutrino', 'nu_mu'), Muon],
                    [Muon, ('LightNeutrino', 'nu_e'), Electron],
                ],
                'π': [
                    [Electron, PiPlus],
                    [Muon, PiPlus],
                    [('LightNeutrino', 'nu_e'), PiZero],
                    [('LightNeutrino', 'nu_mu'), PiZero],
                    [('LightNeutrino', 'nu_tau'), PiZero],
                ],
                'η': [
                    [('LightNeutrino', 'nu_e'), Eta],
                    [('LightNeutrino', 'nu_mu'), Eta],
                    [('LightNeutrino', 'nu_tau'), Eta],
                ],
                'ρ': [
                    [Electron, RhoPlus],
                    [Muon, RhoPlus],
                    [('LightNeutrino', 'nu_e'), RhoZero],
                    [('LightNeutrino', 'nu_mu'), RhoZero],
                    [('LightNeutrino', 'nu_tau'), RhoZero],
                ],
                'K': [
                    [Electron, KPlus],
                    [Muon, KPlus],
                ],
                
                'ALP': [
                    [('LightNeutrino', 'nu_e'), ALP],
                    [('LightNeutrino', 'nu_mu'), ALP],
                    [('LightNeutrino', 'nu_tau'), ALP],
                ]
                

                
            }
            
            mass_range = np.logspace(np.log10(0.05), np.log10(1.0), num=50)
            plotter.plot_branching_ratios(plot_specification, mass_range)

        case "PLOT_BRANCHING_RATIOS_QUARKS":
            print(f"--- Preparing to plot branching ratios for HNL decays at quark level ---")
            plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
            mass_range = np.logspace(np.log10(1.0), np.log10(5.0), num=50)
            plotter.plot_quark_level_branching_ratios(mass_range)

        case "PLOT_QCD_CORRECTION_VS_MASS":
            print(f"--- Preparing to plot QCD correction vs HNL mass ---")
            plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
            mass_range = np.logspace(np.log10(0.05), np.log10(10.0), 300)
            plotter.plot_qcd_correction(mass_range)

        case "PLOT_REL_DEGREES_OF_FREEDOM_VS_Z":
            print(f"--- Preparing to plot relativistic degrees of freedom vs z ---")
            model = Model(params=base_params, g_star_csv_path=str(csv_path))
            plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
            plotter.plot_g_star(model, z_range)

        case "PLOT_THERMALLY_AVERAGED_WIDTHS":
            print(f"--- Preparing to plot thermally averaged widths ---")
            m_N = 0.1  
            z_range = np.logspace(np.log10(z_init), np.log10(z_final), 1000)    
            plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
            plotter.plot_thermally_averaged_decay_rates(m_N, z_range)
        case "LIFETIME_CONTOUR_PLOT":
                plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
                plotter.plot_lifetime_iso_lines(
                    mN_min=0.2,    # GeV
                    mN_max=10.0,    # GeV
                    n_masses=200,   # resolution in m_N
                    levels=[0.1,1.0,10.0]  # seconds
                )
        case "PLOT_COMBINED_BRANCHING_RATIO_SM_ALP":
            print(f"--- Preparing to plot combined SM vs ALP branching ratios for all flavours ---")
            plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
            
            mass_range = np.logspace(np.log10(0.05), np.log10(5.0), num=100)
            plotter.plot_combined_branching(mass_range)
        case "Fermi_theory_error":
                plotter = Plotting(model_params=base_params, g_star_csv_path=str(csv_path))
                plotter.plot_fermi_theory_error(
                    mN_min=0.5,    # GeV
                    mN_max=50,    # GeV
                    n_masses=200  # resolution in m_N
                    
                )

        case _:
            print(f"Error: Unknown task '{TASK_TO_RUN}'. Please choose a valid task.")

