import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
csv_path = "rel_degree_of_Freedom.csv"
class RelDegreesOfFreedom:
    """
    A class to calculate the effective number of relativistic degrees of freedom, g_star,
    as a function of the dimensionless parameter z = m_N / T.
    """
    def __init__(self, csv_path, m_N):

        self.m_N = m_N
        df = pd.read_csv(csv_path)

        # Convert the temperature T from the file to z = m_N / T
        T_values = df["T (GeV)"]
        z_values = self.m_N / T_values

        # Get the g_star values
        g_star_values = df["g∗s"]

        # Since T is decreasing, z = m_N / T is increasing.

        logZ = np.log10(z_values.to_numpy())
        logg = np.log10(g_star_values.to_numpy())

        # Create the interpolation function based on log10(z)
        self._interp = interp1d(logZ, logg, kind="linear", fill_value="extrapolate")

    def g_star(self, z):

        # The interpolation is directly in terms of log10(z)
        return 10 ** self._interp(np.log10(z))


