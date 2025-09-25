import pandas as pd
import numpy as np
from munacco.input.network_data import NetworkData


class Scenario:
    """
    Represents a single uncertainty scenario for RES forecasts.

    Attributes:
        id (str): Unique identifier for the scenario.
        res_df (pd.DataFrame): Modified RES DataFrame for this scenario (includes p_d2, p_d1, p_d0).
        metadata (dict): Optional metadata for debugging or tracing.
    """
    def __init__(
        self,
        id: str,
        res_forecast: pd.DataFrame,
        network: NetworkData,
    ):
        self.id = id
        self.network = network
        self.res_forecast = network.res.copy()
        if res_forecast is not None:
            self.res_forecast = res_forecast.copy()
        self.z_ptdf = None
        self.gsk = None
        self.results = {}
        self.kpis = {}
        self.fb_parameters = {'initial': None,
                        'minram': None,
                        'iva': None,
                        'a': None,
                        'amax': None}
        self.vertices = None
        #self.plot = Visualization(self)
        self.npf = None
        self.alpha = np.zeros(len(network.A))
        
        self.GENMargin = np.zeros(len(network.P))
        self.CCMargin = np.zeros(len(network.nes))
