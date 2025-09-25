import numpy as np
from typing import List, Optional
from .scenario import Scenario
from munacco.input.network_data import NetworkData


class ScenarioGenerator:
    """
    Generates multiple forecast uncertainty scenarios for RES.

    Attributes:
        spatial_correlation (bool): Whether to consider spatial correlation.
        random_seed (int, optional): Seed for reproducibility.
    """
    def __init__(self, spatial_correlation: bool = False, random_seed: Optional[int] = None):
        self.spatial_correlation = spatial_correlation
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate(self, network: NetworkData, n_scenarios: int, forecast_timing = ["d0"], forecast_sigma: Optional[dict] = None ) -> List[Scenario]:
        """
        Generate a list of Scenario objects with random forecast deviations.

        Args:
            network (NetworkData): The input grid data.
            n_scenarios (int): Number of scenarios to generate.
            forecast_timing (str): Timing of forecast with deviation. Options: ['d1'],['d0'], ['d1','d0'] etc
            forecast_sigma (dict): sigma values for different res carriers (wind, solar)

        Returns:
            List[Scenario]: List of generated scenarios.
        """
        res_forecast = network.res.copy()
        
        # Assign sigma (forecast error) to technology 
        if forecast_sigma is not None:
            res_forecast.loc[res_forecast.carrier == 'wind', 'sigma'] = forecast_sigma['wind']
            res_forecast.loc[res_forecast.carrier == 'solar', 'sigma'] = forecast_sigma['solar']
        
        std_devs = res_forecast.sigma
        scenarios = []
        for i in range(n_scenarios):
            err = np.random.normal(loc=0.0, scale=std_devs)
            new_p_max_pu = np.clip(res_forecast["p_max_pu"] + err, 0, 1)
            for day in forecast_timing:
                res_forecast.loc[res_forecast.RD==False, f'p_{day}'] = res_forecast.loc[res_forecast.RD==False, "g_max"]*new_p_max_pu
                
            scenario = Scenario(
                id=f"sc_{i:04d}",
                network = network,
                res_forecast=res_forecast
                )
            
            scenarios.append(scenario)
        return scenarios
        
        
    
