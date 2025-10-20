# models/belief_state.py
from pydantic import BaseModel, Field
import scipy.stats as stats
import numpy as np

class HotPotBelief(BaseModel):
    """Parametric belief for Hot-Pot environment"""
    heating_rate_mean: float = Field(default=1.5, description="°C per second")
    heating_rate_std: float = Field(default=0.3, ge=0.01)
    measurement_noise: float = Field(default=2.0, ge=0.1, description="σ for obs")
    base_temp: float = Field(default=20.0)
    
    def log_likelihood(self, observation: dict, time_elapsed: float) -> float:
        """
        Compute P(observation | belief) for surprisal calculation.
        Returns log-likelihood under Gaussian observation model.
        """
        predicted_temp = self.base_temp + self.heating_rate_mean * time_elapsed
        
        if 'measured_temp' not in observation:
            return 0.0  # No observation to evaluate
            
        return stats.norm.logpdf(
            observation['measured_temp'],
            loc=predicted_temp,
            scale=self.measurement_noise
        )
    
    def update(self, observation: dict, time_elapsed: float) -> 'HotPotBelief':
        """
        Bayesian update (simplified conjugate prior for demo).
        In practice, LLM will update parameters via reasoning.
        """
        if 'measured_temp' not in observation:
            return self
            
        # Simple Bayesian linear regression update
        observed_rate = (observation['measured_temp'] - self.base_temp) / time_elapsed
        
        # Weight by inverse variance
        prior_precision = 1 / (self.heating_rate_std ** 2)
        obs_precision = 1 / (self.measurement_noise ** 2 / time_elapsed)
        
        new_precision = prior_precision + obs_precision
        new_mean = (prior_precision * self.heating_rate_mean + 
                   obs_precision * observed_rate) / new_precision
        new_std = np.sqrt(1 / new_precision)
        
        return HotPotBelief(
            heating_rate_mean=new_mean,
            heating_rate_std=new_std,
            measurement_noise=self.measurement_noise,
            base_temp=self.base_temp
        )


class SwitchLightBelief(BaseModel):
    """Categorical belief over hidden states"""
    wiring_probs: dict[str, float] = Field(
        default={'layout_A': 0.5, 'layout_B': 0.5}
    )
    failure_prob: float = Field(default=0.1, ge=0.0, le=1.0)
    
    def log_likelihood(self, observation: dict) -> float:
        """
        P(observation | belief) marginalizing over wiring layouts
        """
        if 'light_on' not in observation:
            return 0.0
            
        light_on = observation['light_on']
        switch_pos = observation.get('switch_position', 'unknown')
        
        # Marginalize over wiring hypotheses
        total_prob = 0.0
        for layout, prob in self.wiring_probs.items():
            # P(light_on | layout, switch_pos, no_failure)
            expected_on = self._predict_light(layout, switch_pos)
            
            # Account for failure probability
            if light_on:
                obs_prob = expected_on * (1 - self.failure_prob)
            else:
                obs_prob = (1 - expected_on) * (1 - self.failure_prob) + \
                          expected_on * self.failure_prob
            
            total_prob += prob * obs_prob
        
        return np.log(total_prob + 1e-10)
    
    def _predict_light(self, layout: str, switch_pos: str) -> float:
        """Helper: expected light state given layout and switch"""
        if layout == 'layout_A':
            return 1.0 if switch_pos == 'on' else 0.0
        else:  # layout_B is inverted
            return 1.0 if switch_pos == 'off' else 0.0


class ChemTileBelief(BaseModel):
    """Categorical reaction outcomes with noise"""
    reaction_probs: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            'A+B': {'C': 0.8, 'explode': 0.1, 'nothing': 0.1},
            'C+B': {'D': 0.7, 'explode': 0.2, 'nothing': 0.1},
        }
    )
    temperature_modifier: float = Field(default=1.0, ge=0.5, le=2.0)
    
    def log_likelihood(self, observation: dict) -> float:
        """P(outcome | reaction, belief)"""
        reaction = observation.get('reaction')
        outcome = observation.get('outcome')
        
        if not reaction or not outcome:
            return 0.0
            
        if reaction not in self.reaction_probs:
            return np.log(0.01)  # Small prior for unknown reactions
            
        probs = self.reaction_probs[reaction]
        prob = probs.get(outcome, 0.01) * self.temperature_modifier
        
        return np.log(prob + 1e-10)