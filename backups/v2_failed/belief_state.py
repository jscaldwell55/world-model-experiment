# models/belief_state.py
from pydantic import BaseModel, Field
from typing import ClassVar
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

        Uses predictive distribution that accounts for both:
        1. Measurement noise (constant variance)
        2. Model uncertainty (grows with time due to heating_rate_std)

        Predictive std = sqrt(measurement_noise^2 + (heating_rate_std * time)^2)
        """
        predicted_temp = self.base_temp + self.heating_rate_mean * time_elapsed

        if 'measured_temp' not in observation:
            return 0.0  # No observation to evaluate

        # Predictive uncertainty combines measurement noise and model uncertainty
        # Model uncertainty grows with time: heating_rate_std * time
        predictive_variance = (
            self.measurement_noise ** 2 +
            (self.heating_rate_std * time_elapsed) ** 2
        )
        predictive_std = np.sqrt(predictive_variance)

        return stats.norm.logpdf(
            observation['measured_temp'],
            loc=predicted_temp,
            scale=predictive_std
        )
    
    def update(self, observation: dict, time_elapsed: float) -> 'HotPotBelief':
        """
        Bayesian update (simplified conjugate prior for demo).
        In practice, LLM will update parameters via reasoning.
        """
        if 'measured_temp' not in observation:
            return self

        # Handle initial observation (time_elapsed = 0)
        if time_elapsed == 0 or time_elapsed < 1e-6:
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
    temperature: str = Field(default='medium')  # Track temperature state

    # Temperature modifiers match environment (ClassVar to avoid Pydantic field inference)
    TEMP_MODIFIERS: ClassVar[dict] = {
        'low': {'success': 0.7, 'explode': 0.5, 'nothing': 1.3},
        'medium': {'success': 1.0, 'explode': 1.0, 'nothing': 1.0},
        'high': {'success': 1.2, 'explode': 2.0, 'nothing': 0.5}
    }

    def log_likelihood(self, observation: dict) -> float:
        """
        P(outcome | reaction, belief) with temperature effects.

        Properly applies outcome-specific temperature modifiers and normalizes.
        """
        reaction = observation.get('reaction')
        outcome = observation.get('outcome')

        if not reaction or not outcome:
            return 0.0

        if reaction not in self.reaction_probs:
            return np.log(0.01)  # Small prior for unknown reactions

        # Get base probabilities for this reaction
        base_probs = self.reaction_probs[reaction]

        # Apply temperature modifiers (matching environment logic)
        temp_mod = self.TEMP_MODIFIERS[self.temperature]
        adjusted_probs = {}

        for out, base_prob in base_probs.items():
            if out == 'explode':
                adjusted_probs[out] = base_prob * temp_mod['explode']
            elif out == 'nothing':
                adjusted_probs[out] = base_prob * temp_mod['nothing']
            else:  # Successful product (C, D, etc.)
                adjusted_probs[out] = base_prob * temp_mod['success']

        # Normalize probabilities
        total = sum(adjusted_probs.values())
        if total > 0:
            normalized_probs = {k: v / total for k, v in adjusted_probs.items()}
        else:
            # Fallback to uniform if all zero
            normalized_probs = {k: 1.0 / len(adjusted_probs) for k in adjusted_probs}

        # Get probability of observed outcome
        prob = normalized_probs.get(outcome, 0.01)

        return np.log(prob + 1e-10)

    def update(self, observation: dict, time_elapsed: float = 0.0) -> 'ChemTileBelief':
        """
        Update belief based on observation.

        Handles:
        1. Temperature changes (heat/cool actions)
        2. Reaction outcomes (update reaction_probs based on results)

        Args:
            observation: Environment observation
            time_elapsed: Time elapsed (not used for ChemTile, but kept for consistency)

        Returns:
            Updated ChemTileBelief instance
        """
        EPSILON = 1e-10  # Small constant to prevent division by zero

        # Create a copy of current parameters
        updated_params = self.model_dump()

        # Update temperature state based on observation
        if 'temperature' in observation:
            updated_params['temperature'] = observation['temperature']

        # Update reaction probabilities based on observed outcomes
        # This is a simple Bayesian update for categorical distributions
        if 'reaction' in observation and 'outcome' in observation:
            reaction = observation['reaction']
            outcome = observation['outcome']

            # Initialize reaction if not seen before
            if reaction not in updated_params['reaction_probs']:
                updated_params['reaction_probs'][reaction] = {
                    'explode': 0.33,
                    'nothing': 0.33,
                    'product': 0.34
                }

            # Get current probabilities for this reaction
            current_probs = updated_params['reaction_probs'][reaction].copy()

            # Check for degenerate case: all probabilities are zero
            total_prob = sum(current_probs.values())
            if total_prob < EPSILON:
                # Reinitialize to uniform distribution over known outcomes
                num_outcomes = len(current_probs)
                if num_outcomes > 0:
                    uniform_prob = 1.0 / num_outcomes
                    current_probs = {k: uniform_prob for k in current_probs.keys()}
                else:
                    # No outcomes at all - add the observed one
                    current_probs = {outcome: 1.0}

            # Add the observed outcome if it's not in our model
            if outcome not in current_probs:
                # Add with small initial probability
                current_probs[outcome] = 0.01
                # Renormalize existing probabilities with epsilon guard
                total = sum(current_probs.values())
                current_probs = {k: v / max(total, EPSILON) for k, v in current_probs.items()}

            # Simple Bayesian update: increase probability of observed outcome
            # This is a simplified version - in practice, LLM can do more sophisticated updates
            learning_rate = 0.1
            num_outcomes = len(current_probs)

            # Handle edge case: only one outcome (no redistribution needed)
            if num_outcomes == 1:
                # Nothing to update - already at 100%
                current_probs[outcome] = 1.0
            else:
                for key in current_probs:
                    if key == outcome:
                        # Increase probability of observed outcome
                        current_probs[key] = current_probs[key] * (1 + learning_rate)
                    else:
                        # Decrease other outcomes proportionally
                        # Safe division: max(num_outcomes - 1, 1) prevents division by zero
                        current_probs[key] = current_probs[key] * (1 - learning_rate / max(num_outcomes - 1, 1))

                # Normalize with epsilon guard to prevent division by zero
                total = sum(current_probs.values())
                current_probs = {k: v / max(total, EPSILON) for k, v in current_probs.items()}

            updated_params['reaction_probs'][reaction] = current_probs

        return ChemTileBelief(**updated_params)