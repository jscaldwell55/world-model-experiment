# models/transition_model.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional


class SimpleTransitionModel(nn.Module):
    """
    2-layer MLP for predicting next observation.

    Learns P(next_state | state, action) from trajectories.

    Architecture:
        Input: [state_features, action_encoding]
        Hidden: 2 ReLU layers
        Output: next_state_features
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        learning_rate: float = 0.01
    ):
        """
        Initialize transition model.

        Args:
            state_dim: Dimensionality of state representation
            action_dim: Dimensionality of action encoding
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for optimizer
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]

        Returns:
            Predicted next state [batch_size, state_dim]
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def fit(
        self,
        trajectories: List[Tuple[Dict, str, Dict]],
        epochs: int = 50,
        verbose: bool = False
    ):
        """
        Fit model to trajectory data.

        Args:
            trajectories: List of (obs, action_str, next_obs) tuples
            epochs: Number of training epochs
            verbose: Whether to print training progress
        """
        if not trajectories:
            if verbose:
                print("Warning: No trajectories to fit")
            return

        # Convert to tensors
        states, actions, next_states = self._prepare_data(trajectories)

        # Check if we have valid data
        if states.shape[0] == 0:
            if verbose:
                print("Warning: No valid data after preparation")
            return

        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            predictions = self.forward(states, actions)
            loss = self.loss_fn(predictions, next_states)

            loss.backward()
            self.optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")

    def predict(self, state: Dict, action: str) -> Dict:
        """
        Predict next observation given current state and action.

        Args:
            state: Current observation dictionary
            action: Action string (e.g., "measure_temp()")

        Returns:
            Predicted next observation dictionary
        """
        state_vec = self._obs_to_vector(state)
        action_vec = self._action_to_vector(action)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            action_tensor = torch.FloatTensor(action_vec).unsqueeze(0)

            next_state_vec = self.forward(state_tensor, action_tensor)
            next_state_vec = next_state_vec.squeeze(0).numpy()

        # Convert back to dict
        return self._vector_to_obs(next_state_vec, state)

    def _prepare_data(
        self,
        trajectories: List[Tuple[Dict, str, Dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert trajectories to tensor format.

        Args:
            trajectories: List of (obs, action, next_obs) tuples

        Returns:
            Tuple of (states, actions, next_states) tensors
        """
        states_list = []
        actions_list = []
        next_states_list = []

        for obs, action, next_obs in trajectories:
            try:
                states_list.append(self._obs_to_vector(obs))
                actions_list.append(self._action_to_vector(action))
                next_states_list.append(self._obs_to_vector(next_obs))
            except Exception as e:
                # Skip invalid trajectories
                print(f"Warning: Skipping trajectory due to error: {e}")
                continue

        if not states_list:
            # Return empty tensors
            return (
                torch.zeros(0, self.state_dim),
                torch.zeros(0, self.action_dim),
                torch.zeros(0, self.state_dim)
            )

        return (
            torch.FloatTensor(states_list),
            torch.FloatTensor(actions_list),
            torch.FloatTensor(next_states_list)
        )

    def _obs_to_vector(self, obs: Dict) -> List[float]:
        """
        Convert observation dictionary to fixed-size vector.

        Extracts numerical features in sorted order.

        Args:
            obs: Observation dictionary

        Returns:
            Feature vector of size state_dim
        """
        vec = []

        # Extract numerical features in sorted order for consistency
        for key in sorted(obs.keys()):
            val = obs[key]

            if isinstance(val, (int, float)):
                vec.append(float(val))
            elif isinstance(val, bool):
                vec.append(1.0 if val else 0.0)
            # Skip non-numerical values

        # Pad or truncate to state_dim
        while len(vec) < self.state_dim:
            vec.append(0.0)

        return vec[:self.state_dim]

    def _action_to_vector(self, action: str) -> List[float]:
        """
        Convert action string to one-hot-like vector.

        Simple hash-based encoding of action names.

        Args:
            action: Action string (e.g., "measure_temp()")

        Returns:
            Action encoding of size action_dim
        """
        # Extract action name
        import re
        match = re.match(r'(\w+)', action)
        action_name = match.group(1) if match else action

        # Map common action names to indices
        action_names = [
            'measure', 'wait', 'touch', 'toggle',
            'flip', 'jiggle', 'inspect', 'observe',
            'mix', 'heat', 'cool'
        ]

        vec = [0.0] * self.action_dim

        # Find matching action
        for i, name in enumerate(action_names[:self.action_dim]):
            if name in action_name.lower():
                vec[i] = 1.0
                break

        return vec

    def _vector_to_obs(self, vec: np.ndarray, template: Dict) -> Dict:
        """
        Convert vector back to observation dictionary.

        Uses template to determine which keys to populate.

        Args:
            vec: Feature vector
            template: Template observation with keys

        Returns:
            Reconstructed observation dictionary
        """
        obs = {}
        vec_idx = 0

        # Reconstruct numerical values in sorted order
        for key in sorted(template.keys()):
            if isinstance(template[key], (int, float, bool)):
                if vec_idx < len(vec):
                    obs[key] = float(vec[vec_idx])
                    vec_idx += 1
                else:
                    obs[key] = 0.0
            else:
                # Keep non-numerical values from template
                obs[key] = template[key]

        return obs
