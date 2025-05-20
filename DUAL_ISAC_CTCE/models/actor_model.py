# ===================================================================
# File: actor_model.py
# Description: Actor network for multi-agent SAC using PyTorch and RLlib.
#
# Author: Jacob Stephens
# Email: psyjs32@nottingham.ac.uk
# University: University of Nottingham
#
# Dependencies:
# - torch
# - ray[rllib]
# ===================================================================


import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override


class ActorModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Initialises the actor model for multi-agent SAC using PyTorch and RLlib.
    
        Parameters:
            obs_space (gym.Space): The observation space provided by the environment.
            action_space (gym.Space): The action space provided by the environment.
            num_outputs (int): The number of outputs.
            model_config (dict): Configuration dictionary from RLlib.
            name (str): Name of the model instance.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        print("[ACTOR INIT]")

        self.local_obs_size = 18 
        self.target_obs_size = 24
        self.actor_input_size = (self.local_obs_size * 2) + self.target_obs_size
        """
        self.obs_dim = obs_space.shape[0]
        self.actor_input_size = self.obs_dim
        """
        
        self.actor_net = nn.Sequential(
            nn.Linear(self.actor_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
            nn.Tanh()
        )
        print("[ACTOR] num_outputs", num_outputs)

        self._value_out = None


    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass of the actor network.
    
        Parameters:
            input_dict (Dict): Contains:
                - "obs": A dictionary with keys:
                    - "local_obs_0": Local observation vector of robot_0.
                    - "local_obs_1": Local observation vector of robot_1.
                    - "target_obs": Global target and cable positions.
            state (List): RNN state.
            seq_lens (Tensor): Unused (included for compatibility with RNNs).
    
        Returns:
            Tuple[Tensor, List]: 
                - action_outputs: The action vector predicted by the actor network.
                - state: Updated RNN state after the forward pass.
        """

        obs = input_dict["obs"]

        local_obs_0 = obs["local_obs_0"]
        local_obs_1 = obs["local_obs_1"]
        target_obs = obs["target_obs"]

        actor_input = torch.cat([local_obs_0, local_obs_1, target_obs], dim=-1)
        action_outputs = self.actor_net(actor_input)

        return action_outputs, state

