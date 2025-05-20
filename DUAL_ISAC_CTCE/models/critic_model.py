# ===================================================================
# File: critic_model.py
# Description: Centralised critic model for multi-agent SAC using PyTorch.
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
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Initialises the centralised critic model.
    
        Parameters:
            obs_space (gym.Space): The observation space.
            action_space (gym.Space): The action space.
            num_outputs (int): Number of output nodes.
            model_config (dict): Configuration dictionary passed by RLlib.
            name (str): Name of the model instance.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.local_obs_size = 18
        self.target_obs_size = 24
        self.agents_local_obs_size = self.local_obs_size * 2
        self.global_obs_size = self.agents_local_obs_size + self.target_obs_size
        self.actuator_size = 4

        critic_input_dim = self.global_obs_size + self.actuator_size

        """
        # PETTINGZOO
        self.obs_dim = obs_space.shape[0]
        self.act_dim = action_space.shape[0]
        print("[CRITIC] action_space.shape[0]", self.act_dim)
        
        self.num_agents = 2
        self.global_obs_size = self.obs_dim * self.num_agents
        self.global_act_size = self.act_dim * self.num_agents

        critic_input_dim = self.global_obs_size + self.global_act_size
        """

        self.critic_net = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass of the centralised critic network.
    
        Parameters:
            input_dict (Dict): Dictionary containing:
                - "obs": A tuple of (obs_dict, action_tensor)
                    - obs_dict includes per-agent observations.
                    - action_tensor is the joint action vector.
            state (List): RNN state.
            seq_lens (Tensor): Not used (for RNN compatibility).
    
        Returns:
            Tuple[Tensor, List]: 
                - q_value: Estimated Q-value from the critic.
                - state: Updated RNN state after the forward pass.
        """

        obs, actions = input_dict["obs"]
        agent_id = obs["agent_id"] 
            
        local_obs_0 = obs["local_obs_0"]
        local_obs_1 = obs["local_obs_1"]
        target_obs = obs["target_obs"]

        critic_input = torch.cat([local_obs_0, local_obs_1, target_obs, actions], dim=-1)

        q_value = self.critic_net(critic_input)
        
        return q_value, state

