# ===============================================================
# NOTE: This model is not functional and does not currently work.
# ===============================================================


# import torch
# import torch.nn as nn
# from ray.rllib.models.modelv2 import ModelV2
# from ray.rllib.utils.annotations import override
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


# class CentralisedCriticModel(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         print("[MODEL INIT] CentralisedCriticModel initialised")
        
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)

#         self._last_logits = {}
        
#         robot_local_obs_size = 18
#         robot_global_obs_size = 60
#         robot_action_size = 8

#         self.global_obs_size = robot_global_obs_size
#         self.actuator_size = 2 * robot_action_size

#         self.actor_nets = nn.ModuleDict({
#             "robot_0": nn.Sequential(
#                 nn.Linear(robot_local_obs_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 16)
#             ),
#             "robot_1": nn.Sequential(
#                 nn.Linear(robot_local_obs_size, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 128),
#                 nn.ReLU(),
#                 nn.Linear(128, 16)
#             )
#         })

#         critic_input_dim = self.global_obs_size + self.actuator_size
#         self.critic_net = nn.Sequential(
#             nn.Linear(critic_input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )


#     def forward(self, input_dict, state, seq_lens):
#         obs_input = input_dict["obs"]
    
#         if isinstance(obs_input, tuple):
#             obs_input = obs_input[0]
    
#         if not isinstance(obs_input, dict):
#             raise ValueError(f"[ERROR] Expected dict, got: {type(obs_input)}")
    
#         agent_id = list(obs_input.keys())[0]
#         agent_obs = obs_input[agent_id].float()
    
#         local_obs = agent_obs[:, :32]
#         global_obs = agent_obs
    
#         logits = self.actor_nets[agent_id](local_obs)
    
#         self._current_agent_id = agent_id
#         self._current_logits = logits
#         self._current_obs = global_obs
    
#         return logits, state


#     def get_q_values(self, obs_tensor, action_tensor):
#         q_input = torch.cat([obs_tensor, action_tensor], dim=1)
#         q_values, _ = self.q_net({"obs": q_input})  # q_net is undefined
#         return q_values


#     def value_function(self):
#         critic_input = self._current_obs
#         return self.critic_net(critic_input).squeeze(-1)


#     def get_action_model_outputs(self, model_out, state_in=None, seq_lens=None):
#         return self.action_model({"obs": model_out}, state_in, seq_lens)  # action_model is undefined

