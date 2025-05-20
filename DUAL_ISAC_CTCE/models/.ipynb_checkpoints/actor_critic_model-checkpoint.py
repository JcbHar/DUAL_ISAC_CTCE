# ================================================
# NOTE: This model is not functional and does not work.
# ================================================


# import torch
# import torch.nn as nn
# from ray.rllib.models.modelv2 import ModelV2
# from ray.rllib.utils.annotations import override
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


# class ActorCriticModel(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)
        
#         local_obs_size = 18 
#         global_obs_size = 60
        
#         actor_hidden_size = 32
#         critic_hidden_size = 128
        
#         self.actor_net = nn.Sequential(
#             nn.Linear(local_obs_size, actor_hidden_size),
#             nn.ReLU(),
#             nn.Linear(actor_hidden_size, num_outputs)
#         )

#         self.critic_net = nn.Sequential(
#             nn.Linear(global_obs_size, critic_hidden_size),
#             nn.ReLU(),
#             nn.Linear(critic_hidden_size, 1)
#         )

#         self._value_out = None


#     @override(ModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         local_obs = input_dict["obs"].float()
#         global_obs = input_dict["global_obs"].float()

#         actor_output = self.actor_net(local_obs)
#         self._value_out = self.critic_net(global_obs)
#         return actor_output, state


#     @override(ModelV2)
#     def value_function(self):
#         return self._value_out.squeeze(-1)

