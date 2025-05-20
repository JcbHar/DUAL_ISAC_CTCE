# ===================================================================
# File: dual_sac_agent.py
# Description: SAC-based multi-agent trainer for Franka Panda robots.       
#
# Author: Jacob Stephens
# Email: psyjs32@nottingham.ac.uk
# Created: April 2025
# University: University of Nottingham
#
# Dependencies:
# - mujoco
# - ray [rllib]
# - numpy
# - matplotlib 
# - pandas
#
# ===================================================================


import os
import ray
import time
import numpy as np
from ray import tune
from ray.rllib.models import ModelCatalog
from models.actor_model import ActorModel
from models.critic_model import CriticModel
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from wrappers.multi_agent_mpe_wrapper import MPEMultiAgentWrapper
from wrappers.multi_agent_pandas_wrapper import DualPandaParallelEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch


class DualPandaSACAgent:
    def __init__(self, env_name="dual-panda-env-v1", env_instance=None):
        """
        Initialises the training interface for the DualPandaParallelEnv environment using RLlib's SAC algorithm.
    
        Parameters:
            env_name (str): Name used to register the environment with RLlib (default: "dual-panda-env-v1").
    
        This setup includes:
            - Creating an instance of the DualPandaParallelEnv environment.
            - Registering custom actor and critic models.
            - Defining the SAC configuration with appropriate parameters.
            - Setting up multi-agent policies for each robot.
            - Preparing for training by initialising the config.
        """
        self.env_name = env_name
        self.env = env_instance if env_instance else DualPandaParallelEnv(
            model_path="models/two_robots_one_cable/dual_panda_cable_sim.xml"
        )

        
        ModelCatalog.register_custom_model("actor_model", ActorModel)
        ModelCatalog.register_custom_model("critic_model", CriticModel)
        """
        # PETTINGZOO MPE CONFIG
        obs_spaces = {
            agent: self.env.env.observation_space(agent)
            for agent in self.env.env.possible_agents
        }
        act_spaces = {
            agent: self.env.env.action_space(agent)
            for agent in self.env.env.possible_agents
        }
    
        self.policy_mapping = {
            "adversary_0": "adversary_policy",
            "adversary_1": "adversary_policy",
            "agent_0": "good_policy",
            "agent_1": "good_policy",
        }
    
        self.config = (
            SACConfig()
            .environment(env=self.env_name, disable_env_checking=True)
            .rollouts(rollout_fragment_length=25, num_rollout_workers=4)
            .training(
                replay_buffer_config={"learning_starts": 1000},
                gamma=0.9,
                train_batch_size=64,
                optimization_config={
                    "actor_learning_rate": 0.001,
                    "critic_learning_rate": 0.001,
                    "entropy_learning_rate": 0.2,
                },
            )
            .multi_agent(
                policies={
                    "adversary_policy": PolicySpec(
                        None,
                        obs_spaces["adversary_0"],
                        act_spaces["adversary_0"]
                    ),
                    "good_policy": PolicySpec(
                        None,
                        obs_spaces["agent_0"],
                        act_spaces["agent_0"]
                    ),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: self.policy_mapping[agent_id],
            )
            .framework("torch")
            .resources(num_gpus=0)
        )
        """
        # DUAL PANDA CONFIG
        self.config = (
            SACConfig()
            .environment(env=self.env_name)
            .rollouts(rollout_fragment_length=25, num_rollout_workers=4)
            .training(
                replay_buffer_config={
                    "learning_starts": 1_000
                },
                gamma=0.99,
                train_batch_size=64,
                policy_model_config={
                    "custom_model": "actor_model",
                },
                q_model_config={
                    "custom_model": "critic_model",
                },
                optimization_config={
                    "actor_learning_rate": 3e-2,
                    "critic_learning_rate": 3e-2,
                    "entropy_learning_rate": 1e-3,
                },
            )
            .multi_agent(
                policies={
                    "robot_0": PolicySpec(
                        None,
                        self.env.observation_space,
                        self.env.action_space,
                    ),
                    "robot_1": PolicySpec(
                        None,
                        self.env.observation_space,
                        self.env.action_space,
                    ),
                },                
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            )
            .framework("torch")
            .resources(num_gpus=0)
        )

        self.trainer = None
        

    def train(self):
        """
        Trains a Soft Actor-Critic (SAC) policy using Ray RLlib.
    
        Returns:
            Tuple[str, ray.tune.ExperimentAnalysis]:
                - best_checkpoint (str): Path to the best-performing checkpoint.
                - results (ExperimentAnalysis): Full training results object from Ray Tune.
    
        This method:
            - Shuts down any previous Ray sessions.
            - Initialises Ray with specified resources.
            - Runs training with the current config for 200 iterations.
            - Restores the best-performing checkpoint into self.trainer.
        """
        ray.shutdown()
        local_logs_dir = "D:/dissertation/panda_multi_agent/logs"
        
        ray.init(ignore_reinit_error=True, num_cpus=5)
                
        results = tune.run(
            "SAC",
            config=self.config.to_dict(),
            stop={
                "training_iteration": 1_000,
            },
            local_dir=local_logs_dir,
            checkpoint_freq=250,
            checkpoint_at_end=True,
        )

        best_checkpoint = results.get_best_checkpoint(
            results.trials[0], metric="episode_reward_mean", mode="max"
        )
        
        self.trainer = self.config.build()
        print("[DEBUG] Final config:", self.config.to_dict())
        self.trainer.restore(best_checkpoint)
        
        return best_checkpoint, results

    
    def compute_action(self, obs, policy_id="robot_0"):
        """
        Computes an action for a single agent given its observation and policy ID.
    
        Parameters:
            obs (np.ndarray or dict): The observation input for the agent.
            policy_id (str): The ID of the policy to use (default: "robot_0").
    
        Returns:
            np.ndarray: The computed action for the agent.
    
        Raises:
            ValueError: If the trainer has not been initialised.
    
        """
        if self.trainer is None:
            raise ValueError("trainer not initialised.")
        return self.trainer.compute_single_action(obs, policy_id=policy_id)

    
    def compute_multi_agent_actions(self, obs_dict):
        """
        Computes actions for all agents based on their observations using the trained policies.
    
        Parameters:
            obs_dict (Dict[str, np.ndarray]): A dictionary mapping each agent ID to its observation.
    
        Returns:
            Dict[str, np.ndarray]: A dictionary mapping each agent ID to its computed action.
    
        Raises:
            ValueError: If the trainer has not been initialised.
        """
        if self.trainer is None:
            raise ValueError("trainer not initialised.")
        return {
            agent_id: self.trainer.compute_single_action(obs, policy_id=agent_id)
            for agent_id, obs in obs_dict.items()
        }

    
    def load_checkpoint(self, checkpoint_path):
        """
        Loads a previously saved checkpoint into the trainer.
    
        Parameters:
            checkpoint_path (str): Path to the checkpoint file to restore.
        """
        if self.trainer is None:
            self.trainer = self.config.build()
        self.trainer.restore(checkpoint_path)

    
    def simulate(self, env, num_episodes=1, render=False, test_threshold=0.05):
        """
        Runs evaluation episodes in the environment using a trained policy.
    
        Parameters:
            env (DualPandaParallelEnv): The environment to simulate.
            num_episodes (int): Number of evaluation episodes to run (default: 1).
            render (bool): If True, renders the environment during simulation.
            test_threshold (float): Distance threshold for considering an episode successful.
    
        Raises:
            ValueError: If the trainer has not been initialised before calling this method.
        """
        if self.trainer is None:
            raise ValueError("trainer not initialised.")
    
        agents = env.agents
        final_dists = {agent: [] for agent in agents}
        successes = {agent: 0 for agent in agents}
    
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
    
            while not done:
                actions = self.compute_multi_agent_actions(obs)
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
    
                if render:
                    env.render()
    
                done = truncateds.get("__all__", False) or all(terminateds.values())
    
            for agent in agents:
                dist = infos[agent].get("final_distance")
                if dist is not None:
                    final_dists[agent].append(dist)
                    if dist < test_threshold:
                        successes[agent] += 1
    
            print(f"Episode {episode + 1} finished.")
    
        for agent in agents:
            dists = final_dists[agent]
            if not dists:
                print(f"\n{agent}: No distance info recorded.")
                continue
    
            mean = np.mean(dists)
            std = np.std(dists)
            best = np.min(dists)
            success_rate = 100 * successes[agent] / num_episodes
    
            print(f"\nResults for {agent}:")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Mean ± σ: {mean:.3f} ± {std:.3f} m")
            print(f"Best: {best:.3f} m")

