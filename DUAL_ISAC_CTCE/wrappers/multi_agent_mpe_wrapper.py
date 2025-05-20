# ===================================================================
# File: mpe_multiagent_wrapper.py
# Description: Wrapper class to adapt PettingZoo-style environments 
#              for use with RLlib's MultiAgentEnv interface.
#
# Author: Jacob Stephens
# Email: psyjs32@nottingham.ac.uk
# University: University of Nottingham
#
# Dependencies:
# - gymnasium
# - ray[rllib]
# ===================================================================

from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MPEMultiAgentWrapper(MultiAgentEnv):
    """
    A wrapper to adapt PettingZoo multi-agent environments for use with RLlib.
    Handles mapping of observation and action spaces for each agent and formats
    outputs to match RLlib's expectations.
    """
    def __init__(self, env):
       """
        Initialise the wrapper with a PettingZoo-style environment.

        Parameters:
            env: A PettingZoo environment instance.
        """
        super().__init__()
        self.env = env
        if hasattr(env, "possible_agents") and env.possible_agents:
            self.agents = list(env.possible_agents)
        elif hasattr(env, "agents") and env.agents:
            self.agents = list(env.agents)
        else:
            self.agents = ["agent_0", "agent_1"]
        self._agent_ids = set(self.agents)

        if callable(env.observation_space):
            self.observation_spaces = {agent: env.observation_space(agent) for agent in self.agents}
        else:
            self.observation_spaces = {agent: env.observation_space for agent in self.agents}

        if callable(env.action_space):
            self.action_spaces = {agent: env.action_space(agent) for agent in self.agents}
        else:
            self.action_spaces = {agent: env.action_space for agent in self.agents}

        self.observation_space = self.observation_spaces[self.agents[0]]
        self.action_space = self.action_spaces[self.agents[0]]

    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return initial observations and info.

        Returns:
            obs (dict): Initial observations per agent.
            infos (dict): Additional info per agent.
        """
        reset_result = self.env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, infos = reset_result
        else:
            obs = reset_result
            infos = {}
        if not isinstance(obs, dict):
            obs = {agent: o for agent, o in zip(self.agents, obs)}
        if not isinstance(infos, dict):
            infos = {agent: {} for agent in self.agents}
        return obs, infos

    
    def step(self, action_dict):
        """
        Perform a single environment step.

        Parameters:
            action_dict (dict): Actions per agent.

        Returns:
            obs (dict): New observations per agent.
            rewards (dict): Rewards per agent.
            terminateds (dict): Episode termination flags.
            truncateds (dict): Truncation flags.
            infos (dict): Additional info per agent.
        """
        print("Sampled action (agent_0):", self.action_spaces["agent_0"].sample())
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)
        if not isinstance(obs, dict):
            obs = {agent: o for agent, o in zip(self.agents, obs)}
        for agent in action_dict:
            if agent in infos:
                infos[agent]["actions"] = action_dict[agent]
            else:
                infos[agent] = {"actions": action_dict[agent]}
                
        if "__all__" not in terminateds:
            terminateds["__all__"] = all(terminateds.values())
        if "__all__" not in truncateds:
            truncateds["__all__"] = all(truncateds.values())
        return obs, rewards, terminateds, truncateds, infos

    
    def render(self, mode="human"):
        """
        Render the environment.

        Parameters:
            mode (str): Rendering mode.
        """
        return self.env.render(mode=mode)

    
    def close(self):
        """
        Close the environment.
        """
        return self.env.close()


    def observation_space_sample(self, agent_ids=None):
        """
        Sample from the observation space of one or more agents.

        Parameters:
            agent_ids (str or list): Agent ID(s).

        Returns:
            dict or sample: A sample observation per agent or single agent.
        """
        if agent_ids is None:
            agent_ids = self.agents
        if isinstance(agent_ids, list):
            return {agent: self.observation_spaces[agent].sample() for agent in agent_ids}
        else:
            return self.observation_spaces[agent_ids].sample()

    
    def action_space_sample(self, agent_ids=None):
        """
        Sample from the action space of one or more agents.

        Parameters:
            agent_ids (str or list): Agent ID(s).

        Returns:
            dict or sample: A sample action per agent or single agent.
        """
        if agent_ids is None:
            agent_ids = self.agents
        if isinstance(agent_ids, list):
            return {agent: self.action_spaces[agent].sample() for agent in agent_ids}
        else:
            return self.action_spaces[agent_ids].sample()

    
    def observation_space_contains(self, x):
        """
        Check if a given observation is valid for all agents.

        Parameters:
            x (dict): Observation dict per agent.

        Returns:
            bool: True if all observations are valid.
        """
        return all(self.observation_spaces[agent].contains(x.get(agent)) for agent in self.agents)

    
    def action_space_contains(self, x):
        """
        Check if a given action is valid for all agents.

        Parameters:
            x (dict): Action dict per agent.

        Returns:
            bool: True if all actions are valid.
        """
        return all(self.action_spaces[agent].contains(x.get(agent)) for agent in self.agents)

    
    def simulate(self, algorithm, episodes=5, render=False):
        """
        Run simulation episodes using a given algorithm.

        Parameters:
            algorithm: Trained RLlib algorithm instance.
            episodes (int): Number of episodes to simulate.
            render (bool): Whether to render the environment.
        """
        for ep in range(episodes):
            obs, _ = self.reset()
            done = {"__all__": False}
            ep_reward = {agent: 0.0 for agent in self.agents}

            while not done["__all__"]:
                actions = {}
                for agent_id, obs_i in obs.items():
                    policy_id = algorithm.config.multi_agent_config["policy_mapping_fn"](agent_id)
                    action = algorithm.compute_single_action(obs_i, policy_id=policy_id)
                    actions[agent_id] = action

                obs, rewards, done, truncated, _ = self.step(actions)
                
                for agent_id, r in rewards.items():
                    ep_reward[agent_id] += r

                if render:
                    self.render()

            print(f"Episode {ep+1}: Total reward per agent: {ep_reward}")

