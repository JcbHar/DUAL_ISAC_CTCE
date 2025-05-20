# ===================================================================
# File: multi_agent_pandas_wrapper.py
# Description: SAC-based multi-agent trainer for Franka Panda robots.       
#
# Author: Jacob Stephens
# Email: psyjs32@nottingham.ac.uk
# University: University of Nottingham
#
# Dependencies:
# - mujoco
# - numpy
# - gymnasium
# - ray[rllib]
#
# ===================================================================


import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
import mujoco
from mujoco import viewer
from utils.methods import generate_snake_U_targets
from gymnasium.spaces import Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class DualPandaParallelEnv(MultiAgentEnv):
    """
    A parallel multi-agent MuJoCo environment for dual Franka Panda robots
    collaboratively manipulating a DLO.

    Inherits from:
        MultiAgentEnv

    Key Features:
        - Two agents: "robot_0" and "robot_1", each controlling a Franka Panda arm.
        - Designed for cooperative MARL with a shared objective.
        - Deformable object (rope) modelled using MuJoCo body segments.
        - Target alignment reward encourages precise rope placement.
        - Commented out inverse kinematics used to assist with rope end tracking.
        - Custom observation and action spaces with per-agent views.

    Core Methods:
        - __init__: Environment and model setup.
        - step: Executes one control step using agent actions.
        - reset: Resets the simulation and environment state.
        - render: Displays the current simulation state using MuJoCo viewer.
        - _compute_reward: Reward based on rope-to-target alignment.
        - _get_obs: Constructs observations.
        - _check_done: Checks if the rope segments are correctly aligned with targets.
        - change_target_pos: Dynamically updates target body positions.
    """
    def __init__(self, model_path, max_steps=500):
        """
        Initialises the dual-arm rope manipulation environment using MuJoCo.
    
        Parameters:
            model_path (str): Path to the MuJoCo XML model file.
            max_steps (int): Maximum number of simulation steps per episode (default: 500).
        
        This setup includes:
            - Loading the MuJoCo model and data.
            - Setting up robot configurations, actuator indices, and joint ranges.
            - Initialising cable segment and target body IDs.
            - Preparing observation and action spaces.
            - Generating initial target positions.
        """
        self.model_path = model_path
        self.max_steps = max_steps
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        for i in range(self.model.nu):
            name_address = self.model.name_actuatoradr[i]
            name = self.model.names[name_address:].split(b'\x00', 1)[0].decode("utf-8")
        
        self.current_step = 0
        self.viewer = None
        self.target_positions = None
        self.current_agent = 0

        self.agents = ["robot_0", "robot_1"]
        
        self.robot_configs = {}
        def compute_robot_config(joint_start, joint_end, actuator_start, actuator_end):
            nq = self.model.jnt_qposadr[joint_end] - self.model.jnt_qposadr[joint_start] + 1
            nv = self.model.jnt_dofadr[joint_end] - self.model.jnt_dofadr[joint_start] + 1
            nu = actuator_end - actuator_start + 1
            actuator_ids = list(range(actuator_start, actuator_end + 1))
            return {"nq": nq, "nv": nv, "nu": nu, "actuator_ids": actuator_ids}
        
        # ROBOT 0
        joint_start_name_0 = "joint1_01"
        joint_end_name_0   = "finger_joint2_01"
        actuator_start_name_0 = "actuator4_01"
        actuator_end_name_0   = "actuator7_01"
        
        self.robot_configs["robot_0"] = compute_robot_config(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_start_name_0),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_end_name_0),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_start_name_0),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_end_name_0),
        )

        print("robot_0 actuator ids:", self.robot_configs["robot_0"]["actuator_ids"])
        
        print("Robot 0 config:")
        print(f"  Joints: {joint_start_name_0} -> {joint_end_name_0}")
        print(f"  Actuators: {actuator_start_name_0} -> {actuator_end_name_0}")
        print("  Computed nq:", self.robot_configs["robot_0"]["nq"],
              "nv:", self.robot_configs["robot_0"]["nv"],
              "nu:", self.robot_configs["robot_0"]["nu"])
        
        # ROBOT 1
        joint_start_name_1 = "joint1_02"
        joint_end_name_1   = "finger_joint2_02"
        actuator_start_name_1 = "actuator4_02"
        actuator_end_name_1   = "actuator7_02"
        
        self.robot_configs["robot_1"] = compute_robot_config(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_start_name_1),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_end_name_1),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_start_name_1),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_end_name_1),
        )

        print("robot_1 actuator ids:", self.robot_configs["robot_1"]["actuator_ids"])
        
        print("Robot 1 config:")
        print(f"  Joints: {joint_start_name_1} -> {joint_end_name_1}")
        print(f"  Actuators: {actuator_start_name_1} -> {actuator_end_name_1}")
        print("  Computed nq:", self.robot_configs["robot_1"]["nq"],
              "nv:", self.robot_configs["robot_1"]["nv"],
              "nu:", self.robot_configs["robot_1"]["nu"])

        nq_0 = self.robot_configs["robot_0"]["nq"]
        nq_1 = self.robot_configs["robot_1"]["nq"]
        nv_0 = self.robot_configs["robot_0"]["nv"]
        nv_1 = self.robot_configs["robot_1"]["nv"]
        
        self.robot_indices = {
            "robot_0": {
                "qpos_start": 0,
                "qpos_end": nq_0,
                "qvel_start": 0,
                "qvel_end": nv_0,
            },
            "robot_1": {
                "qpos_start": nq_0,
                "qpos_end": nq_0 + nq_1,
                "qvel_start": nv_0,
                "qvel_end": nv_0 + nv_1,
            },
        }

        self.cable_seg_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"rope_seg{i}")
            for i in [10, 16, 24, 30]
        ]
        self.target_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"target{i}")
            for i in range(4, 8)
        ]

        self.target_obs_size = (len(self.target_ids) * 3) + (len(self.cable_seg_ids) * 3)
        print("[ENV INIT] self.target_obs_size: ", self.target_obs_size)
        self.local_obs_sizes = {
            agent: config["nq"] + config["nv"]
            for agent, config in self.robot_configs.items()
        }
        print("[ENV INIT] local observation sizes:")
        for agent, size in self.local_obs_sizes.items():
            print(f"  {agent}: {size}")

        self.observation_space = spaces.Dict({
            "local_obs_0": spaces.Box(low=-1.0, high=1.0, shape=(self.local_obs_sizes["robot_0"],)),
            "local_obs_1": spaces.Box(low=-1.0, high=1.0, shape=(self.local_obs_sizes["robot_1"],)),
            "target_obs": spaces.Box(low=-1.0, high=1.0, shape=(self.target_obs_size,)),
            "agent_id":    spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        total_obs_size = (
            self.observation_space.spaces["local_obs_0"].shape[0] +
            self.observation_space.spaces["local_obs_1"].shape[0] +
            self.observation_space.spaces["target_obs"].shape[0]
        )
        print("[ENV INIT] total observation size:", total_obs_size)
 
        self.finger_joint_names = [
            "finger_joint1_01", "finger_joint2_01",
            "finger_joint1_02", "finger_joint2_02"
        ]
        for joint_name in self.finger_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_index] = 0.00 # CLOSED FINGERS (0.04 OPEN)
        
        self.target_positions = generate_snake_U_targets(
            env=self,
            n_targets=4,
            x_bounds=(-0.52, 0.52),
            z_bounds=(0.6, 0.75),
            y_fixed=0.0,
            jitter=-0.06,
            spread_factor=0.87,
            seed=42
        )


    def _get_obs(self, agent):
        """
        Constructs the observation dictionary for the specified agent.
    
        Parameters:
            agent (str): The identifier of the agent/robot.
    
        Returns:
            Dict[str, np.ndarray]: A dictionary containing:
                - local_obs_0: Local observation of robot_0.
                - local_obs_1: Local observation of robot_1.
                - target_obs: Concatenated positions of cable segments and target positions.
                - agent_id: 1.0 or 0.
        """
        # ROBOT_0 LOCAL OBS
        qpos_0 = self.data.qpos[self.robot_indices["robot_0"]["qpos_start"]:
                               self.robot_indices["robot_0"]["qpos_end"]]
        qvel_0 = self.data.qvel[self.robot_indices["robot_0"]["qvel_start"]:
                               self.robot_indices["robot_0"]["qvel_end"]]
        local_obs_0 = np.concatenate([qpos_0, qvel_0], axis=0)

        # ROBOT_1 LOCAL OBS
        qpos_1 = self.data.qpos[self.robot_indices["robot_1"]["qpos_start"]:
                               self.robot_indices["robot_1"]["qpos_end"]]
        qvel_1 = self.data.qvel[self.robot_indices["robot_1"]["qvel_start"]:
                               self.robot_indices["robot_1"]["qvel_end"]]
        local_obs_1 = np.concatenate([qpos_1, qvel_1], axis=0)

        # TARGET OBS
        cable_positions = [self.data.xpos[seg_id] for seg_id in self.cable_seg_ids]
        target_positions = [self.data.xpos[tid] for tid in self.target_ids]
        target_obs = np.concatenate(cable_positions + target_positions, axis=0)

        if agent == "robot_0":
            self.current_agent = 0
        else:
            self.current_agent = 1
        return {
            "local_obs_0": np.clip(local_obs_0, -1.0, 1.0).astype(np.float32),
            "local_obs_1": np.clip(local_obs_1, -1.0, 1.0).astype(np.float32),
            "target_obs": np.clip(target_obs, -1.0, 1.0).astype(np.float32),
            "agent_id": np.array([0.0], dtype=np.float32) if agent == "robot_0" else np.array([1.0], dtype=np.float32),
        }


    def step(self, actions):
        """
        Executes a single environment step by applying actions, updating simulation state,
        and computing observations, rewards, and termination conditions.
    
        Parameters:
            actions (Dict[str, np.ndarray]): Dictionary mapping each agent to its action vector.
                                             Expected keys: "robot_0" and "robot_1".
    
        Returns:
            Tuple[
                Dict[str, np.ndarray],  |  observations
                Dict[str, float],       |  rewards
                Dict[str, bool],        |  terminateds
                Dict[str, bool],        |  truncateds
                Dict[str, dict]         |  infos
            ]
        """
        self.current_step += 1
    
        nu_0 = self.robot_configs["robot_0"]["nu"]
        nu_1 = self.robot_configs["robot_1"]["nu"]
    
        act_robot_0 = np.array(actions["robot_0"], dtype=np.float32).reshape(-1)
        act_robot_1 = np.array(actions["robot_1"], dtype=np.float32).reshape(-1)
        for j, aid in enumerate(self.robot_configs["robot_0"]["actuator_ids"]):
            self.data.ctrl[aid] = np.clip(act_robot_0[j], -1.0, 1.0)
        
        for j, aid in enumerate(self.robot_configs["robot_1"]["actuator_ids"]):
            self.data.ctrl[aid] = np.clip(act_robot_1[j], -1.0, 1.0)

        # INVERSE KINEMATIC
        #self.apply_hand_ik_to_rope_ends()
        
        mujoco.mj_step(self.model, self.data)

        for joint_name in self.finger_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_index] = 0.00
    
        obs = {
            "robot_0": self._get_obs("robot_0"),
            "robot_1": self._get_obs("robot_1"),
        }
    
        rewards = {
            agent: float(self._compute_reward(agent))
            for agent in self.agents
        }
        terminateds = {
            agent: self._check_done(agent)
            for agent in self.agents
        }
        terminateds["__all__"] = all(terminateds.values())
        truncateds = {
            agent: self.current_step >= self.max_steps
            for agent in self.agents
        }
        truncateds["__all__"] = self.current_step >= self.max_steps
        infos = {agent: {} for agent in self.agents}

        if terminateds["__all__"] or truncateds["__all__"]:
            green_positions = self._get_tracked_segment_positions()
            red_positions = [self.data.xpos[tid] for tid in self.target_ids]
            distances = [
                np.linalg.norm(green - red)
                for green, red in zip(green_positions, red_positions)
            ]
            avg_final_distance = float(np.mean(distances))
            for agent in self.agents:
                infos[agent]["final_distance"] = avg_final_distance

        for tname, pos in self.target_positions.items():
            self.change_target_pos(tname, *pos)
            
        return obs, rewards, terminateds, truncateds, infos
    

    def reset(self, *, seed=None, options=None):
        """
        Resets the simulation environment to its initial state.
    
        Parameters:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Environment-specific options for reset (currently unused).
    
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
                - obs: A dictionary mapping each agent to its initial observation.
                - infos: A dictionary containing empty info dicts for each agent.
        """
        if seed is not None:
            np.random.seed(seed)
    
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # INVERSE KINEMATIC
        #self.apply_hand_ik_to_rope_ends()
    
        self._update_robot_indices()
        
        obs = {
            "robot_0": self._get_obs("robot_0"),
            "robot_1": self._get_obs("robot_1"),
        }
    
        infos = {agent: {} for agent in self.agents}
    
        return obs, infos

    
    def _compute_reward(self, agent):
        """
        Computes the reward for the given agent based on the alignment of tracked rope segments
        with their corresponding target positions.
    
        Parameters:
            agent (str): The identifier of the agent/robot (currently unused).
    
        Returns:
            float: The computed reward value. The reward is negatively correlated with the
                   average distance between the tracked rope segments and their target positions,
                   with a bonus added for near threshold distance.
        """
        reward = 0

        green_positions = self._get_tracked_segment_positions()
        red_positions = [self.data.xpos[tid] for tid in self.target_ids]
    
        distances = [
            np.linalg.norm(green_pos - red_pos)
            for green_pos, red_pos in zip(green_positions, red_positions)
        ]

        avg_distance = np.mean(distances)
        #reward += 1 / (1 + avg_distance)
        reward -= avg_distance

        if avg_distance < 0.05:
            reward += 10

        return reward


    def _check_done(self, agent):
        """
        Checks whether all tracked cable segments are close to the target positions.
    
        Parameters:
            agent (str): The identifier of the agent/robot.
    
        Returns:
            bool: True if every cable segment is within a threshold distance, else False.
        """
        threshold = 0.05

        for seg_id in self.cable_seg_ids:
            seg_pos = self.data.xpos[seg_id]
            if not any(np.linalg.norm(seg_pos - self.data.xpos[target_id]) < threshold 
                       for target_id in self.target_ids):
                return False
        return True
            
            
    def get_local_obs_size(self, agent):
        """
        Returns the size of the local observation vector for specified agent.
    
        Parameters:
            agent (str): Identifier for the agent/robot.
    
        Returns:
            int: The dimensionality of the agent's local observation space.
        """
        return self.local_obs_sizes[agent]

    
    def get_global_obs_size(self, agent):
        """
        Returns the size of the global observation vector for specified agent.
    
        Parameters:
            agent (str): Identifier for the agent/robot.
    
        Returns:
            int: The dimensionality of the agent's global observation space.
        """
        return self.global_obs_sizes[agent]
    

    def get_total_obs_size(self):
        """
        Returns the total size of the observation vector.
    
        Returns:
            int: The total dimensionality of the combined observation space.
        """
        return self.total_obs_size

    
    def render(self, mode="human"):
        """
        Renders the simulation using the MuJoCo viewer.
    
        Parameters:
            mode (str): Rendering mode.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    
    def change_target_pos(self, target_name, x, y, z):
        """
        Updates the position of a target body in the simulation by modifying its associated free joint.
    
        Parameters:
            target_name (str): The base name of the target body.
            x (float): New x-coordinate.
            y (float): New y-coordinate.
            z (float): New z-coordinate.
    
        Raises:
            ValueError: If the joint associated with the target does not have at least 3 DoF.
        """
        joint_name = f"{target_name}_free"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qposadr = self.model.jnt_qposadr[joint_id]
    
        if joint_id < self.model.njnt - 1:
            qpos_num = self.model.jnt_qposadr[joint_id + 1] - self.model.jnt_qposadr[joint_id]
        else:
            qpos_num = self.model.nq - self.model.jnt_qposadr[joint_id]
    
        if qpos_num < 3:
            raise ValueError(f"expected at least 3 DoF for a free joint, got {qpos_num}.")
    
        self.data.qpos[qposadr:qposadr+3] = np.array([x, y, z], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)


    def _get_tracked_segment_positions(self):
        """
        Retrieves the world positions of all tracked cable segments.
    
        Returns:
            List[np.ndarray]: A list of 3D position vectors corresponding to self.cable_seg_ids.
        """
        return [self.data.xpos[seg_id].copy() for seg_id in self.cable_seg_ids]

        
    def _update_robot_indices(self):
        """
        Computes and stores the joint and velocity index ranges for each robot agent.
        - Iterates through all self.agents.
        - Uses agent's self.robot_configs to determine the number of position and velocity variables.
        - Updates self.robot_indices with start and end indices for both position and velocity.
        """
        self.robot_indices = {}
        current_qpos = 0
        current_qvel = 0
    
        for agent in self.agents:
            nq = self.robot_configs[agent]["nq"]
            nv = self.robot_configs[agent]["nv"]
    
            self.robot_indices[agent] = {
                "qpos_start": current_qpos,
                "qpos_end": current_qpos + nq,
                "qvel_start": current_qvel,
                "qvel_end": current_qvel + nv
            }
    
            current_qpos += nq
            current_qvel += nv


    def solve_inverse_kinematics(self, agent, target_pos, max_iters=50, tol=1e-3):
        """
        Solves inverse kinematics (IK) for a given robot hand to reach a specified target position.
        
        Parameters:
            agent (str): Agent to identify the robot.
            target_pos (np.ndarray): 3D world coordinates that the robot hand should reach.
            max_iters (int): Maximum number of IK iterations (default: 50).
            tol (float): Position tolerance threshold.
    
        Returns:
            np.ndarray: Joint position configurationfor the agent that moves its hand near the target_pos.
        """
        hand_name = "hand_1_01" if agent == "robot_0" else "hand_1_02"
        hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
    
        qpos_start = self.robot_indices[agent]["qpos_start"]
        qpos_end = self.robot_indices[agent]["qpos_end"]
        nv_start = self.robot_indices[agent]["qvel_start"]
        nv_end = self.robot_indices[agent]["qvel_end"]
        
        qpos = self.data.qpos.copy()
    
        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.xpos[hand_body_id]
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                break
    
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, hand_body_id)
    
            jac = jacp[:, nv_start:nv_end]
    
            dq = np.linalg.pinv(jac.T @ jac + 1e-6 * np.eye(jac.shape[1])) @ jac.T @ error
    
            qpos[qpos_start:qpos_end] += dq
    
            self.data.qpos[:] = qpos
    
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[qpos_start:qpos_end].copy()


    def apply_hand_ik_to_rope_ends(self):
        """
        Applies inverse kinematics (IK) to position each robot hand at opposite ends of a deformable rope.
        - Robot 0 is assigned to reach the rope root ("rope_root").
        - Robot 1 is assigned to reach the far end of the rope ("rope_seg40").
        """
        target_pos_0 = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rope_root")]
        target_pos_1 = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rope_seg40")]
    
        ik_qpos_0 = self.solve_inverse_kinematics("robot_0", target_pos_0)
        self.data.qpos[self.robot_indices["robot_0"]["qpos_start"]:self.robot_indices["robot_0"]["qpos_end"]] = ik_qpos_0
    
        ik_qpos_1 = self.solve_inverse_kinematics("robot_1", target_pos_1)
        self.data.qpos[self.robot_indices["robot_1"]["qpos_start"]:self.robot_indices["robot_1"]["qpos_end"]] = ik_qpos_1
    
        mujoco.mj_forward(self.model, self.data)

