# DUAL_ISAC_CTCE

A dual-agent Soft Actor-Critic (SAC) implementation under a Centralised Training with Centralised Execution (CTCE) paradigm, applied to deformable object manipulation in a simulated environment.

## Specifications

- **Observation space:** 60-dimensional  
- **Local observation space:** 4-dimensional* 
- **Agents:** 2 (each controlling a separate robot arm)  
- **Algorithm:** Soft Actor-Critic (SAC)  
- **Architecture:** Independent actors with centralised critic (ISAC-CTCE)  
- **Environment:** Custom MuJoCo environment with deformable rope and target segments  

## Visual Overview

![Environment Overview](https://i.imgur.com/SJXScKj.png)  
*Figure: Dual-robot setup manipulating a deformable object toward dynamic targets*

## Key Features

- Centralised critic with access to global observations and actions  
- Independent actors for decentralised execution  
- Continuous control in a multi-agent setting  
- Reward shaped via inverse kinematics or object alignment

## Project Structure

```text
DUAL_ISAC_CTCE/
+-- agents/
|   +-- dual_sac_agent.py            # Main multi-agent SAC implementation
+-- logs/                             # TensorBoard logs and training outputs
+-- models/
|   +-- rwo_robots_one_cable/         # Environment XML
|   +-- actor_model.py                # Actor network architecture
|   +-- critic_model.py               # Critic network architecture
+-- utils/
|   +-- methods.py                    # Utility functions
+-- wrappers/
|   +-- multi_agent_mpe_wrapper.py    # Wrapper for MPE-style environments
|   +-- multi_agent_pandas_wrapper.py # Wrapper for dual Franka Panda setup
+-- train_test.ipynb                  # Jupyter notebook for training/testing
```

## Getting Started

```bash
git clone https://github.com/JcbHar/DUAL_ISAC_CTCE.git
cd DUAL_ISAC_CTCE
pip install -r requirements.txt
