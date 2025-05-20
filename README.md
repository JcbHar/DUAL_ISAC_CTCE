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

![Environment Overview](https://i.imgur.com/yourimagename.png)
*Figure: Dual-robot setup manipulating a deformable object toward dynamic targets*

## Key Features

- Centralised critic with access to global observations and actions  
- Independent actors for decentralised execution  
- Continuous control in a multi-agent setting  
- Reward shaped via inverse kinematics or object alignment  

## Getting Started

```bash
git clone https://github.com/JcbHar/DUAL_ISAC_CTCE.git
cd DUAL_ISAC_CTCE
pip install -r requirements.txt
