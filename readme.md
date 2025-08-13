# Enhanced Verification of Safety and Security for Advanced Driver Assistance Systems
This repository is for the paper "Enhanced Verification of Safety and Security for Advanced Driver Assistance Systems" 

## System requirement
- Operating system: Linux
- Flow* version: 1.2.0
- CARLA version: 0.9.14
- Scenic version: 2.1.0
- Python ≥ 3.7

## Usage

- Data generation: run tools > flowstar.py to generate reachable sets

- DRL training: run Advanced_Ablation_Runner.py

- CARLA/Scenic

  - please download the repository from https://github.com/BerkeleyLearnVerify/Scenic

  - choose carlaChallenge10.scenic, and run benchmark_runner.py