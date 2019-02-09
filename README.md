# RL-CBF
This code implements the RL-CBF algorithm on top of two baseline model-free algorithms: Trust Region Policy Optimization (TRPO) and Deep Deterministic Policy Gradients (DDPG). The RL-CBF algorithm provides safety guarantees during the learning process, and details of the algorithm can be found in the paper "End-to-End Safe Reinforcement Learning for Safety-Critical Continuous Control Tasks". We show learning on two simulated tasks: (1) Inverted Pendulum Control, (2) Car-Following in a chain of 5 cars.

Within each folder for each problem domain, there are 4 subfolders implementing the RL algorithms:
TRPO-CBF - Run the RL-CBF algorithm on top of TRPO. For the car-following example, run sim.py to begin learning. For the pendulum example, run main.py to begin learning. 
DDPG-CBF - Run the RL-CBF algorithm on top of DDPG. For both examples, run ddpg.py to begin learning.
TRPO - Run the baseline TRPO algorithm for comparison. For the car-following example, run sim.py to begin learning. For the pendulum example, run main.py to begin learning. 
DDPG - Run the baseline DDPG algorithm. For both examples, run ddpg.py to begin learning.

The files plotResults.m and plotCollisions.m can be run in MATLAB to generate plots of the reward and safety violations, respectively, as seen in the paper. However due to space constraints, the data files are not included, here, but all code with the data files can be found at: https://www.dropbox.com/sh/23t9ez2ho3wbp7g/AABGjN1GSvYkCEE8xA7cB707a?dl=0

Hyperparameters can be tuned in the sim.py or main.py files for car-following and pendulum, respectively. Once learning is finished (for the specified number of iterations), data is output in a .mat file, which includes rewards and trajectories for each episode. For any questions/issues regarding the code, contact rcheng@caltech.edu. 
