# Algorithmic Market Making: Q-Learning vs DDPG

This repository contains the code and thesis for the Masters thesis:

**"Algorithmic Pricing with AI Agents: A Comparative Analysis of DeepReinforcement Learning Approaches"**  
Aditya Goswami
University of Bonn
2026

## Abstract

This thesis conducts experiments in which Q-learning and Deep Deterministic Policy Gradient (DDPG) algorithms act as market makers in a Glosten-Milgrom market making game. The results show that Q-learning algorithms consistently converge to supra-competitive prices consistent with implicit collusion, while DDPG converges to 
prices close to the Nash equilibrium benchmark. This suggests that algorithm architecture is a first-order determinant of market quality in algorithmic market making.

## How to Reproduce Results

### Q-Learning

Open `code/Qlearning.py`,`code/qlearning_nas.py`,`code/qlearning_N,py` and run all cells. The simulation 
runs $K = 10$ independent experiments each with a maximum of $T_Q = 500,000$ iterations.Note: 'nas' corresponds to 'no adverse selection' and 'N' corresponds to 'N number of dealers'

### DDPG

Open `code/DDPG.py`, `code/DDPG_nas.py`, `code/DDPG_N.py` and run all cells. The simulation runs $K = 10$ independent experiments each with $T_D = 154,000$ iterations, calibrated to the mean convergence iteration of the Q-learning algorithm. Note: 'nas' corresponds to 'no adverse selection' and 'N' corresponds to 'N number of dealers'

### Generating the Plots

After running both simulations, generate the plots by running the `code/compare_plot.py`. The convergence table and loss plots are in a different file named `code/qlearn_plot.py` and `code/ddpg_plot.py` respectively.

