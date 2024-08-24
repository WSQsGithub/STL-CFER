# Tractable Reinforcement Learning for Signal Temporal Logic Tasks with Counterfactual Experience Replay

## Abstract

We investigate the control synthesis problem for Markov decision processes (MDPs) with unknown transition probabilities under signal temporal logic (STL) specifications. Our primary objective is to learn a control policy that maximizes the probability of satisfying the STL task.  However, existing approaches to STL control synthesis using reinforcement learning encounter a significant scalability challenge, particularly when expanding the state space to incorporate STL tasks. In this work, we propose a novel reinforcement learning algorithm tailored for STL tasks, addressing the scalability issue by effectively leveraging **counterfactual experiences** to expedite the training process. In particular, we represent task progress using flag variables and introduce an approach to generate counterfactual experiences that are then replayed during the learning process. These generated counterfactual experiences enable us to fully employ the knowledge embedded within the task, resulting in a substantial reduction in the number of trial-and-error explorations required before achieving convergence. We conduct  a series of experiments and comparative analyses to we demonstrate the effectiveness and scalability of our algorithm compared with existing algorithms in the literature.

## Demonstration

[![Video]()](https://github.com/WSQsGithub/TimeRobustLearning/assets/70429350/5e0114a4-abae-408d-8909-5cd21170de1a)


## Comment

This paper has been accepted to L-CSS.
