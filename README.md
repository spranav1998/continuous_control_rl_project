[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Continuous_Control2.ipynb` to get started with training your own agent!  

## Report
### Introduction
In this project, we tackled the continuous control problem using a reinforcement learning approach. The aim was to train an agent to perform a task and reach a high average score.  
  
### Methodology  
We used the Deep Deterministic Policy Gradient (DDPG) algorithm, which is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. It was implemented in Python, using the PyTorch library for deep learning models.  
  
### The Model  
The model architecture we used for this project is the Deep Deterministic Policy Gradient (DDPG), an Actor-Critic model. It consists of two main components: the Actor and the Critic. Both of these components have been implemented as neural networks using PyTorch.   
  
Actor: The actor directly optimizes the policy function to produce the best action given a state. Our implementation of the actor is a neural network model composed of two hidden layers, a batch normalization layer, and an output layer. The hidden layers have 400 and 300 nodes, respectively. They take the current state as an input and output the action values, which are then passed through a tanh activation function to ensure the output values are within the valid action range. The weights of the neural network layers are initialized with a uniform distribution to enhance the stability of learning.
  
Critic: The critic calculates the Q-value of the current policy (provided by the Actor). The critic is also a neural network with two hidden layers, a batch normalization layer, and an output layer. The hidden layers have 400 and 300 nodes, respectively. They take the current state and action as input, where the state is passed through the first hidden layer and the action is concatenated before passing through the second hidden layer. The output is a single Q-value indicating the expected return. Similar to the Actor, the Critic's weights are also initialized with a uniform distribution to ensure the stability of learning.
  
The networks are trained simultaneously, with the Critic network learning the Q-value of the current policy and the Actor network aiming to maximize these expected returns.
  
### Results
The model was trained over 107 episodes, with the average and current scores for each episode reported. The training results showed a consistent improvement of the agent's performance over time, as reflected by the increasing average score across the episodes.
  
Here is a snapshot of the training results:
`
Episode 1	Average Score: 0.71 	Current Score: 0.71
...
Episode 50	Average Score: 20.33 	Current Score: 39.34
...
Episode 100	Average Score: 29.19 	Current Score: 38.04
...
Episode 107	Average Score: 31.81 	Current Score: 38.27
`
By the 107th episode, the model achieved an average score of 31.81 over the last 100 episodes, which is considered a benchmark score for many continuous control tasks.  

### Conclusion
The DDPG agent performed exceptionally well in this task, reaching an average score of 31.81 by the 107th episode. These results suggest that our model is capable of learning to perform complex continuous control tasks effectively.

This project highlights the power and potential of reinforcement learning for solving complex control tasks that have continuous state and action spaces.
