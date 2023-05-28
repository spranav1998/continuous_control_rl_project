import argparse
from unityagents import UnityEnvironment
import numpy as np
import torch

from agent import Agent

def main():
    parser = argparse.ArgumentParser(description='Select the UnityEnvironment for testing')
    parser.add_argument('--version', type=int, default=1, help='Version of the environment to use: 1 for single agent, 2 for 20 agents')
    args = parser.parse_args()

    if args.version == 1:
        env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe')
    elif args.version == 2:
        env = UnityEnvironment(file_name='./Reacher_Windows_x86_64_20/Reacher.exe')
    else:
        raise ValueError('Invalid version number. Please enter 1 for single agent or 2 for 20 agents')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name] # Change train_mode to False for testing

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    # Load trained model weights
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor20.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic20.pth'))

    scores_agents = np.zeros(num_agents)
    while True:
        actions = agent.act(states, add_noise=False) # Set add_noise=False for testing
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores_agents += env_info.rewards
        states = next_states
        if np.any(dones):
            break

    print('Total score for this episode: {}'.format(np.mean(scores_agents)))

    env.close()

if __name__ == "__main__":
    main()
