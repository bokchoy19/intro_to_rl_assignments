# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from ShortCutAgents import QLearningAgent
from ShortCutAgents import SARSAAgent
from ShortCutAgents import ExpectedSARSAAgent
from ShortCutAgents import nStepSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutEnvironment import WindyShortcutEnvironment

def run_repetitions(n_reps, n_episodes, agent_type = 'qlearning', alpha = 0.1, n=0): 
    all_rep_results = []
    
    for _ in range (n_reps): 
        env = ShortcutEnvironment()
        
        #depends which agent
        if agent_type == 'qlearning': 
            agent = QLearningAgent(n_actions = env.action_size(), n_states = env.state_size(), alpha=alpha)
        elif agent_type == 'sarsa': 
            agent = SARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), alpha=alpha)
        elif agent_type == 'expected_sarsa': 
            agent = ExpectedSARSAAgent(n_actions = env.action_size(), n_states = env.state_size())
        elif agent_type == 'n_step_sarsa': 
            agent = nStepSARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), n=n)
        
        single_rep_result = agent.train(env, n_episodes)
        all_rep_results.append(single_rep_result)
    
    return all_rep_results

def rendering_version(agent_type = 'qlearning', environment = ShortcutEnvironment): 
    #Assumes n_reps = 1, n_episodes = 10000
    env = environment()

    #depends which agent
    if agent_type == 'qlearning': 
        agent = QLearningAgent(n_actions = env.action_size(), n_states = env.state_size())
    elif agent_type == 'sarsa': 
        agent = SARSAAgent(n_actions = env.action_size(), n_states = env.state_size())
    elif agent_type == 'expected_sarsa': 
        agent = ExpectedSARSAAgent(n_actions = env.action_size(), n_states = env.state_size())
    elif agent_type == 'n_step_sarsa': 
        agent = nStepSARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), n=3)
    
    single_rep_result = agent.train(env, 10000)
    env.render_greedy(agent.Q)



#Note here we modified run_repetitions() to take alpha as an input
def smooth_curve(y, window=10):
    pad = window // 2
    y_padded = np.pad(y, (pad, pad), mode='edge')
    smoothed = np.convolve(y_padded, np.ones(window)/window, mode='valid')
    return smoothed

def compare_alpha_values(agent_type): 
    alpha_values = [0.01, 0.1, 0.5, 0.9]
    plt.figure(figsize=(15, 7))
    for alpha in alpha_values:
        print(f"Running experiments for alpha={alpha}...")
        results = run_repetitions(100, 1000, agent_type, alpha)
        mean_rewards = np.mean(results, axis=0)
        smoothed_rewards = smooth_curve(mean_rewards, window=20) 
        plt.plot(smoothed_rewards, label=f'alpha={alpha}')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")

    if agent_type == 'qlearning': 
        plt.title(f"Q-Learning Average Reward for Different Alpha Values (100 repetitions)")
    elif agent_type == 'sarsa': 
        plt.title(f"SARSA Average Reward for Different Alpha Values (100 repetitions)")
    elif agent_type == 'expected_sarsa': 
        plt.title(f"Expected SARSA Average Reward for Different Alpha Values (100 repetitions)")
    elif agent_type == 'n_step_sarsa': 
        plt.title(f"N-step SARSA Average Reward for Different Alpha Values (100 repetitions)")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_n_values(): 
    n_values = [1, 2, 5, 10, 25]
    plt.figure(figsize=(15, 7))
    for n in n_values:
        print(f"Running experiments for n={n}...")
        results = run_repetitions(100, 1000, 'n_step_sarsa', n)
        mean_rewards = np.mean(results, axis=0)
        smoothed_rewards = smooth_curve(mean_rewards, window=20) 
        plt.plot(smoothed_rewards, label=f'N={n}')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")

    plt.title(f"N-step SARSA Average Reward for Different N Values (100 repetitions)")
    plt.legend()
    plt.grid(True)
    plt.show()



## Part 1: Q-learning

# rendering_version()

# results = run_repetitions(n_reps = 100, n_episodes = 1000)
# mean_rewards = np.mean(results, axis=0)
# plt.figure(figsize=(10,6))
# plt.plot(mean_rewards, label="Average Cumulative Reward")
# plt.xlabel("Episode")
# plt.ylabel("Cumulative Reward")
# plt.title(f"Graph of average cumulative rewards of Q-Learning over 100 repetitions, each with 1000 episodes")
# plt.grid(True)
# plt.legend()
# plt.show()


## Part 2

# rendering_version('sarsa')

# compare_alpha_values('sarsa')



## Part 3
# rendering_version('qlearning', WindyShortcutEnvironment)
# rendering_version('sarsa', WindyShortcutEnvironment)


## Part 4
# rendering_version('expected_sarsa')
# compare_alpha_values('expected_sarsa')

## Part 5
# rendering_version('n_step_sarsa')
compare_n_values()