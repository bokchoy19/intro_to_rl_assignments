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

'''
note: the testing has been split into functions to make it much easier to run individually (which is advised as they take a long time to run)
(as requested, all the graphing functions have been called at the bottom)
the tables do not auto save as images, since they were graphed in terminal and not actually in python (e.g. plt) 
'''


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
    plt.savefig(f"compare_alpha_{agent_type}.png")
    plt.show()
    plt.close()


def compare_n_values(): 
    n_values = [1, 2, 5, 10]
    plt.figure(figsize=(15, 7))
    for n in n_values:
        print(f"Running experiments for n={n}...")
        results = run_repetitions(100, 1000, 'n_step_sarsa', n=n)
        mean_rewards = np.mean(results, axis=0)
        smoothed_rewards = smooth_curve(mean_rewards, window=20) 
        plt.plot(smoothed_rewards, label=f'N={n}')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title(f"N-step SARSA Average Reward for Different N Values (100 repetitions)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"compare_n_n_step_sarsa.png")
    plt.show()
    plt.close()


def compare_agents(): 
    agents = ['qlearning', 'sarsa', 'expected_sarsa', 'n_step_sarsa']
    plt.figure(figsize=(15, 7))
    for agent in agents:
        results = run_repetitions(100, 1000, agent, alpha = 0.5, n=10)
        mean_rewards = np.mean(results, axis=0)
        smoothed_rewards = smooth_curve(mean_rewards, window=20) 
        plt.plot(smoothed_rewards, label=f'Agent={agent}')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Average Reward for different agents (100 repetitions)")
    plt.legend()
    plt.grid(True)
    plt.savefig("compare_agents.png")
    plt.show()
    plt.close()


def compare_agents_table(): 
    print(f"{'Agent':<20} {'Mean Last 100 Reward':>25}")
    print("-" * 45)

    for agent in ['qlearning', 'sarsa', 'expected_sarsa', 'n_step_sarsa']: 
        results = run_repetitions(n_reps = 100, n_episodes = 1000, agent_type = agent, alpha = 0.5, n=10)
        mean_rewards = np.mean(results, axis=0)
        final_score = np.mean(mean_rewards[-100:])
        print(f"{agent:<20} {final_score:>25.2f}")







## Part 1: Q-learning

rendering_version()

# for alpha in [0.01, 0.1, 0.5, 0.9]: 
#     results = run_repetitions(n_reps = 100, n_episodes = 1000, agent_type = 'qlearning', alpha = alpha)
#     mean_rewards = np.mean(results, axis=0)
#     print(f'Mean rewards of last 100 episodes for alpha = {alpha} is {np.mean(mean_rewards[-100:])}')

compare_alpha_values('qlearning')

## Part 2

rendering_version('sarsa')

compare_alpha_values('sarsa')



## Part 3
rendering_version('qlearning', WindyShortcutEnvironment)
rendering_version('sarsa', WindyShortcutEnvironment)

## Part 4
rendering_version('expected_sarsa')
compare_alpha_values('expected_sarsa')

# for alpha in [0.01, 0.1, 0.5, 0.9]: 
#     results = run_repetitions(n_reps = 100, n_episodes = 1000, agent_type = 'expected_sarsa', alpha = alpha)
#     mean_rewards = np.mean(results, axis=0)
#     print(f'Mean rewards of last 100 episodes for alpha = {alpha} is {np.mean(mean_rewards[-100:])}')


## Part 5
rendering_version('n_step_sarsa')
compare_n_values()

# for n in [1, 2, 5, 10, 25]: 
#     results = run_repetitions(n_reps = 100, n_episodes = 1000, agent_type = 'n_step_sarsa', n = n)
#     mean_rewards = np.mean(results, axis=0)
#     print(f'Mean rewards of last 100 episodes for n = {n} is {np.mean(mean_rewards[-100:])}')


## Part 6
compare_agents()
compare_agents_table()