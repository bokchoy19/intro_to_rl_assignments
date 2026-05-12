#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sas = np.zeros((n_states, n_actions, n_states))
        self.Rsum_sas = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, s, epsilon):
        if np.random.rand() < epsilon: #explore
            a = np.random.randint(self.n_actions)
        else: #exploit 
            a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):

        #update Q-table values using real experience
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q_sa[s_next])
        td_error = target - self.Q_sa[s,a]
        self.Q_sa[s,a] += self.learning_rate * td_error

        #update model using real experience
        self.n_sas[s,a,s_next] += 1
        self.Rsum_sas[s,a,s_next] += r

        #update Q-table values using model (planning)
        for _ in range(n_planning_updates):
            #sample previously seen state
            visited_states = np.where(np.sum(self.n_sas, axis=(1,2)) > 0)[0]
            s_plan = np.random.choice(visited_states)

            #sample previously seen action
            visited_actions = np.where(np.sum(self.n_sas[s_plan], axis=1) > 0)[0]
            a_plan = np.random.choice(visited_actions)

            #sample next state from model
            counts = self.n_sas[s_plan,a_plan]
            probs = counts / np.sum(counts)
            s_next_plan = np.random.choice(self.n_states, p=probs)

            #estimated reward
            r_plan = (self.Rsum_sas[s_plan,a_plan,s_next_plan] / self.n_sas[s_plan,a_plan,s_next_plan])

            #planning Q update
            target = (r_plan + self.gamma * np.max(self.Q_sa[s_next_plan]))
            td_error = target - self.Q_sa[s_plan,a_plan]
            self.Q_sa[s_plan,a_plan] += (self.learning_rate * td_error)

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sas = np.zeros((n_states, n_actions, n_states))
        self.Rsum_sas = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, s, epsilon):
        if np.random.rand() < epsilon: #explore
            a = np.random.randint(self.n_actions)
        else: #exploit 
            a = np.argmax(self.Q_sa[s])
        return a
    def update(self, s, a, r, done, s_next, n_planning_updates):
        #update Q-table values using real experience 
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q_sa[s_next])

        td_error = target - self.Q_sa[s,a]
        self.Q_sa[s,a] += self.learning_rate * td_error

        #update model using real experience
        self.n_sas[s,a,s_next] += 1
        self.Rsum_sas[s,a,s_next] += r

        #add to priority queue
        priority = abs(td_error)
        if priority > self.priority_cutoff:
            self.queue.put((-priority, (s,a)))

        #planning Q update
        for _ in range(n_planning_updates):

            if self.queue.empty():
                break

            _, (s_plan, a_plan) = self.queue.get()

            counts = self.n_sas[s_plan, a_plan]
            probs = counts / np.sum(counts)
            s_next_plan = np.random.choice(self.n_states, p=probs)

            r_plan = self.Rsum_sas[s_plan,a_plan,s_next_plan] / self.n_sas[s_plan,a_plan,s_next_plan]
            target_plan = r_plan + self.gamma * np.max(self.Q_sa[s_next_plan])
            td_error_plan = target_plan - self.Q_sa[s_plan,a_plan]
            self.Q_sa[s_plan,a_plan] += self.learning_rate * td_error_plan

            #working backwards
            #s_bar and a_bar are state-action pairs that can lead to current state
            # r_bar is E[reward] of (s_bar, a_bar, s) 
            for s_bar in range(self.n_states):
                for a_bar in range(self.n_actions):
                    if self.n_sas[s_bar, a_bar, s_plan] > 0:
                        r_bar = self.Rsum_sas[s_bar,a_bar,s_plan] / self.n_sas[s_bar,a_bar,s_plan]
                        p_bar = abs(r_bar + self.gamma * np.max(self.Q_sa[s_plan]) - self.Q_sa[s_bar,a_bar])

                        if p_bar > self.priority_cutoff:
                            self.queue.put((-p_bar, (s_bar, a_bar)))


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
