#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
import matplotlib.pyplot as plt
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updatess = [1,3,5] 
    
    # IMPLEMENT YOUR EXPERIMENT HERE
    
if __name__ == '__main__':
    experiment()



def run_repetitions(AgentClass, n_planning_updates, epsilon, learning_rate, wind_proportion, n_timesteps=10001, eval_interval=250, n_repetitions=20, gamma=1.0):
    all_curves = []

    for _ in range(n_repetitions):
        env = WindyGridworld(wind_proportion=wind_proportion)
        agent = AgentClass(env.n_states, env.n_actions, learning_rate, gamma)
        s = env.reset()
        curve = []
        #curve is going to be a list of values of episode returns at t=250,500,750... etc

        for t in range(n_timesteps):
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s, a, r, done, s_next, n_planning_updates)
            s = env.reset() if done else s_next

            #evaluate every 250 steps (or whatever frequency u want)
            if t % eval_interval == 0:
                curve.append(agent.evaluate(env))
        all_curves.append(curve)

    #return average over all curves
    return np.mean(np.array(all_curves), axis=0)


def run_experiment(agent_class):
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_list = [1,3,5]

    if agent_class == DynaAgent:
        plot_stochastic = LearningCurvePlot(title="Dyna agent with stochastic wind (0.9)")
        plot_deterministic = LearningCurvePlot(title="Dyna agent with deterministic wind (1.0)")
    elif agent_class == PrioritizedSweepingAgent: 
        plot_stochastic = LearningCurvePlot(title="Prioritised sweeping agent with stochastic wind (0.9)")
        plot_deterministic = LearningCurvePlot(title="Prioritised sweeping agent with deterministic wind (1.0)")
    for plot in [plot_stochastic, plot_deterministic]:
        plot.ax.set_xlabel("Timestep")
        plot.ax.set_ylabel("Mean episode return")
        plot.ax.grid(True)


    #timestep values of points in curve()
    x = np.arange(0, n_timesteps, eval_interval)

    #stochastic environment
    for K in n_planning_list:
        curve = run_repetitions(agent_class, K, epsilon, learning_rate, 0.9, n_timesteps, eval_interval, n_repetitions)
        plot_stochastic.add_curve(x[:len(curve)], smooth(curve, window=5), label=f"{K} planning updates")
    #add Q-learning baseline to compare against, note this is just Dyna with K=0
    q_curve = run_repetitions(agent_class, 0, epsilon, learning_rate, 0.9, n_timesteps, eval_interval, n_repetitions)
    plot_stochastic.add_curve(x[:len(q_curve)], smooth(q_curve, window=5), label="Q-learning")
    plot_stochastic.ax.legend()
    #save plot
    if agent_class == DynaAgent:
        plot_stochastic.save("dyna_stochastic.png")
    elif agent_class == PrioritizedSweepingAgent: 
        plot_stochastic.save("ps_stochastic.png")

    #deterministic environment
    for K in n_planning_list:
        curve = run_repetitions(agent_class, K, epsilon, learning_rate, 1.0, n_timesteps, eval_interval, n_repetitions)
        plot_deterministic.add_curve(x[:len(curve)], smooth(curve, window=5), label=f"{K} planning updates")
    #add Q-learning baseline to compare against, note this is just Dyna with K=0
    q_curve = run_repetitions(agent_class, 0, epsilon, learning_rate, 1.0, n_timesteps, eval_interval, n_repetitions)
    plot_deterministic.add_curve(x[:len(q_curve)], smooth(q_curve, window=5), label="Q-learning")
    plot_deterministic.ax.legend()
    #save plot
    if agent_class == DynaAgent:
        plot_deterministic.save("dyna_deterministic.png")
    elif agent_class == PrioritizedSweepingAgent: 
        plot_stochastic.save("ps_deterministic.png")


    plt.show()


#run functions
run_experiment(DynaAgent)
#run_experiment(PrioritizedSweepingAgent)