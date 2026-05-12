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
import time
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
        plot_deterministic.save("ps_deterministic.png")


    plt.show()


def compare_best_models():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    epsilon = 0.1
    learning_rate = 0.2

    best_dyna_K = 5
    best_ps_K = 5
    x = np.arange(0, n_timesteps, eval_interval)

    #stochastic comparison
    stochastic_plot = LearningCurvePlot(title="Best models comparison (stochastic wind 0.9)")
    stochastic_plot.ax.set_xlabel("Timestep")
    stochastic_plot.ax.set_ylabel("Mean episode return")
    stochastic_plot.ax.grid(True)

    dyna_curve = run_repetitions(DynaAgent, best_dyna_K, epsilon, learning_rate, 0.9, n_timesteps, eval_interval, n_repetitions)
    ps_curve = run_repetitions(PrioritizedSweepingAgent, best_ps_K, epsilon, learning_rate, 0.9, n_timesteps, eval_interval, n_repetitions)
    q_curve = run_repetitions(DynaAgent, 0, epsilon, learning_rate, 0.9, n_timesteps, eval_interval, n_repetitions)

    stochastic_plot.add_curve(x[:len(dyna_curve)], smooth(dyna_curve, window=5), label=f"Dyna (K={best_dyna_K})")
    stochastic_plot.add_curve(x[:len(ps_curve)], smooth(ps_curve, window=5), label=f"Prioritized Sweeping (K={best_ps_K})")
    stochastic_plot.add_curve(x[:len(q_curve)], smooth(q_curve, window=5), label="Q-learning")

    stochastic_plot.save("comparison_stochastic.png")

    #deterministic comparison
    deterministic_plot = LearningCurvePlot(title="Best models comparison (deterministic wind 1.0)")
    deterministic_plot.ax.set_xlabel("Timestep")
    deterministic_plot.ax.set_ylabel("Mean episode return")
    deterministic_plot.ax.grid(True)

    dyna_curve = run_repetitions(DynaAgent, best_dyna_K, epsilon, learning_rate, 1.0, n_timesteps, eval_interval, n_repetitions)
    ps_curve = run_repetitions(PrioritizedSweepingAgent, best_ps_K, epsilon, learning_rate, 1.0, n_timesteps, eval_interval, n_repetitions)
    q_curve = run_repetitions(DynaAgent, 0, epsilon, learning_rate, 1.0, n_timesteps, eval_interval, n_repetitions)

    deterministic_plot.add_curve(x[:len(dyna_curve)], smooth(dyna_curve, window=5), label=f"Dyna (K={best_dyna_K})")
    deterministic_plot.add_curve(x[:len(ps_curve)], smooth(ps_curve, window=5), label=f"Prioritized Sweeping (K={best_ps_K})")
    deterministic_plot.add_curve(x[:len(q_curve)], smooth(q_curve, window=5), label="Q-learning")

    deterministic_plot.save("comparison_deterministic.png")

    plt.show()

def measure_runtime(agent_class, n_planning_updates):
    n_timesteps = 10001
    epsilon = 0.1
    learning_rate = 0.2
    gamma = 1.0

    #create environment and agent
    env = WindyGridworld(wind_proportion=0.9)
    agent = agent_class(env.n_states, env.n_actions, learning_rate, gamma)
    s = env.reset()

    start_time = time.time()

    #run one repetition
    for _ in range(n_timesteps):
        a = agent.select_action(s, epsilon)
        s_next, r, done = env.step(a)

        #learning + planning updates
        agent.update(s, a, r, done, s_next, n_planning_updates)

        #reset episode if goal reached
        s = env.reset() if done else s_next

    return time.time() - start_time


def runtime_table():
    n_repetitions = 10
    K = 5

    qlearning_times = []
    dyna_times = []
    ps_times = []

    #measure runtime repeatedly and average
    for _ in range(n_repetitions):
        qlearning_times.append(measure_runtime(DynaAgent, 0))
        dyna_times.append(measure_runtime(DynaAgent, K))
        ps_times.append(measure_runtime(PrioritizedSweepingAgent, K))

    #compute averages
    qlearning_mean = np.mean(qlearning_times)
    dyna_mean = np.mean(dyna_times)
    ps_mean = np.mean(ps_times)

    #print formatted table
    print("\nAverage Runtime over", n_repetitions, "runs\n")
    print("+-------------------------------+------------------+")
    print("| Algorithm                     | Runtime (s)      |")
    print("+-------------------------------+------------------+")
    print(f"| Q-learning                    | {qlearning_mean:<16.4f} |")
    print(f"| Dyna (K={K})                  | {dyna_mean:<16.4f} |")
    print(f"| Prioritized Sweeping (K={K})  | {ps_mean:<16.4f} |")
    print("+-------------------------------+------------------+")

#run functions
run_experiment(DynaAgent)
run_experiment(PrioritizedSweepingAgent)
compare_best_models()
runtime_table()