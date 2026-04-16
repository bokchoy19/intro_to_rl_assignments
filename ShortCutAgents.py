import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # DONE: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # DONE: Implement greedy policy
        if np.random.rand() > self.epsilon: 
            action = np.argmax(self.Q[state])
        else: 
            choices = [a for a in range(self.n_actions) if a != np.argmax(self.Q[state])]
            action = np.random.choice(choices)
        return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # DONE: Implement Q-learning update
        best_next = np.max(self.Q[next_state])

        if not done: 
            self.Q[state, action] += self.alpha * (reward + self.gamma * best_next - self.Q[state, action])
        else: 
            #because agent terminates after reaching end
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
    
    def train(self, env, n_episodes):
        # DONE: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for i in range(n_episodes): 
            cumulative_reward = 0
            env.reset()
            state = env.state()
            done = env.done()

            while not done: 
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                self.update(state, action, reward, next_state, done)

                state = next_state
                cumulative_reward += reward
            
            episode_returns.append(cumulative_reward)

        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # DONE: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))

        
    def select_action(self, state):
        # DONE: Implement policy
        if np.random.rand() > self.epsilon: 
            action = np.argmax(self.Q[state])
        else: 
            choices = [a for a in range(self.n_actions) if a != np.argmax(self.Q[state])]
            action = np.random.choice(choices)
        return action
        
    def update(self, state, action, reward, next_state, next_action, done): # Augment arguments if necessary
        # DONE: Implement SARSA update
        if not done: 
            self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
        else: 
            #because agent terminates after reaching end
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])

    def train(self, env, n_episodes):
        # DONE: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for i in range(n_episodes): 
            cumulative_reward = 0
            env.reset()
            state = env.state()
            done = env.done()

            while not done: 
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                cumulative_reward += reward
            
            episode_returns.append(cumulative_reward)

        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # DONE: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # DONE: Implement policy
        if np.random.rand() > self.epsilon: 
            action = np.argmax(self.Q[state])
        else: 
            choices = [a for a in range(self.n_actions) if a != np.argmax(self.Q[state])]
            action = np.random.choice(choices)
        return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        expected_Q = 0
        for a in range(self.n_actions): 
            if a == np.argmax(self.Q[next_state]): 
                expected_Q += (1 - self.epsilon) * self.Q[next_state, a]
            else: 
                expected_Q += self.epsilon/(self.n_actions - 1) * self.Q[next_state, a]

        if not done: 
            self.Q[state, action] += self.alpha * (reward + self.gamma * expected_Q - self.Q[state, action])
        else: 
            #because agent terminates after reaching end
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for i in range(n_episodes): 
            cumulative_reward = 0
            env.reset()
            state = env.state()
            done = env.done()

            while not done: 
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, done)

                state = next_state
                action = next_action
                cumulative_reward += reward
            
            episode_returns.append(cumulative_reward)

        return episode_returns


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # DONE: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # DONE: Implement policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
    def update(self, states, actions, rewards, t, T, done): # Augment arguments if necessary
        # DONE: Implement n-step SARSA update
        tau = t - self.n + 1

        #only start updating after the nth action, aka t = n-1
        if tau < 0:
            return

        G = 0
        upper = min(tau + self.n, T)

        #add rewards
        for i in range(tau + 1, upper + 1):
            G += (self.gamma ** (i - tau - 1)) * rewards[i]

        #bootstrap if not terminal
        if tau + self.n < T:
            G += (self.gamma ** self.n) * self.Q[
                states[tau + self.n],
                actions[tau + self.n]
            ]

        s_tau = states[tau]
        a_tau = actions[tau]
        self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])

    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for i in range(n_episodes):
            states = []
            actions = []
            rewards = [0]

            env.reset()
            state = env.state()
            action = self.select_action(state)

            states.append(state)
            actions.append(action)

            T = float('inf')
            t = 0
            total_reward = 0

            while True:
                #next step
                if t < T:
                    reward = env.step(actions[t])
                    next_state = env.state()
                    done = env.done()

                    rewards.append(reward)
                    states.append(next_state)
                    total_reward += reward

                    if done:
                        T = t + 1
                    else:
                        next_action = self.select_action(next_state)
                        actions.append(next_action)

                tau = t - self.n + 1
                if tau >= 0:
                    self.update(states, actions, rewards, t, T, done)
                if tau == T - 1:
                    break
                t += 1

            episode_returns.append(total_reward)
        return episode_returns