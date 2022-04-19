import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class eps_bandit:
    '''

    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0.
        Set to "sequence" for the means to be ordered from
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''

    def __init__(self, k, eps, iters, mu='random'):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)

        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k - 1, k)

    def pull(self):
        # Generate random number
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)

        reward = np.random.normal(self.mu[a], 1)

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
                reward - self.k_reward[a]) / self.k_n[a]

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)


k = 10
iters = 1000

eps_0_rewards = np.zeros(iters)
eps_1_rewards = np.zeros(iters)
eps_2_rewards = np.zeros(iters)
eps_3_rewards = np.zeros(iters)
eps_4_rewards = np.zeros(iters)

episodes = 1
# Run experiments
for i in range(episodes):
    # Initialize bandits
    eps_0 = eps_bandit(k, 0, iters)
    eps_1 = eps_bandit(k, 0.15, iters, eps_0.mu.copy())
    eps_2 = eps_bandit(k, 0.25, iters, eps_0.mu.copy())
    eps_3 = eps_bandit(k, 0.35, iters, eps_0.mu.copy())
    eps_4 = eps_bandit(k, 0.45, iters, eps_0.mu.copy())

    # Run experiments
    eps_0.run()
    eps_1.run()
    eps_2.run()
    eps_3.run()
    eps_4.run()

    # Update long-term averages
    eps_1_rewards = eps_1_rewards + (
            eps_1.reward - eps_1_rewards) / (i + 1)
    eps_2_rewards = eps_2_rewards + (
            eps_2.reward - eps_2_rewards) / (i + 1)
    eps_3_rewards = eps_3_rewards + (
            eps_3.reward - eps_3_rewards) / (i + 1)
    eps_4_rewards = eps_4_rewards + (
            eps_4.reward - eps_4_rewards) / (i + 1)

plt.figure(figsize=(12, 8))
plt.plot(eps_1_rewards, label="$\epsilon=0.15$")
plt.plot(eps_2_rewards, label="$\epsilon=0.25$")
plt.plot(eps_3_rewards, label="$\epsilon=0.35$")
plt.plot(eps_4_rewards, label="$\epsilon=0.45$")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Bez Habituacji Average $\epsilon-greedy$ Rewards after " + str(episodes)
          + " Episodes")
plt.show()