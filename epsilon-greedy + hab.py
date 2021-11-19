import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class eps_bandit:

    def __init__(self, k, eps, iters, mu='random'):
        # Liczba ramion
        self.k = k
        # Szukane prawdopodobieństwo
        self.eps = eps
        # Liczba iteracji
        self.iters = iters
        # Liczba kroków
        self.n = 0
        # Liczba kroków dla każdego ramienia
        self.k_n = np.zeros(k)
        # Nagroda całościowa
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Średnia nagroda za każde ramię
        self.k_reward = np.zeros(k)

        ads_selected = []
        total_reward = 500
        self.ads_hab = []
        self.tau = tau
        self.alpha = alpha
        S = 200
        steps = 0
        st = 0


        for a in range(0, self.k):
            a = {'ad': a, 'h': 1, 'cc': 0, 'ncc': 0, 'ohu': 1, 'ohd': 1, 'p': 0, 'att': 0}
            self.ads_hab.append(a)


        if type(mu) == list or type(mu).__module__ == np.__name__:
            # Średnie zdefiniowane przez usera
            self.mu = np.array(mu)
        elif mu == 'random':
            # Średnie z rozkładu prawdopodobieństwa
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Zwiększ średnią dla każdego ramienia o jeden
            self.mu = np.linspace(0, k - 1, k)

    def pull(self):
        # Wygeneruj losową liczbę
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            # Losowo wybierz akcję
            a = np.random.choice(self.k)
        else:
            # "Chciwe działanie"
            a = np.argmax(self.k_reward)

        #print(a)
        # print(self.ads_hab[a])

        # print("reklama ", a, "przed :", self.ads_hab[a])
        self.ads_hab[a]['cc'] += 1
        self.ads_hab[a]['att'] += 1
        hab = 1 - (1 / alpha) * (1 - math.exp(-1 * (alpha * self.ads_hab[a]['cc']) / tau))
        self.ads_hab[a]['ncc'] = 0
        oldhab = self.ads_hab[a]['ohu']
        self.ads_hab[a]['h'] = round(hab, 2)
        self.ads_hab[a]['h'] = round((oldhab * hab), 2)
        self.ads_hab[a]['ohd'] = self.ads_hab[a]['h']
        # print("reklama ", a, "po :", self.ads_hab[a])

        for n in range(self.k):
            # print(n)
            if n != a and self.ads_hab[n]['h'] < 1:
                #print("ODBUDOWA!",n)
                #print("reklama ", n, "przed :", self.ads_hab[n])
                self.ads_hab[n]['ncc'] += 1
                self.ads_hab[n]['p'] += 1
                hab = 1 - (1 - self.ads_hab[n]['ohd']) * math.exp(-1 * self.alpha * (self.ads_hab[n]['ncc'] / self.tau))
                self.ads_hab[n]['cc'] = 0
                self.ads_hab[n]['h'] = round(hab, 2)
                self.ads_hab[n]['ohu'] = self.ads_hab[n]['h']
                #print("reklama ", n, "po :", self.ads_hab[n])


        # print(self.k)
        # print(self.eps)
        # print(self.iters)
        print(self.n)
        print(self.k_n)
        print(self.mean_reward)
        # print(self.reward)

        reward = np.random.normal(self.mu[a], 1)
        # print(reward)
        # Aktualizacja liczebności
        self.n += 1
        self.k_n[a] += 1

        # Aktualizacja all nagród
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n

        # Update wyników dla a_k
        self.k_reward[a] = self.k_reward[a] + (
                reward - self.k_reward[a]) / self.k_n[a]

        print('')
        # print(self.k_reward[a])
        print(self.k_reward)
        for x in range(self.k):
            print(self.ads_hab[x]['ad'], self.ads_hab[x]['h'])

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self):
        # Resetuje wyniki z zachowaniem ustawień

        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)


k = 10
iters = 1000
alpha = 1.05
tau = 3

eps_0_rewards = np.zeros(iters)
eps_1_rewards = np.zeros(iters)
eps_2_rewards = np.zeros(iters)

episodes = 10
# Odpalamy eksperyment
for i in range(episodes):
    # print("episodes :", i)
    # Bandyci -start
    eps_0 = eps_bandit(k, 0.25, iters)
    eps_1 = eps_bandit(k, 0.02, iters, eps_0.mu.copy())
    eps_2 = eps_bandit(k, 0.35, iters, eps_0.mu.copy())
    # print(eps_0)

    # Zliczanie
    eps_0.run()
    eps_1.run()
    eps_2.run()

    # Update średnich all
    eps_0_rewards = eps_0_rewards + (
            eps_0.reward - eps_0_rewards) / (i + 1)
    eps_1_rewards = eps_1_rewards + (
            eps_1.reward - eps_1_rewards) / (i + 1)
    eps_2_rewards = eps_2_rewards + (
            eps_2.reward - eps_2_rewards) / (i + 1)


'-------------------------------------------------------------------------------'
"Wykresy pomocnicze - średnia nagroda przed habituacją"


plt.figure(figsize=(12, 8))
plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
plt.plot(eps_1_rewards, label="$\epsilon=1$ ")
plt.plot(eps_2_rewards, label="$\epsilon=2$")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iteracje")
plt.ylabel("Średnia nagroda")
plt.title("Średni $\epsilon-greedy$ Nagroda po " + str(episodes)
          + " epizodach")
plt.show()



'-------------------------------------------------------------------------------'
"Wykresy pomocnicze - średnia nagroda po habituacją"