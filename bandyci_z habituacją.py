import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random


class eps_bandit:

    def __init__(self, k, eps, iters, tau, alpha, mu='random'):
        # Liczba ramion
        self.k = k
        # print(self.k)
        # Szukane prawdopodobieństwo
        self.eps = eps
        print(self.eps)
        # Liczba iteracji
        self.iters = iters
        print(self.iters)
        # Liczba kroków
        self.n = 0
        # print(self.n)
        # Liczba kroków dla każdego ramienia
        self.k_n = np.zeros(k)
        # print(self.k_n)
        # Nagroda całościowa
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # print(self.reward)
        # Średnia nagroda za każde ramię
        self.k_reward = np.zeros(k)
        # print(self.k_reward)

        ads_selected = []
        reward = 30
        self.ads_hab = []
        self.habibi = habibi
        self.habs = habs
        self.tau = tau
        self.alpha = alpha
        self.a = 0
        S = 200
        steps = 0
        st = 0
        # responsywność ustawiona na poziom od 0-1,czyli random
        h = round(random.uniform(0, 1),2)
        for z in range(0, self.k):
            a = {'ad': z, 'h': []}

            b = {'ad': z, 'h': 1, 'cc': 0, 'ncc': 0,
                 'ohu': 1, 'ohd': 1, 'p': 0, 'att': 0}
            self.ads_hab.append(b)
            self.habibi.append(a)

        print(self.ads_hab)
        print('')
        print('')
# Opis zmiennych do wzoru habituacji
# # ad - reklama
# # h - poziom responsywności
# # cc - licznik kontaktów pod rząd
# # ncc - licznk braku kontaktów pod rząd
# # ohu - wartosc habituacji przed rozpoczeciem spadku
# # ohd - to samo, ale przed rozpoczeciem odbudowy
# # p - liczba pauz - braku kontaktow ogolnie w symulacji
# # att - ile razy kliknięto
# # episodes - pojedyńcza akcja w przypadku

        if type(mu) == list or type(mu).__module__ == np.__name__:
            # Średnie zdefiniowane przez usera
            self.mu = np.array(mu)
            print('array :', self.mu)
        elif mu == 'random':
            # Średnie z rozkładu prawdopodobieństwa
            self.mu = np.random.normal(0, 1, k)
            print('random :', self.mu)
        elif mu == 'sequence':
            # Zwiększ średnią dla każdego ramienia o jeden
            self.mu = np.linspace(0, k - 1, k)

    def pull(self, a):
        # Wygeneruj losowe ramię
        # print(c)
        # a = np.random.choice(self.k)

        print('reklama :', self.a)
        print('responsywność :', (self.ads_hab[int(self.a)]['h']))
        self.n += 1
        print('n :', self.n)

        p = np.random.rand()
        hab = self.ads_hab[int(self.a)]["h"]
        habp = hab * p
        print('p :', p)
        print('habp :', habp)
        if self.eps == 0 and self.n == 0:
            self.a = np.random.choice(self.k)
            print(' :', self.a)
        elif habp < self.eps:
            # Losowo wybierz akcję
            self.a = np.random.choice(k)
            print('eksploracja:', self.a)
        else:
            # "Chciwe działanie"
            # self.a = np.argmax(self.k_reward)
            print('chciwiec :', self.a)

        def hab_down(x):
            self.ads_hab[x]["cc"] += 1
            print('spadek responsywnosci :')
            print('cc :', self.ads_hab[x]["cc"])

            print('hab przed :', self.ads_hab[x]["h"])
            hab = 1 - (1 / alpha) * (
                    1 - math.exp(-1 * (alpha * self.ads_hab[x]["cc"]) / tau))

            oldhab = self.ads_hab[x]["ohd"]

            self.ads_hab[x]["h"] = round((oldhab * hab), 2)

            print('hab po :', self.ads_hab[x]["h"])

            self.ads_hab[x]["ohd"] = self.ads_hab[x]["h"]

            self.ads_hab[x]["ncc"] = 0

        def hab_up(y):
            print('')
            print('odbudowa responsywnosci')
            self.ads_hab[y]["ncc"] += 1
            print('ncc :', self.ads_hab[y]["ncc"])

            hab = 1 - (1 - self.ads_hab[y]["h"]) * math.exp(
                -1 * alpha * (
                        (self.ads_hab[y]["ncc"]) / tau))

            print('hab :', hab)

            self.ads_hab[y]["h"] = round(hab, 2)

            self.ads_hab[y]["ohd"] = self.ads_hab[y]["h"]

            self.ads_hab[y]["cc"] = 0

        # print(a)
        # print(self.ads_hab[a])

        reward = np.random.normal(self.mu[self.a], 1)
        print(self.mu[self.a])
        # print(a)
        print('reward :', reward)

        # Aktualizacja liczebności
        # self.n += 1
        # print(self.n)
        self.k_n[int(self.a)] += 1
        print(self.k_n[self.a])
        print(self.k_n)

        # Aktualizacja all nagród
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n

        # print('mean reward :', self.mean_reward)

        # Update wyników dla a_k
        self.k_reward[self.a] = self.k_reward[self.a] + (
                reward - self.k_reward[self.a]) / self.k_n[self.a]

        avg_ha = []

        for i in range(0, k):

            if i == self.a:
                print('ten pyknięty :', i)
                hab_down(i)
            if i != self.a and self.ads_hab[i]["h"] < 1:
                print('pozostale :', i)
                hab_up(i)
                print('poszlo?')

            avg_ha.append(self.ads_hab[i]["h"])
            habibi[i]["h"].append(self.ads_hab[i]["h"])

        print(avg_ha)
        print(sum(avg_ha))

        avh = round(sum(avg_ha)/self.k, 4)

        print('avh :', avh)

        self.habs.append(avh)

        if reward <= 0:
            self.a = np.argmax(self.k_reward)
            print('chciwiec wybiera:', self.a)

        # print('')
        # print('self_k_a', self.k_reward[a])
        print('self_k :', self.k_reward)
        print(self.ads_hab)

        # for x in range(self.k):
        #     print(self.ads_hab[x]['ad'], self.ads_hab[x]['h'])

    def run(self):
        for i in range(self.iters):
            print('')
            print('iteracja :', i)
            print(self.habs)
            self.pull(i)
            self.reward[i] = self.mean_reward

    def reset(self):
        # Resetuje wyniki z zachowaniem ustawień

        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k)
        self.habibi.clear()
        self.habs.clear()


k = 5 # liczba ramion/ liczba reklam
iters = 100 # lość interacji konkretnego pociągnięcia ramienia
# alpha = 1.05
# tau = 3

n_alpha = [1.05] # max do ok 1.25
n_tau = [1,5,25] #  przyjać wartośći 1 - 100
episodes = [1] # pojedyńcza akcja w przypadku
epsy = [0.25]



# print(eps_0_rewards)

for epes in epsy:
    for episode in episodes:
        for alpha in n_alpha:
            for tau in n_tau:
                # Odpalamy eksperyment
                for i in range(episode):

                    eps_0_rewards = np.zeros(iters)
                    habs = []
                    habibi = []
                    print("episodes :", i)
                    # Bandyci -start
                    eps_0 = eps_bandit(k, epes, iters, tau, alpha)

                    eps_0.run()

                    # Update średnich all
                    eps_0_rewards = eps_0_rewards + (
                        eps_0.reward - eps_0_rewards) / (i + 1)
                    print(habs)
                    print(habibi)
                    print('rewards :', eps_0_rewards)

                    '-------------------------------------------------------------------------------'
                    "Wykresy pomocnicze - średnia nagroda po habituacją"

                    plt.figure(figsize=(12, 8))
                    plt.plot(eps_0_rewards, label="epsilon " + str(epes))
                    plt.plot(habs, label="avg responsiveness")

                    for i in range(0, k):
                        plt.plot(habibi[i]["h"], label='ad ' + str(i) + ' responsiveness')

                    plt.legend(loc='lower right')
                    plt.xlabel("Iteracje")
                    plt.ylabel("Średnia nagroda")
                    plt.title(
                        f'Średni $\epsilon-greedy$ Nagroda po {episode} epizodach Tau = {tau} Alpha = {alpha} ilość reklam {k}')
                    plt.show()
                    # plt.savefig(f'e{episode}_t{tau}_a{alpha}.png')
                    eps_0_rewards = np.zeros(iters)
                    habs = []
                    habibi = []
