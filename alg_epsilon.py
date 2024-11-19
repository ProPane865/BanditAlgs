# Epsilon-Greedy Algorithm

import numpy
import random
import time

trials = 100
epsilon = 0.1
n_arms = 5

data = [[] for i in range(n_arms)]
reward = 0
running_optimal = 0

true_means = [random.uniform(0.3, 0.7) for i in range(n_arms)]
true_sds = [0.5 for i in range(n_arms)]

print(true_means)
print(true_sds)

time.sleep(5)

optimal = max(true_means)

for i in range(1, trials + 1):

    arms = [numpy.random.normal(true_means[j], true_sds[j]) for j in range(n_arms)]
    running_optimal += optimal

    if random.random() < epsilon:
        index = random.choice(range(len(arms)))
        arm = arms[index]

        data[index].append(arm)
        reward += arm

        print(f"Explored: {index}, Arm Reward: {arm}, Regret: {100 * ((optimal - arm) / optimal)}%")
    else:
        means = numpy.array([numpy.mean(numpy.array(data[0])), numpy.mean(numpy.array(data[1])), numpy.mean(numpy.array(data[2]))])
        max_ind = numpy.argmax(means)

        arm = arms[max_ind]

        data[max_ind].append(arm)
        reward += arm

        print(f"Exploited: {max_ind}, Arm Reward: {arm}, Regret: {100 * ((optimal - arm) / optimal)}%")
    
    print(f"Running Pseudoregret: {100 * ((running_optimal - reward) / reward)}%")
    time.sleep(0.5)