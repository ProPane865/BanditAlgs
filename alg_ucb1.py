# UCB-1 Algorithm

import numpy
import random
import time

trials = 100
n_arms = 5

data = [[] for i in range(n_arms)]
armplays = [1 for i in range(n_arms)]
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

    ucb_mean_sum = numpy.array([numpy.mean(numpy.array(data[j])) + numpy.sqrt(2*numpy.log(i) / armplays[j]) for j in range(n_arms)])
    max_ind = numpy.argmax(ucb_mean_sum)

    arm = arms[max_ind]
    armplays[max_ind] += 1

    data[max_ind].append(arm)
    reward += arm

    print(f"Chosen Arm: {max_ind}, Arm Reward: {arm}, Regret: {100 * ((optimal - arm) / optimal)}%")
    
    print(f"Running Pseudoregret: {100 * ((running_optimal - reward) / reward)}%")
    time.sleep(0.5)