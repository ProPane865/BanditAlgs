# Thompson Sampling Algorithm

import numpy
import random
import time

trials = 100
n_arms = 5

data = [[] for i in range(n_arms)]
armplays = [0 for i in range(n_arms)]
reward = 0
running_optimal = 0

true_means = [random.uniform(0.3, 0.7) for i in range(n_arms)]
true_sds = [0.5 for i in range(n_arms)]

prior_means = [0 for i in range(n_arms)]
prior_sds = [100 for i in range(n_arms)]
estimated_true_sd = 0.5

print(true_means)
print(true_sds)

time.sleep(5)

optimal = max(true_means)

for i in range(1, trials + 1):

    arms = [numpy.random.normal(true_means[j], true_sds[j]) for j in range(n_arms)]
    running_optimal += optimal

    posterior_sds = [numpy.power((1 / prior_sds[j])**2 + (armplays[j] / (estimated_true_sd**2)), -0.5) for j in range(n_arms)]
    posterior_means = [numpy.power(posterior_sds[j], 2) * ((prior_means[j] / (prior_sds[j]**2)) + (numpy.sum(numpy.array(data[j])) / estimated_true_sd**2)) for j in range(n_arms)]

    thompson_samples = [numpy.random.normal(posterior_means[j], posterior_sds[j]) for j in range(n_arms)]
    max_ind = numpy.argmax(thompson_samples)

    arm = arms[max_ind]
    armplays[max_ind] += 1

    data[max_ind].append(arm)
    reward += arm

    print(f"Chosen Arm: {max_ind}, Arm Reward: {arm}, Regret: {100 * ((optimal - arm) / optimal)}%")
    
    print(f"Running Pseudoregret: {100 * ((running_optimal - reward) / reward)}%")
    time.sleep(0.5)