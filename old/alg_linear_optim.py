# Contextual Linear Algorithm

import numpy
import random
import time

slow = False

trials = 10000
n_arms = 5
n_params = 3

reward = 0
running_optimal = 0

true_params = numpy.array([numpy.round(random.uniform(0, 1), 3) for i in range(n_params)]).reshape((1, n_params))
true_sds = [0.5 for i in range(n_arms)]

print(true_params)
print(true_sds)

regularization = 0.01

gramian = numpy.identity(n_params) * regularization
moment_matrix = numpy.zeros((n_params, 1))

params_estimate = numpy.matmul(numpy.linalg.inv(gramian), moment_matrix).reshape((1, n_params))

time.sleep(5)

for i in range(1, trials + 1):
    features = numpy.array([[numpy.round(random.uniform(0, 1), 3) for j in range(n_params)] for k in range(n_arms)])
    mean_rewards = numpy.inner(true_params, features)
    optimal = numpy.max(mean_rewards)
    optimal_arm = numpy.argmax(mean_rewards)

    arms = [numpy.random.normal(mean_rewards[0][j], true_sds[j]) for j in range(n_arms)]
    running_optimal += optimal

    est_rewards = numpy.inner(params_estimate, features)
    max_ind = numpy.argmax(est_rewards)

    reward += arms[max_ind]

    arm = numpy.array(arms[max_ind]).reshape((1, 1))
    selected = features[max_ind].reshape((n_params, 1))

    gramian += numpy.outer(selected, selected)
    moment_matrix += numpy.matmul(selected, arm)

    params_estimate = numpy.matmul(numpy.linalg.inv(gramian), moment_matrix).reshape((1, n_params))
    params_difference = numpy.linalg.norm(true_params - params_estimate) / numpy.sqrt(n_params)
    print(params_difference)

    print(f"Chosen Arm: {max_ind}, Optimal Arm: {optimal_arm}, Arm Reward: {arm}, Regret: {100 * ((optimal - arm) / optimal)}%")
    
    print(f"Running Pseudoregret: {100 * ((running_optimal - reward) / reward)}%")

    if slow:
        time.sleep(0.25)