# Contextual Linear Algorithm (Unoptimized)

import numpy
import random
import time

trials = 100
n_arms = 5
n_params = 3

data = []
feature_history = []
reward = 0
running_optimal = 0

true_params = [numpy.round(random.uniform(0, 1), 3) for i in range(n_params)]
true_sds = [0.5 for i in range(n_arms)]

print(true_params)
print(true_sds)

time.sleep(5)

params_estimate = numpy.random.uniform(0, 1, n_params)

for i in range(1, trials + 1):
    features = [[numpy.round(random.uniform(0, 1), 3) for j in range(n_params)] for k in range(n_arms)]
    optimal = max([numpy.inner(features[j], true_params) for j in range(n_arms)])
    optimal_arm = numpy.argmax(numpy.array([numpy.inner(features[j], true_params) for j in range(n_arms)]))

    arms = [numpy.random.normal(numpy.inner(features[j], true_params), true_sds[j]) for j in range(n_arms)]
    running_optimal += optimal

    max_ind = numpy.argmax(numpy.array([numpy.inner(features[j], params_estimate) for j in range(n_arms)]))

    arm = arms[max_ind]

    data.append(arm)
    feature_history.append(features[max_ind])
    reward += arm

    # Random noise added to Gramian matrix to prevent zero determinant and guarantee non-singularity
    feature_gramian = numpy.matmul(numpy.transpose(feature_history), feature_history) + numpy.random.normal(0, 0.0001 / i, (n_params, n_params))
    moment_matrix = numpy.matmul(numpy.transpose(feature_history), numpy.array(data))

    # Modified Ordinary Least Squares
    params_estimate = numpy.matmul(numpy.linalg.inv(feature_gramian), moment_matrix)

    print(f"Chosen Arm: {max_ind}, Optimal Arm: {optimal_arm}, Arm Reward: {arm}, Regret: {100 * ((optimal - arm) / optimal)}%")
    
    print(f"Running Pseudoregret: {100 * ((running_optimal - reward) / reward)}%")
    time.sleep(0.5)