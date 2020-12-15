import numpy as np 
import matplotlib.pyplot as plt

def plot(nb_round, WE_result, FedAvg_result, local_result, weight_avg, title):
    plt.figure(num=None, figsize=(9, 7), facecolor='w', edgecolor='k')
    plt.plot(nb_round, WE_result, '-xb', label='WE')
    plt.plot(nb_round, FedAvg_result, '-og', label="FedAvg")
    plt.plot(nb_round, local_result, '-+k', label="local")
    plt.plot(nb_round, weight_avg, color='r', linestyle='dashed', label='Average weight')
    plt.title(title)
    ax = plt.gca()
    ax.legend(loc=3)
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Test accuracy / Agents Average weight", fontsize=12)
    plt.show()

nb_round = np.arange(0, 30, 1)
WE_result = np.random.rand(30)
FedAvg_result = np.random.rand(30)
local_result = np.random.rand(30)
weight_avg = np.concatenate((np.arange(1, 0, - 0.05), np.zeros(10)), axis=None)

plot(nb_round, WE_result, FedAvg_result, local_result, weight_avg, "enter your title here")