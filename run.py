from weightErosion import *
from federated import *
from local import *
from loadTrainTest import *

from cumulator import base
import pickle

#=== parameters for Schemes
selected_agent_index = 0
num_rounds = 30
epochs = 1

#=== parameters for training and testing
batch_size = 32

#distributions = [[0,0,0,0,0.2,0.6,0.2,0,0,0],[0,0,0,0.1,0.2,0.4,0.2,0.1,0,0],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0,0,0,0.4,0.1,0,0.1,0.4,0,0]]
distributions = [[0,0,0.1,0.1,0.2,0.2,0.2,0.1,0.1,0]]
#0.01 = [0.91, 0.1, 0.1 etc..]
#0 = [0.25 * 4 + 0 * 6]

nbClients = [10,20,50,100]

for distribution in distributions:
    for num_clients in nbClients:
        print(' - Number Client %0.3g, distribution: %s' % (num_clients, distribution))
        train_loader, test_loader = get_non_iid_loader_distribution(num_clients,batch_size,distribution,selected_agent_index)

        dataPickle = []

        cumulator = base.Cumulator()
        cumulator.on()
        dataPickle.append(runWeightErosion(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs))
        cumulator.off()
        print('The total carbon footprint for these computations is : ',cumulator.total_carbon_footprint())

        cumulator = base.Cumulator()
        cumulator.on()
        dataPickle.append(runFederated(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs))
        cumulator.off()
        print('The total carbon footprint for these computations is : ',cumulator.total_carbon_footprint())

        cumulator = base.Cumulator()
        cumulator.on()
        dataPickle.append(runLocal(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds*epochs))
        cumulator.off()
        print('The total carbon footprint for these computations is : ',cumulator.total_carbon_footprint())

        with open("./data/"+str(num_clients)+"-"+str(distribution[5])+"_m.pickle", 'wb') as f:
            pickle.dump(dataPickle, f)
