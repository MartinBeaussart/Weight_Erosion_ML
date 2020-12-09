from weightErosion import *
from federated import *
from local import *
from loadTrainTest import *

from cumulator import base

#=== parameters for Schemes
selected_agent_index = 0
num_rounds = 10
epochs = 1

#=== parameters for training and testing
num_clients = 10 #if num_clients < 10, sum(distribution) should be = 10/num_clients with max 1 at each index
batch_size = 32
distribution = [0,0,0,0.25,0.5,0.25,0,0,0,0]

#distribution = [[0,0,0,0,0,1,0,0,0,0],
#[0,0,0,0,0.25,0.5,0.25,0,0,0]]
#nbClients = [5,10,20,50,100]
#for distribution in distributions:
#    for num_clients in nbClients:
train_loader, test_loader = get_non_iid_loader_distribution(num_clients,batch_size,distribution,selected_agent_index)

#cumulator not done yet
cumulator = base.Cumulator()
cumulator.on()

#runWeightErosion(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs)
#runFederated(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs)
runLocal(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds*epochs)

cumulator.off()
dontknow = cumulator.computation_costs()
