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
num_clients = 50
batch_size = 32
homogeneity = True

train_loader, test_loader = getLoader(num_clients,batch_size,homogeneity) #lot of change needed 

#cumulator not done yet
cumulator = base.Cumulator()
cumulator.on()

runWeightErosion(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs)
runFederated(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs)
runLocal(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds*epochs)

cumulator.off()
dontknow = cumulator.computation_costs()
