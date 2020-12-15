from functions import *
import pickle

def runLocal(train_loader,test_loader,num_clients,batch_size,selected_agent_index,epochs,distribution):

    print("=== Local ===")
    np.set_printoptions(precision=3)
    dataPickle = []

    # Instantiate models and optimizers
    shared_model = Net().cuda()
    client_models = [Net().cuda() for _ in range(num_clients)]
    for model in client_models:
        model.load_state_dict(shared_model.state_dict())

    opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

    grad_vector = 0
    weight_vector = np.ones(num_clients)

    # client update
    loss = 0

    print('%d-th Client' % selected_agent_index)
    loss_tmp, grad_vector = client_update(client_models[selected_agent_index], opt[selected_agent_index], train_loader[selected_agent_index], epoch=epochs)
    loss = loss_tmp

    # Evalutate on the global test set (for now)
    test_loss, acc = evaluate(client_models[selected_agent_index], test_loader)

    print(f"Loss   : {loss}")
    print('Test loss %0.3g | Test acc: %0.3f\n' % (test_loss, acc))
    dataPickle.append([acc,test_loss,loss])

    acc_best = acc
    round_best = epochs
    weight_best = [1,0,0,0,0,0,0,0,0,0]

    with open("./data/local_"+str(num_clients)+"-"+str(distribution)+"_m.pickle", 'wb') as f:
        pickle.dump(dataPickle, f)

    return [acc_best, round_best, weight_best]
