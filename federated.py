from functions import *
import pickle

def runFederated(train_loader,test_loader,num_clients,batch_size,selected_agent_index,num_rounds,epochs,distribution):

    print("=== Federated ===")
    np.set_printoptions(precision=3)
    acc_best = 0
    round_best = 0
    weight_best = [0.1,0,0,0,0,0,0,0,0,0]
    dataPickle = []

    # Instantiate models and optimizers
    shared_model = Net().cuda()
    client_models = [Net().cuda() for _ in range(num_clients)]
    for model in client_models:
        model.load_state_dict(shared_model.state_dict())

    opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

    grad_vector = [None for _ in range(num_clients)]
    weight_vector = np.ones(num_clients)

    for r in range(num_rounds):

        print('%d-th round' % r)

        # client update
        loss = np.zeros(num_clients)
        for i in range(num_clients):
            loss_tmp, grad_vector[i] = client_update(client_models[i], opt[i], train_loader[i], epoch=epochs)
            loss[i] = loss_tmp
            weight_vector[i] = 1/num_clients


        # Weight Erosion Scheme
        weighted_mean_gradient = weighted_average_gradients(grad_vector, weight_vector)
        shared_model = update_grad(shared_model, weighted_mean_gradient, 0.1)

        # Share model to all agents
        share_weight_erosion_model(shared_model, client_models)

        # Evalutate on the global test set (for now)
        test_loss, acc = evaluate(shared_model, test_loader)

        print(f"Loss   : {loss}")
        print('Test loss %0.3g | Test acc: %0.3f\n' % (test_loss, acc))
        dataPickle.append([acc,test_loss,loss[selected_agent_index]])

        if acc > acc_best:
            acc_best = acc
            round_best = r+1
            weight_best = weight_vector

    with open("./data/federated_"+str(num_clients)+"-"+str(distribution)+"_m.pickle", 'wb') as f:
        pickle.dump(dataPickle, f)
    return [acc_best, round_best, weight_best]
