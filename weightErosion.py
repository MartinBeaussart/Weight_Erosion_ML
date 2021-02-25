from functions import *
import pickle
from pathlib import Path

# === Run our model training using the Weight Erosion aggregation scheme === #

def run_weight_erosion(train_loader, test_loader, num_clients, batch_size,
                       selected_agent_index, num_rounds, epochs, distribution,size_penalty, distribution_name='distribution'):
    """This function implements the weight erosion scheme for neural networks"""

    # As explained in our paper, we found out that the best distance penalty was 0.1/num_clients
    distance_penalty = 0.04
    if size_penalty == 2:
        distance_penalty = 0.1/num_clients
    #size_penalty = 0
    dataPickle = []

    print("=== Weight Erosion ===")
    #print("Distance_penalty: "+distance_penalty+" size_penalty: "+size_penalty)

    # We have to store the values corresponding to the best test score
    np.set_printoptions(precision=3)
    acc_best = 0
    round_best = 0
    weight_best = [0.1,0,0,0,0,0,0,0,0,0]

    # Instantiate models and optimizers
    shared_model = Net().cuda()
    client_models = [Net().cuda() for _ in range(num_clients)]
    for model in client_models:
        model.load_state_dict(shared_model.state_dict())
    opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

    # Initialize our grad vector to None as there is no grad yet, and our weight vector as 1
    grad_vector = [None for _ in range(num_clients)]
    weight_vector = np.ones(num_clients)


    for r in range(num_rounds):

        print('%d-th round' % r)

        time_updateModel = 0

        # Client update: computation of the gradients of the client, of its relative distance to the selected agent, and its new weight vector.
        loss = np.zeros(num_clients)

        #######################################################33
        processes = []

        for i in range(num_clients):

            loss[i], grad_vector[i] = client_update(client_models[i], opt[i], train_loader[i], epoch=epochs)
            d_rel = grad_vector[selected_agent_index].euclidian_distance(grad_vector[i])
            weight_vector[i] = compute_weight(weight_vector[i], r + 1, d_rel, len(train_loader[i]), batch_size, distance_penalty, size_penalty)


        # Computation of the personalized model
        weighted_mean_gradient = weighted_average_gradients(grad_vector, weight_vector)
        shared_model = update_grad(shared_model, weighted_mean_gradient, 0.1)

        # Share model to all agents
        share_weight_erosion_model(shared_model, client_models)

        # Evalutate on the agent's test set
        test_loss, acc = evaluate(shared_model, test_loader)


        print(f'Weight : {weight_vector}')
        print(f'Loss   : {loss}')
        print(f'Test loss {test_loss:.3f} | Test acc: {acc:.3f} \n')

        # Keep the accuracy for each round
        dataPickle.append([acc,test_loss,loss[selected_agent_index], sum(weight_vector)/num_clients])

        # Update the best accuracy
        if acc > acc_best:
            acc_best = acc
            round_best = r+1
            weight_best = weight_vector

    # Stores the important informations for each round
    with open("./generated/pickles/weight_erosion_"+str(num_clients)+"-"+distribution_name+".pickle", 'wb') as f:
        pickle.dump(dataPickle, f)

    return [acc_best, round_best, weight_best]
