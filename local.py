from functions import *
import pickle
from pathlib import Path

# === Run our model training Locally === #

def run_local(train_loader, test_loader, num_clients, batch_size,
              selected_agent_index, num_rounds, epochs, distribution, distribution_name='distribution'):

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

    acc_best = 0
    round_best = 0
    weight_best = [1,0,0,0,0,0,0,0,0,0]

    print('%d-th Client' % selected_agent_index)
    for r in range(num_rounds):

        print('%d-th round' % r)

        # client update
        loss, grad_vector = client_update(client_models[selected_agent_index], opt[selected_agent_index], train_loader[selected_agent_index], epoch=epochs)

        # Evalutate on the selected agent's test set
        test_loss, acc = evaluate(client_models[selected_agent_index], test_loader)

        # Print the results
        print(f"Loss   : {loss}")
        print('Test loss %0.3g | Test acc: %0.3f\n' % (test_loss, acc))

        # Keep the accuracy for each round
        dataPickle.append([acc,test_loss,loss])

        # Update the best accuracy
        if acc > acc_best:
            acc_best = acc
            round_best = r+1

    with open(Path.cwd()/'generated'/'pickles'/f'local_{num_clients}_{distribution_name}.pickle', 'wb') as f:
        pickle.dump(dataPickle, f)

    return [acc_best, round_best, weight_best]
