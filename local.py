from functions import *
import pickle
from pathlib import Path

# === Run our model training Locally === #

def run_local(train_loader, test_loader, num_clients, batch_size,
              selected_agent_index, num_rounds, epochs, distribution, distribution_name='distribution'):
    """This function implements local training for neural network. We use the same scheme as Federated average, but we train only for the selected agent index"""

    print("=== Local ===")
    np.set_printoptions(precision=3)
    dataPickle = []

    # Instantiate models and optimizers
    client_model = Net().cuda()
    opt = optim.SGD(client_model.parameters(), lr=0.1)

    acc_best = 0
    round_best = 0

    for r in range(num_rounds):

        print('%d-th round' % r)

        # Client update
        train_loss, grad_stocker = client_update(client_model, opt, train_loader[selected_agent_index], epoch=epochs)

        # Evalutate on the selected agent's test set
        test_loss, acc = evaluate(client_model, test_loader)

        # Print the results
        print(f'Loss   : {train_loss}')
        print(f'Test loss {test_loss:.3f} | Test acc: {acc:.3f} \n')

        # Keep the accuracy for each round
        dataPickle.append([acc,test_loss,train_loss])

        # Update the best accuracy
        if acc > acc_best:
            acc_best = acc
            round_best = r+1

    with open("./generated/pickles/local_"+str(num_clients)+"-"+distribution_name+".pickle", 'wb') as f:
        pickle.dump(dataPickle, f)

    return [acc_best, round_best]
