from functions import *

def runLocal(train_loader,test_loader,num_clients,batch_size,selected_agent_index,epochs):

    print("=== Local ===")
    np.set_printoptions(precision=3)

    # Instantiate models and optimizers
    shared_model = Net().cuda()
    client_models = [Net().cuda() for _ in range(num_clients)]
    for model in client_models:
        model.load_state_dict(shared_model.state_dict())

    opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

    grad_vector = [None for _ in range(num_clients)]
    weight_vector = np.ones(num_clients)

    # client update
    loss = np.zeros(num_clients)
    for i in range(num_clients):
        print('%d-th Client' % i)
        loss_tmp, grad_vector[i] = client_update(client_models[i], opt[i], train_loader[i], epoch=epochs)
        loss[i] = loss_tmp

    # Evalutate on the global test set (for now)
    test_loss, acc = evaluate(client_models[selected_agent_index], test_loader)

    print(f"Loss   : {loss}")
    print('Test loss %0.3g | Test acc: %0.3f\n' % (test_loss, acc))
