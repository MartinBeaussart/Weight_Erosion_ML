from functions import *

# IID case: all the clients have images of all the classes
# Hyperparameters

num_clients = 5
num_rounds = 50
epochs = 1
batch_size = 32
distance_penalty = 0.05
size_penalty = 2
selected_agent_index = 0

# weight_vector

weight_vector = np.ones(num_clients)

# Creating decentralized datasets

traindata = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        ), batch_size=batch_size, shuffle=True)
# Instantiate models and optimizers

global_model = Net().cuda()
client_models = [Net().cuda() for _ in range(num_clients)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())

opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

grad_vector = [None for _ in range(num_clients)]
# Runnining Weight Erosion

for r in range(num_rounds):
    # client update
    loss = 0
    for i in range(num_clients):
        loss_tmp, grad_vector[i] = client_update(client_models[i], opt[i], train_loader[i], epoch=epochs)
        loss += loss_tmp
        d_rel = relative_distance_vector(grad_vector[selected_agent_index], grad_vector[i])
        weight_vector[i] = compute_weight(weight_vector[i], r + 1, d_rel, len(train_loader[i]), batch_size, distance_penalty, size_penalty)

    # diffuse params
    weighted_mean_gradient = weighted_average_gradients(grad_vector, weight_vector)
    global_model = update_grad(global_model, weighted_mean_gradient, 0.01)

    #share model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    test_loss, acc = evaluate(global_model, test_loader)

    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_clients, test_loss, acc))
