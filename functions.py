import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def client_update(client_model, optimizer, train_loader, epoch=5):
    """Train a client_model on the train_loder data."""
    gradient = create_gradient(client_model)
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            gradient = update_gradient(gradient,client_model)
    return loss.item(), gradient

def relative_distance(grad_selected_agent, grad_current_agent):
    """Computes the relative distance between the current model and the global model"""
    return torch.dist(grad_selected_agent, grad_current_agent, 2) / torch.norm(grad_selected_agent, 2)

def compute_weight(alpha_prev, round, relative_distance, data_size, batch_size, distance_penalty, size_penalty):
    """Computes the weight alpha for round r"""
    size_factor = (1 + distance_penalty * math.floor(((round - 1) * batch_size) / data_size))
    distance_factor = distance_penalty * relative_distance
    alpha = alpha_prev - size_factor * distance_factor
    return max(0,alpha)


def get_gradient_dict_from_model(model):
    """ Returns a list of the gradient for each parameter"""
    dict_gradients= {}
    for name, param in model.named_parameters():
        dict_gradients[name] = param.grad.data.cpu()
    return dict_gradients

def create_gradient(model):
    """ Returns an empty gradient"""
    dict_params= {}
    for name, param in model.named_parameters():
        dict_params[name] = 0
    return dict_params

def update_gradient(gradient, model):
    """ update global gradient with last computed gradient"""
    new_gradient = get_gradient_dict_from_model(model)
    for name, param in model.named_parameters():
        gradient[name] += new_gradient[name]
    return gradient

def relative_distance_vector(grad_selected_agent, grad_current_agent):
    """Computes the relative euclidean distance of the flattened tensor between the current model and the global model"""
    grad_selected = get_gradient_tensor(list(grad_selected_agent.values()))
    grad_current = get_gradient_tensor(list(grad_current_agent.values()))
    return torch.dist(grad_selected, grad_current, 2) / torch.norm(grad_selected, 2)


# Compute the new weighted average gradient
def weighted_average_gradients(gradients, weights):
    """Compute the weighted average gradient."""
    weighted_averages = {}
    for key in gradients[0].keys():
      weighted_averages[key] = weighted_average_from_key(key, gradients, weights)
    return weighted_averages

def weighted_average_from_key(key, gradients, weights):
  n = 0
  d = 0
  for idx, g_dict in enumerate(gradients) :
    n += g_dict[key] * weights[idx]
    d += weights[idx]
  return n / d

# alpha
def compute_weight(alpha_prev, round, relative_distance, data_size, batch_size, distance_penalty, size_penalty):
    """Computes the weight alpha for round r"""
    size_factor = (1 + distance_penalty * math.floor(((round - 1) * batch_size) / data_size))
    distance_factor = distance_penalty * relative_distance
    alpha = alpha_prev - size_factor * distance_factor
    return max(0,alpha)

# Utilitary functions to get NN model
def get_gradient_list_from_model(model):
    """ Returns a list of the gradient for each parameter"""
    parameters = list(model.parameters())
    gradients = list(map(lambda p : p.grad.data.cpu(), parameters))
    return gradients

def get_gradient_tensor(gradient_list):
    """ Returns an aggregated tensor of all the gradients for one model"""
    gradients = list(map(lambda g : torch.flatten(g), gradient_list))
    return torch.cat(gradients, 0)

def update_grad(model, gradient, alpha):
  for name, param in model.named_parameters():
    param.data -= gradient[name].cuda() * alpha
  return model

def update_models(models):
  for model in models:
    for name, param in model.named_parameters():
      param.data -= models[0].parameters()[param].data


def average_models(global_model, client_models):
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)


def evaluate(global_model, data_loader):
    """Compute loss and accuracy of a model on a data_loader."""
    global_model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc
