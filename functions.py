import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Define the CNN we are using for our task
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


# This class represents the gradient for on NN model. Keeping Gradient for each layer distinctly stored.
# It also allows us to compute easily the relative distance between the gradient of two agents for Weight Erosion.
class GradientStocker:
    def __init__(self, model_names):
        for item in model_names:
            setattr(self, item, 0)

    def get_attributes(self):
        return self.__dict__

    def add_gradient(self, model):
        for name, param in model.named_parameters():
            setattr(self, name, getattr(self, name) + param.grad.data.cpu())

    def euclidian_distance(self, grad_current_agent):
        """Computes the relative euclidean distance of the flattened tensor between the current model and the global model"""
        flattened_grad_selected = self.flatten(list(self.get_attributes().values()))
        flattened_grad_current = self.flatten(list(grad_current_agent.get_attributes().values()))
        return torch.dist(flattened_grad_selected, flattened_grad_current, 2) / torch.norm(flattened_grad_selected, 2)

    def flatten(self, gradient_list):
        """Returns an aggregated tensor of all the gradients for one model"""
        gradients = list(map(lambda g : torch.flatten(g), gradient_list))
        return torch.cat(gradients, 0)


def client_update(client_model, optimizer, train_loader, epoch=5):
    """Train a client_model on the train_loder data."""
    model_names = []
    for name, param in client_model.named_parameters():
        model_names.append(name)
    gradient_stocker = GradientStocker(model_names)
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            gradient_stocker.add_gradient(client_model)
    return loss.item(), gradient_stocker


def weighted_average_gradients(gradients, weights):
    """Compute the weighted average gradient."""
    weighted_averages = {}
    for key in gradients[0].get_attributes().keys():
        weighted_averages[key] = weighted_average_from_key(key, gradients, weights)
    return weighted_averages

def weighted_average_from_key(key, gradients, weights):
    n = 0
    d = 0
    for idx, g_dict in enumerate(gradients) :
        n += g_dict.get_attributes()[key] * weights[idx]
        d += weights[idx]
    return n / d

def compute_weight(alpha_prev, round, relative_distance, data_size, batch_size, distance_penalty, size_penalty):
    """ Computes the weight alpha for round r """
    size_factor = (1 + size_penalty * math.floor(((round - 1) * batch_size) / data_size))
    distance_factor = distance_penalty * relative_distance
    alpha = alpha_prev - size_factor * distance_factor
    return max(0,alpha)

def update_grad(model, gradient, alpha): 
    """ Update the gradient for all parameters"""
    for name, param in model.named_parameters():
        param.data -= gradient[name].cuda() * alpha
    return model

def share_weight_erosion_model(shared_model, client_models):
    """ Share the computed model with all agents"""
    for model in client_models:
        model.load_state_dict(shared_model.state_dict())

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
