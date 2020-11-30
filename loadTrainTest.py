from torchvision import datasets, transforms
import torch


def getLoader(num_clients,batch_size,homogeneity):
    if homogeneity:
        traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),
            (0.3081,))])), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader
    else:
        traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        target_labels = torch.stack([traindata.targets == i for i in range(10)])
        target_labels_split = []
        for i in range(5):
            target_labels_split += torch.split(torch.where(target_labels[(2 * i):(2 * (i + 1))].sum(0))[0], int(60000 / num_clients))
        traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in target_labels_split]
        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader
