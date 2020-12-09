from torchvision import datasets, transforms
import torch
import math

def get_iid_loader(num_clients,batch_size):
    if homogeneity:
        traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

def get_non_iid_loader_distribution(num_clients,batch_size,distribution,selected_agent_index):
    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    testdata = datasets.MNIST('./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    target_labels = torch.stack([traindata.targets == i for i in range(10)])
    target_labels_test = torch.stack([testdata.targets == i for i in range(10)])
    target_labels_split = []
    target_labels_split_test = []

    #divide each target labels in small samples
    target_label_division = 100 #need to check if with this number we have len(target_labels_split) = 10 * target_label_division
    for i in range(10):
        target_labels_data =torch.where(target_labels[i])[0]

        target_labels_split += torch.split(target_labels_data, int((len(target_labels_data)) / (target_label_division-1)))
        target_labels_split_test += torch.split(torch.where(target_labels_test[i%10])[0], int((len(torch.where(target_labels_test[i])[0]))))

        target_labels_split = target_labels_split[:target_label_division*(i+1)] #remove when the split not givin you target_label_division samples but target_label_division +1 samples

    #merge selected samples in each client
    distribution = [target_label_division * x / (max(num_clients,10)/10) for x in distribution]
    samples_used = [0,0,0,0,0,0,0,0,0,0]
    next_samples_used = [0,0,0,0,0,0,0,0,0,0]
    split_client = []
    test_data = torch.tensor([],dtype=torch.long)

    for i in range(num_clients):
        split_client.append(torch.tensor([],dtype=torch.long))
        for n in range(10):
            next_samples_used[n] = samples_used[n] + distribution[n]
        distribution = distribution[1:] + distribution[:1] # shift to left

        for number in range(10):
            if i == selected_agent_index and samples_used[number] < next_samples_used[number]:
                test_data = torch.cat((test_data, target_labels_split_test[number]),0)

            while samples_used[number] < next_samples_used[number]:
                split_client[i] = torch.cat((split_client[i], target_labels_split[number*target_label_division+samples_used[number]]),0)
                samples_used[number] += 1

            if samples_used[number] > next_samples_used[number]:
                samples_used[number] -= 1

    traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in split_client]
    testdata_split = torch.utils.data.Subset(testdata, test_data)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
    test_loader = torch.utils.data.DataLoader(testdata_split, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_specific_non_IID_loader(num_clients,batch_size,homogeneity):

    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    target_labels = torch.stack([traindata.targets == i for i in range(10)])

    target_labels_split = []
    split_size = int(60000 / num_clients)

    for i in range(num_clients):
        target_labels_split += torch.split(torch.where(target_labels[(2 * i):(2 * (i + 1))].sum(0))[0][:split_size], split_size)

    traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in target_labels_split]
    train_loader = []
    test_loader = []
    for x in traindata_split:
      x_size = len(x)
      size_train = int(math.ceil(x_size * 0.7))
      size_test = int(math.floor(x_size * 0.3))
      #print(x_size == size_train + size_test, size_train, size_test)
      train_set, test_set = torch.utils.data.random_split(x, [size_train, size_test])
      train_loader.append(torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True))
      test_loader.append(torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True))

    return train_loader, test_loader

def get_non_IID_loader_digit_pairs(num_clients,batch_size,homogeneity):

        traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        testdata = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

        train_target_labels = torch.stack([traindata.targets == i for i in range(10)])
        test_target_labels = torch.stack([testdata.targets == i for i in range(10)])

        train_split_size = int(60000 / num_clients)
        test_split_size = int(10000 / num_clients)

        train_target_labels_split = []
        test_target_labels_split = []

        for i in range(num_clients):
            train_target_labels_split += torch.split(torch.where(train_target_labels[(2 * i):(2 * (i + 1))].sum(0))[0][:train_split_size], train_split_size)
            test_target_labels_split += torch.split(torch.where(test_target_labels[(2 * i):(2 * (i + 1))].sum(0))[0][:test_split_size], test_split_size)

        traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in train_target_labels_split]
        testdata_split = [torch.utils.data.Subset(testdata, tl) for tl in test_target_labels_split]

        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
        test_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in testdata_split]

        return train_loader, test_loader

def get_non_IID_loader_digit_trios(num_clients,batch_size,homogeneity):

        traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        testdata = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

        train_target_labels = torch.stack([traindata.targets == i for i in range(10)])
        test_target_labels = torch.stack([testdata.targets == i for i in range(10)])

        train_split_size = int(60000 / num_clients)
        test_split_size = int(10000 / num_clients)

        train_target_labels_split = []
        test_target_labels_split = []

        triplets = generate_permutations(num_clients)

        for i in range(num_clients):
            i_labels = triplets[i]
            print(f"Agent {i} is assigned labels {i_labels}")
            train_target_labels_split += torch.split(torch.where(train_target_labels[i_labels].sum(0))[0][:train_split_size], train_split_size)
            test_target_labels_split += torch.split(torch.where(test_target_labels[i_labels].sum(0))[0][:test_split_size], test_split_size)

        traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in train_target_labels_split]
        testdata_split = [torch.utils.data.Subset(testdata, tl) for tl in test_target_labels_split]

        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
        test_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in testdata_split]

        return train_loader, test_loader
