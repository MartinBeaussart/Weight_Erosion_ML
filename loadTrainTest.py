from torchvision import datasets, transforms
import torch
import math

# === Distribution functions === #


def get_non_iid_loader_distribution(num_clients,batch_size,distribution,selected_agent_index, validation_size=0.1):
    """Get the train and test set based on a specific distribution """

    # Get train and test data of MNIST_
    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    testdata = datasets.MNIST('./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    # Find data for each targets value [0,1,..,9]. We do this for train and test set
    target_labels = torch.stack([traindata.targets == i for i in range(10)])
    target_labels_test = torch.stack([testdata.targets == i for i in range(10)])
    target_labels_split = []
    target_labels_split_test = []

    # Divide each target labels in small samples
    target_label_division = 100 # Number of samples for each digits
    for i in range(10):
        target_labels_data = torch.where(target_labels[i])[0]    # We get the train data for the current target value (i)
        target_labels_data_test = torch.where(target_labels_test[i])[0] # We get the test data for the current target value (i)

        # Split data in 100 (target_label_division) subsamples
        target_labels_split += torch.split(target_labels_data, int((len(target_labels_data)) / (target_label_division-1)))
        target_labels_split_test += torch.split(target_labels_data_test, int((len(target_labels_data_test))))

        target_labels_split = target_labels_split[:target_label_division*(i+1)] # Sometimes we get more than 100 values (the digits do not have th exact same amount). We take the first 100 samples.

    # We now have 10 (unique targets values) * 100 (subsample for each target value) = 1000 samples

    # Merge selected samples for each client based on the distribution
    savedDistribution = distribution
    distribution = [target_label_division * x / (max(num_clients,10)/10) for x in distribution] # Adapt the initial distribution depending on the number of sample and number of clients
    samples_used = [0,0,0,0,0,0,0,0,0,0]
    next_samples_used = [0,0,0,0,0,0,0,0,0,0]
    split_client = []
    test_data = torch.tensor([],dtype=torch.long)

    # Add data for each clients
    for i in range(num_clients):
        split_client.append(torch.tensor([],dtype=torch.long))

        # The array sample_used represent where we actually are in the different samples
        # Distribution represent the distribution of our current client
        # Thus, to get the next sample to use, we have to add samples_used and distribution, for each digits
        for n in range(10):
            next_samples_used[n] = samples_used[n] + distribution[n]

        # Update the distribution by shifting to the left after each clients
        distribution = distribution[1:] + distribution[:1]

        # For each digits
        for number in range(10):
            # Add data to the Test loader if it's the good client
            if i == selected_agent_index and samples_used[number] < next_samples_used[number]:
                # Get the size of the two chunks
                sizeDataTest = int(savedDistribution[number] * len(target_labels_split_test[number]))
                sizeDataTestLeft = len(target_labels_split_test[number]) - sizeDataTest

                # Split the data to get the same distribution as the train set, then stores the data in test_data
                t1, t2 = torch.split(target_labels_split_test[number], [sizeDataTest,sizeDataTestLeft])
                test_data = torch.cat((test_data, t1),0)

            # Add data to train set, as long as we can do it
            while samples_used[number] < next_samples_used[number]:
                # Get the first value for our digit by doing the computation number * number of chunks per digit + current sample for our number
                # Ex: We compute, for the number 8 if we are currently at the 43 value: 8 * 100 + 43 = 843
                begining_chunk_target_value = number*target_label_division
                id_data_available = begining_chunk_target_value+samples_used[number]
                split_client[i] = torch.cat((split_client[i], target_labels_split[id_data_available]),0)
                samples_used[number] += 1

            # Security if the sum of distribution > 1 (which should not be the case)
            if samples_used[number] > next_samples_used[number]:
                samples_used[number] -= 1

    # Split the data in test and train loader
    traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in split_client]
    testdata_split = torch.utils.data.Subset(testdata, test_data)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True,pin_memory=True) for x in traindata_split]
    test_loader = torch.utils.data.DataLoader(testdata_split, batch_size=batch_size, shuffle=True,pin_memory=True)

    return train_loader, test_loader


def get_distributions_non_iid_different_sizes(num_clients,batch_size,distribution,selected_agent_index,distribution_size, validation_size=0.1):
    """Get the train and test set based on a specific distribution """

    # Get train and test data of MNIST_
    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    testdata = datasets.MNIST('./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    # Find data for each targets value [0,1,..,9]. We do this for train and test set
    target_labels = torch.stack([traindata.targets == i for i in range(10)])
    target_labels_test = torch.stack([testdata.targets == i for i in range(10)])
    target_labels_split = []
    target_labels_split_test = []

    # Divide each target labels in small samples
    target_label_division = 100 # Number of samples for each digits
    for i in range(10):
        target_labels_data = torch.where(target_labels[i])[0]    # We get the train data for the current target value (i)
        target_labels_data_test = torch.where(target_labels_test[i])[0] # We get the test data for the current target value (i)

        # Split data in 100 (target_label_division) subsamples
        target_labels_split += torch.split(target_labels_data, int((len(target_labels_data)) / (target_label_division-1)))
        target_labels_split_test += torch.split(target_labels_data_test, int((len(target_labels_data_test))))

        target_labels_split = target_labels_split[:target_label_division*(i+1)] # Sometimes we get more than 100 values (the digits do not have th exact same amount). We take the first 100 samples.

    # We now have 10 (unique targets values) * 100 (subsample for each target value) = 1000 samples

    # Merge selected samples for each client based on the distribution
    savedDistribution = distribution
    distribution = [target_label_division * x / (max(num_clients,10)/10) for x in distribution] # Adapt the initial distribution depending on the number of sample and number of clients

    samples_used = [0,0,0,0,0,0,0,0,0,0]
    next_samples_used = [0,0,0,0,0,0,0,0,0,0]
    split_client = []
    test_data = torch.tensor([],dtype=torch.long)

    # Calculat the max number of batch we obtain for each digits respecting constraints on both distributions
    # we want this because both constraints create more or less batch and we want 100, we can go below 100 but can't have more than 100 so we divide by the max
    nb_batchs_by_digits = [0,0,0,0,0,0,0,0,0,0]
    distribution_updated_size_list = []

    for i in range(num_clients):
        distribution_updated_size = [x*(distribution_size[i%10])*100 for x in distribution]
        distribution = distribution[1:] + distribution[:1]

        for n in range(10):
            nb_batchs_by_digits[n] += (distribution_updated_size[n])

        distribution_updated_size_list.append(distribution_updated_size)

    max_batchs_digits = max(nb_batchs_by_digits)/100


    # Add data for each clients
    for i in range(num_clients):
        split_client.append(torch.tensor([],dtype=torch.long))

        # Update the distribution based on the size we want for that client and without overflow the limit of 100
        distribution_updated_size = [x*(distribution_size[i%10])*100/max_batchs_digits for x in distribution]

        # The array sample_used represent where we actually are in the different samples
        # Distribution represent the distribution of our current client
        # Thus, to get the next sample to use, we have to add samples_used and distribution, for each digits
        for n in range(10):
            next_samples_used[n] = samples_used[n] + distribution_updated_size[n]

        # Update the distribution by shifting to the left after each clients
        distribution = distribution[1:] + distribution[:1]


        # For each digits
        for number in range(10):
            # Add data to the Test loader if it's the good client
            if i == selected_agent_index and samples_used[number] < next_samples_used[number]:
                # Get the size of the two chunks
                sizeDataTest = int(savedDistribution[number] * len(target_labels_split_test[number]))
                sizeDataTestLeft = len(target_labels_split_test[number]) - sizeDataTest

                # Split the data to get the same distribution as the train set, then stores the data in test_data
                t1, t2 = torch.split(target_labels_split_test[number], [sizeDataTest,sizeDataTestLeft])
                test_data = torch.cat((test_data, t1),0)

            # Add data to train set, as long as we can do it
            while samples_used[number] < next_samples_used[number]:
                # Get the first value for our digit by doing the computation number * number of chunks per digit + current sample for our number
                # Ex: We compute, for the number 8 if we are currently at the 43 value: 8 * 100 + 43 = 843
                begining_chunk_target_value = number*target_label_division
                id_data_available = begining_chunk_target_value+samples_used[number]
                split_client[i] = torch.cat((split_client[i], target_labels_split[id_data_available]),0)
                samples_used[number] += 1

            # Security if the sum of distribution > 1 (which should not be the case)
            if samples_used[number] > next_samples_used[number]:
                samples_used[number] -= 1

    # Split the data in test and train loader
    traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in split_client]
    testdata_split = torch.utils.data.Subset(testdata, test_data)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True,pin_memory=True) for x in traindata_split]
    test_loader = torch.utils.data.DataLoader(testdata_split, batch_size=batch_size, shuffle=True,pin_memory=True)

    return train_loader, test_loader


def get_iid_loader(num_clients,batch_size):
    """Get a i.i.d test set and train set"""
    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_iid_loader_with_validation(num_clients, batch_size, validation_size=0.1):
    """Get i.i.d data with validation set"""
    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    testdata = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))

    traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
    testdata_split = torch.utils.data.random_split(testdata, [int(testdata.data.shape[0] / num_clients) for _ in range(num_clients)])

    train_loader = []
    validation_loader = []

    for x in traindata_split:
        x_size = len(x)
        size_train = int(math.ceil(x_size * (1 - validation_size)))
        size_validation = int(math.floor(x_size * validation_size))
        train_set, validation_set = torch.utils.data.random_split(x, [size_train, size_validation])

        train_loader.append(torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True))
        validation_loader.append(torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True))

    test_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in testdata_split]


    return train_loader, validation_loader, test_loader


def get_non_IID_loader_digit_pairs(num_clients,batch_size):
    """Generates train and test loader for digits pairs (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)"""
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


def generate_permutations(nb_agents=5, sample_size=3):
    """
    Generate random permutations with sample_size digits for each agents (number of agents = nb_agents)
    It ensures that each digits is assigned at least once
    """
    available_labels = np.array([0,1,2,3,4,5,6,7,8,9])
    triplets = {}

    valid = False
    while not valid :
        all_digits = []
        for i in range(nb_agents):
            triplets[i] = np.random.choice(available_labels,sample_size,replace=False)
            all_digits.extend(triplets[i])
        valid = len(np.unique(all_digits)) == len(available_labels)

    return triplets


def get_non_IID_loader_digit_trios(num_clients,batch_size):
    """Get train and test loader for random trios of digits"""
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
