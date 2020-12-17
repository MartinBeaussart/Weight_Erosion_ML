from torchvision import datasets, transforms
import torch
import math

def get_non_iid_loader_distribution(num_clients,batch_size,distribution,selected_agent_index, validation_size=0.1):
    """ get the train and test set based on a specific distribution """
    # get train and test data of MNIST_
    traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    testdata = datasets.MNIST('./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    # find data for each targets value [0,1,..,9l
    target_labels = torch.stack([traindata.targets == i for i in range(10)])
    target_labels_test = torch.stack([testdata.targets == i for i in range(10)])
    target_labels_split = []
    target_labels_split_test = []

    # divide each target labels in small samples
    target_label_division = 100 #how many sample for each targets value we want
    for i in range(10):
        target_labels_data = torch.where(target_labels[i])[0]    # train data for the current target value (i)
        target_labels_data_test = torch.where(target_labels_test[i])[0] # test data for the current target value (i)

        #split data in 100 (target_label_division) subsamples
        target_labels_split += torch.split(target_labels_data, int((len(target_labels_data)) / (target_label_division-1)))
        target_labels_split_test += torch.split(target_labels_data_test, int((len(target_labels_data_test))))

        target_labels_split = target_labels_split[:target_label_division*(i+1)] #remove when the split not givin you target_label_division samples but target_label_division +1 samples

    # we now have 10 (unique targets values) * 100 (subsample for each target value) = 1000 samples

    #merge selected samples for each client based on the distribution
    savedDistribution = distribution
    distribution = [target_label_division * x / (max(num_clients,10)/10) for x in distribution] #update distribution depending on the number of sample and number of clients
    samples_used = [0,0,0,0,0,0,0,0,0,0]
    next_samples_used = [0,0,0,0,0,0,0,0,0,0]
    split_client = []
    test_data = torch.tensor([],dtype=torch.long)

    #add data for each clients
    for i in range(num_clients):
        split_client.append(torch.tensor([],dtype=torch.long))

        # update when we should stop add samples to the current client
        for n in range(10):
            next_samples_used[n] = samples_used[n] + distribution[n]

        # update the distribution by shifting to the left after each clients
        distribution = distribution[1:] + distribution[:1]

        for number in range(10): #for each target value
            #add data to test if it's the good client
            if i == selected_agent_index and samples_used[number] < next_samples_used[number]:
                # get the size of the two chuncks
                sizeDataTest = int(savedDistribution[number] * len(target_labels_split_test[number]))
                sizeDataTestLeft = len(target_labels_split_test[number]) - sizeDataTest

                t1, t2 = torch.split(target_labels_split_test[number], [sizeDataTest,sizeDataTestLeft])
                test_data = torch.cat((test_data, t1),0)

            # add data to train
            while samples_used[number] < next_samples_used[number]: #while we can add data
                begining_chunck_target_value = number*target_label_division
                id_data_available = begining_chunck_target_value+samples_used[number]
                split_client[i] = torch.cat((split_client[i], target_labels_split[id_data_available]),0)
                samples_used[number] += 1

            if samples_used[number] > next_samples_used[number]:
                samples_used[number] -= 1


    traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in split_client]
    testdata_split = torch.utils.data.Subset(testdata, test_data)
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
    test_loader = torch.utils.data.DataLoader(testdata_split, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_iid_loader(num_clients,batch_size):
      traindata = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
      traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])
      train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]
      test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size, shuffle=True)

      return train_loader, test_loader

def get_iid_loader_with_validation(num_clients, batch_size, validation_size=0.1):
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

# We want to give each agent 3 different digits
# I'd say we do want to have all digitis at least once
def generate_permutations(nb_agents=5, sample_size=3):
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
