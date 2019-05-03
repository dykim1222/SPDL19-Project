import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torchvision import datasets, transforms

import numpy as np
import pdb

#change to path to dataset
# path = '/Users/dae/Desktop/dlproject/ssl_data_96'

def dataLabel(class_num, path, train=True):
    """args:
        class_num: The number of samples from each class in the training set
     """
    #import pdb; pdb.set_trace()

    train = 'train' if train else 'val'
    raw_dataset = datasets.ImageFolder('{}/{}/{}'.format(path, 'supervised', train), transform=transforms.ToTensor())
    # print("loaded")
    
    class_tot = [0] * 1000
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        # print(i)
        datum, label = raw_dataset.__getitem__(perm[i]) # [3, 96, 96], class_number
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 1000 * class_num:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))

def dataUnlabel():
    raw_dataset = datasets.ImageFolder('{}/{}'.format(path, 'unsupervised'), transform=transforms.ToTensor())
    data = []
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset[i]
        data.append(datum.numpy())
    return TensorDataset(torch.FloatTensor(np.array(data)))

def dataVal():
    raw_dataset = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transforms.ToTensor())
    data = []
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset[i]
        data.append(datum.numpy())
    return TensorDataset(torch.FloatTensor(np.array(data)))

if __name__ == '__main__':
    # print(dir(dataVal()))

    class_num = 100
    """args:
        class_num: The number of samples from each class in the training set
     """
    #import pdb; pdb.set_trace()
    raw_dataset = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised_test'), transform=transforms.ToTensor())
    print("loaded")
    
    class_tot = [0] * 1000
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        print(i)
        datum, label = raw_dataset.__getitem__(perm[i]) # [3, 96, 96], class_number
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 1000 * class_num:
                break

    train=TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))
    loader = DataLoader(train, batch_size=64)
    model = models.resnet152()
    pdb.set_trace()
    for batch_idx, (x,l) in enumerate(loader):
        pdb.set_trace()
        print(batch_idx)


    # raw_dataset = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised_test'), transform=transforms.ToTensor())
    # data = []
    # for i in range(raw_dataset.__len__()):
    #     datum, label = raw_dataset[i]
    #     data.append(datum.numpy())
