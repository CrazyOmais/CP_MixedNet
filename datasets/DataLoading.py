import h5py
from scipy.io import loadmat
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np


def data_in_one(inputdata):
    min = np.min(inputdata)
    max = np.max(inputdata)
    #print(min, max)
    outputdata = (inputdata-min)/(max-min)
    return outputdata

# EEGDataset class
class EEGDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data    # Tensor(X*(22*1*1125))
        self.Label = Label  # Tensor(X*1)
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index]).squeeze()
        return data, label
        #return self.Data[index], self.Label[index]

# Load dataset
def import_EEGData_train(start=0, end=9, dir = '../data/A0'):
    electrodes = 22
    X, y = [], []

    for i in range(start, end):

        dataFile = loadmat(dir + str(i + 1) + 'T_slice.mat')
        X1 = np.copy(dataFile['image'])  # (1125,22,288)
        X1 = X1[:, :electrodes, np.newaxis, :]  # (1125,22,1,288)
        X1 = np.swapaxes(X1, 0, 3)  # (288,22,1,1125)
        y1 = np.copy(dataFile["type"][: 288])
        y1 = y1 - np.ones(y1.shape)

        # print(np.any(np.isnan(X1)))

        X.extend(X1)
        y.extend(y1)


    delete_list = []
    for i in range(len(X)):
        if np.any(np.isnan(X[i])):
            delete_list.append(i)

    X = np.delete(X, delete_list, 0)
    y = np.delete(y, delete_list, 0)

    #X = np.resize(X,(len(X),22,1,1125))
    X = data_in_one(np.array(X))
    y = np.array(y)
    print("X:",X.shape)
    print("y:", y.shape)
    dataset = EEGDataset(X, y)
    return dataset

def import_EEGData_test(start=0, end=9, dir = '../data/A0'):
    electrodes = 22
    X, y = [], []

    for i in range(start, end):
        dataFile = loadmat(dir + str(i + 1) + 'E_slice.mat')
        labeldataFile = loadmat(dir + str(i + 1) + 'E.mat')
        X1 = np.copy(dataFile['image'])  # (1125,22,288)
        X1 = X1[:, :electrodes, np.newaxis, :]  # (1125,22,1,288)
        X1 = np.swapaxes(X1, 0, 3)  # (288,22,1,1125)
        y1 = np.copy(labeldataFile["classlabel"][: 288])
        y1 = y1 - np.ones(y1.shape)

        X.extend(X1)
        y.extend(y1)


    delete_list = []
    for i in range(len(X)):
        if np.any(np.isnan(X[i])):
            delete_list.append(i)

    X = np.delete(X, delete_list, 0)
    y = np.delete(y, delete_list, 0)

    #X = np.resize(X,(len(X),22,1,1125))

    X = data_in_one(np.array(X))
    y = np.array(y)
    print("test_set_X:",X.shape)
    print("test_set_y:", y.shape)
    dataset = EEGDataset(X, y)
    return dataset

if __name__ == '__main__':
    print("****** Testing Block: DataLoading ******")
    dataset = import_EEGData_test(0, 1, '../data/A0')
    print(dataset)
    print('dataset大小为：', dataset.__len__())
    data1_data, data1_label = dataset.__getitem__(0)
    print('数据和标签的尺寸为：', data1_data.shape, data1_label.shape)

    print(dataset.__getitem__(0))
    print(dataset[0])

    dataloader = DataLoader.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)






