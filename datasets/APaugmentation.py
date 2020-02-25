import torch
from torch_stft import STFT
import datasets.DataLoading as DL
import h5py
import numpy as np
from scipy.io import loadmat
import librosa

def APaugmentation(image, device = 'cpu'):
    # filter尺寸
    filter_length = 250
    # STFT的扫描步长，默认限制为帧之间的50%重叠
    hop_length = 125
    # 每一帧的窗口函数长度
    win_length = 250
    window = 'hann'

    image = torch.FloatTensor(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    stft = STFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    ).to(device)

    sigma = 0.001
    magnitude, phase = stft.transform(image)
    magnitude += torch.randn(magnitude.shape).to(device) * sigma

    augmentedImg = stft.inverse(magnitude, phase)
    return augmentedImg

if __name__ == '__main__':
    print("****** Now start testing AP-augmentation ******")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))

    train_dataset = DL.import_EEGData_train(7, 8, '../data/A0')
    train_data, train_label = [], []
    for i in range(len(train_dataset)):
        data1_data, data1_label = train_dataset.__getitem__(i)
        train_data.append(np.array(data1_data))
        train_label.append(np.array(data1_label))
    train_data = np.array(train_data)
    train_label = np.array(train_label)


    # 取数据6/10-8/10做验证集
    #val_data = train_data[int(len(train_data)*6/10): int(len(train_data)*8/10)]
    #val_label = train_label[int(len(train_label)*6/10): int(len(train_label)*8/10)]
    #print("shape of val set:", val_data.shape, val_label.shape)
    # 取数据后2/10做测试集
    #test_data = train_data[int(len(train_data) * 8 / 10):]
    #test_label = train_label[int(len(train_label) * 8 / 10):]
    #print("shape of test set:", test_data.shape, test_label.shape)
    # 取数据前6/10做数据增强然后做训练集
    #train_data = train_data[:int(len(train_data)*6/10)]
    #train_label = train_label[:int(len(train_label)*6/10)]
    #print("shape of train set:", train_data.shape, train_label.shape)


    data_aug = np.copy(train_data)
    print("labels:", data1_label)
    count = 0
    for eachtest in range(len(data_aug)):
        for eachelectrod in range(len(data_aug[0])):
            for eachline in range(len(data_aug[0][0])):
                
                data_aug[eachtest][eachelectrod][eachline] = APaugmentation(
                    data_aug[eachtest][eachelectrod][eachline],
                    device).to("cpu")
                
                count += 1
                if count % 100 == 0:
                    print("Processing:{}%...".format(count * 100 / (len(data_aug) * len(data_aug[0]))))
    print('AP-Augmentation Finished!')
    
    train_data = np.concatenate((train_data, data_aug), axis=0)
    train_label = np.concatenate((train_label, train_label), axis=0)
    
    print("shape of data:", train_data.shape, "shape of label:", train_label.shape)

    count = 0
    # 数据随机分成测试集和训练集
    slice_num = int(len(train_data) * 3 / 11)
    slice_interval = int(len(train_data) / slice_num)
    count = 0
    slice_point = []
    test_data = []
    test_label = []
    for i in range(slice_num):
        slice_point.append(count)
        test_data.append(train_data[count])
        test_label.append(train_label[count])
        count += slice_interval
    val_data = np.array(test_data)
    val_label = np.array(test_label)
    train_data = np.delete(train_data, slice_point, axis=0)
    train_label = np.delete(train_label, slice_point, axis=0)

    np.save("../processed_data/train_data_BCI_8.npy", train_data)
    np.save("../processed_data/train_label_BCI_8.npy", train_label)
    np.save("../processed_data/val_data_BCI_8.npy", val_data)
    np.save("../processed_data/val_label_BCI_8.npy", val_label)
    #np.save("../processed_data/test_data_BCI_1.npy", test_data)
    #np.save("../processed_data/test_label_BCI_1.npy", test_label)

    print('Data has been saved!')


    # A = np.load("../processed_data/train_data.npy")
