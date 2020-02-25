from datasets.APaugmentation import APaugmentation
import torch
import datasets.DataLoading_HGD as DL
import numpy as np

if __name__ == '__main__':
    print("****** Now start running AP-augmentation_HGD ******")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))

    train_dataset = DL.import_EEGData_train(1, 2)
    train_data, train_label = [], []
    for i in range(len(train_dataset)):
        data1_data, data1_label = train_dataset.__getitem__(i)
        train_data.append(np.array(data1_data))
        train_label.append(np.array(data1_label))
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    # 数据随机分成测试集和训练集
    slice_num = int(len(train_data) * 2 / 10)
    slice_interval = int(len(train_data) / slice_num)
    count = 0
    slice_point = []
    val_data = []
    val_label = []
    for i in range(slice_num):
        slice_point.append(count)
        val_data.append(train_data[count])
        val_label.append(train_label[count])
        count += slice_interval
    val_data = np.array(val_data)
    val_label = np.array(val_label)
    train_data = np.delete(train_data, slice_point, axis=0)
    train_label = np.delete(train_label, slice_point, axis=0)
    print("shape of val_set:", val_data.shape, val_label.shape)
    print("shape of train_set:", train_data.shape, train_label.shape)


    # 数据增强
    data_aug = np.copy(train_data)
    print("shape of data_aug:", data_aug.shape)
    print("shape of data:", train_data.shape)
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

    print("shape of data:", train_data.shape, "shape od label:", train_label.shape)

    np.save("../processed_data/train_data_HGD_5.npy", train_data)
    np.save("../processed_data/train_label_HGD_5.npy", train_label)
    np.save("../processed_data/val_data_HGD_5.npy", val_data)
    np.save("../processed_data/val_label_HGD_5.npy", val_label)
    print('Data has been saved!')
    # A = np.load("../processed_data/train_data.npy")
