import torch
import numpy as np
from scipy.io import loadmat
from braindecode.datasets.bbci import BBCIDataset
from collections import OrderedDict
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize
from datasets.DataLoading import EEGDataset, data_in_one


def import_EEGData_train(start=0, end=9, dir = '../data_HGD/train/'):
    X, y = [], []
    for i in range(start, end):
        dataFile = str(dir + str(i + 1) + '.mat')
        print("File:", dataFile, " loading...")
        cnt = BBCIDataset(filename=dataFile, load_sensor_names=None).load()
        marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                                  ('Rest', [3]), ('Feet', [4])])
        clean_ival = [0, 4000]

        set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def, clean_ival)
        clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

        C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                     'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                     'C6',
                     'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                     'FCC5h',
                     'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                     'CPP5h',
                     'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                     'CCP1h',
                     'CCP2h', 'CPP1h', 'CPP2h']

        cnt = cnt.pick_channels(C_sensors)
        cnt = resample_cnt(cnt, 250.0)
        cnt = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                      init_block_size=1000,
                                                      eps=1e-4).T,
            cnt)
        ival = [-500, 4000]

        dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
        dataset.X = dataset.X[clean_trial_mask]
        dataset.X = dataset.X[:, :, np.newaxis, :]
        dataset.y = dataset.y[clean_trial_mask]
        dataset.y = dataset.y[:, np.newaxis]

        X.extend(dataset.X)
        y.extend(dataset.y)

    X = data_in_one(np.array(X))
    y = np.array(y)
    print("X:", X.shape)
    print("y:", y.shape)
    dataset = EEGDataset(X, y)
    return dataset

def import_EEGData_test(start=0, end=9, dir = '../data_HGD/test/'):
    X, y = [], []
    for i in range(start, end):
        dataFile = str(dir + str(i + 1) + '.mat')
        print("File:", dataFile, " loading...")
        cnt = BBCIDataset(filename=dataFile, load_sensor_names=None).load()
        marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                                  ('Rest', [3]), ('Feet', [4])])
        clean_ival = [0, 4000]

        set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def, clean_ival)
        clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

        C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                     'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                     'C6',
                     'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                     'FCC5h',
                     'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                     'CPP5h',
                     'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                     'CCP1h',
                     'CCP2h', 'CPP1h', 'CPP2h']

        cnt = cnt.pick_channels(C_sensors)
        cnt = resample_cnt(cnt, 250.0)
        cnt = mne_apply(
            lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                      init_block_size=1000,
                                                      eps=1e-4).T,
            cnt)
        ival = [-500, 4000]

        dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
        dataset.X = dataset.X[clean_trial_mask]
        dataset.X = dataset.X[:, :, np.newaxis, :]
        dataset.y = dataset.y[clean_trial_mask]
        dataset.y = dataset.y[:, np.newaxis]

        X.extend(dataset.X)
        y.extend(dataset.y)

    X = data_in_one(np.array(X))
    y = np.array(y)
    print("X:", X.shape)
    print("y:", y.shape)
    dataset = EEGDataset(X, y)
    return dataset
if __name__ == '__main__':
    print("****** Testing Block: DataLoading_HGD ******")
    dataset = import_EEGData_test(0,1)
    print(dataset)
    print('dataset大小为：', dataset.__len__())
    data1_data, data1_label = dataset.__getitem__(0)
    print('数据和标签的尺寸为：', data1_data.shape, data1_label.shape)

    print(dataset.__getitem__(0))
    print(dataset[0])