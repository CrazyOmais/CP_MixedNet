from argparse import ArgumentParser
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader

import datasets.DataLoading as DL
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

from tqdm import tqdm
from tensorboardX import SummaryWriter

class CP_MixedNet(nn.Module):
    def __init__(self):
        super(CP_MixedNet, self).__init__()

        # ***CP-Spatio-Temporal Block***
        # Channel Projection
        self.drop_channelProj = nn.Dropout()
        self.channelProj = nn.Conv2d(22, 35, 1, stride=1, bias=True)   # (44*1*1125)->(35*1*1125)
        self.batchnorm_proj_transf = nn.BatchNorm2d(35, momentum=0.99)
        # Shape Transformation
        #self.drop0 = nn.Dropout2d()
        self.shapeTrans = nn.Conv2d(35, 35, 1, stride=1, bias=True)    # (35*1*1125)->(35*1*1125)
        self.drop_shapeTrans = nn.Dropout() #
        self.batchnorm_shape_transf = nn.BatchNorm2d(1, momentum=0.99)
        # Temporal Convolution
        self.drop1 = nn.Dropout() #
        self.conv1 = nn.Conv2d(1, 25, (1, 11), stride=1, bias=True)     # (1*35*1125)->(25*35*1115)
        self.batchnorm1 = nn.BatchNorm2d(25, momentum=0.99)
        # Spatial Convolution
        self.drop2 = nn.Dropout() #
        self.conv2 = nn.Conv2d(25, 25, (35, 1), stride=1, bias=True)    # (25*35*1115)->(25*1*1115)
        self.batchnorm2 = nn.BatchNorm2d(25, momentum=0.99)
        # Max Pooling
        self.maxPool1 = nn.MaxPool2d((1, 3), stride=3, padding=0)    # (25*1*1115)->(25*1*371)
        #self.batchnorm_raw = nn.BatchNorm2d(25, momentum=0.99)

        # ***MS-Conv Block***
        # unDilated Convolution
        self.drop3_dil = nn.Dropout()#
        self.drop3_undil = nn.Dropout()#
        self.conv3_dil = nn.Conv2d(25, 100, 1, stride=1, bias=True)        # (25*1*371)->(100*1*371)
        self.conv3_undil = nn.Conv2d(25, 100, 1, stride=1, bias=True)
        self.batchnorm3_dil = nn.BatchNorm2d(100, momentum=0.99)
        self.batchnorm3_undil = nn.BatchNorm2d(100, momentum=0.99)
        self.drop4 = nn.Dropout() #
        self.conv4 = nn.Conv2d(100, 100, (1, 11), stride=1, padding=(0, 5), bias=True)  # (100*1*371)->(100*1*371)
        self.batchnorm4 = nn.BatchNorm2d(100, momentum=0.99)
        # Dilated Convolution
        self.dropDil = nn.Dropout()   #
        self.dilatedconv = nn.Conv2d(100, 100, (1, 11), stride=1, padding=(0, 10), dilation=2, bias=True)    # (100*1*371)->(100*1*371)
        self.batchnormDil = nn.BatchNorm2d(100, momentum=0.99)
        # Max pooling after Concatenating
        #self.batchnorm_cancat = nn.BatchNorm2d(225, momentum=0.99)
        self.poolConcatenated = nn.MaxPool2d((1,3), stride=3, padding=0)    # (225*1*371)->(225*1*123)
        #self.batchnorm_poolcan = nn.BatchNorm2d(225, momentum=0.99)

        # ***Classification Block***
        self.drop5 = nn.Dropout() #
        self.conv5 = nn.Conv2d(225, 225, (1, 11), stride=1, bias=True)  # (225*1*123)->(225*1*113)
        self.batchnorm5 = nn.BatchNorm2d(225, momentum=0.99)
        self.maxPool2 = nn.MaxPool2d((1, 3), stride=3, padding=0)    # (225*1*113)->(225*1*37)
        #self.batchnorm_pool2 = nn.BatchNorm2d(225, momentum=0.99)
        self.fcdrop = nn.Dropout()
        self.fc = nn.Linear(8325, 4, bias=True)    # (1*8325)->(1*4)
        self.batchnorm6 = nn.BatchNorm1d(4)

        '''
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''

    def forward(self, x):
        # ***CP-Spatio-Temporal Block***
        #x = F.elu(self.batchnorm_proj_transf(self.channelProj(self.drop_channelProj(x))))
        x = F.elu(self.batchnorm_proj_transf(self.channelProj(x)))
        #x = self.shapeTrans(self.drop_shapeTrans(x))
        x = self.shapeTrans(x)
        x = torch.transpose(x, 1, 2)
        x = F.elu(self.batchnorm_shape_transf(x))
        x = F.elu(self.batchnorm1(self.conv1(self.drop1(x))))
        x = F.elu(self.batchnorm2(self.conv2(self.drop2(x))))
        #x = F.elu(self.batchnorm1(self.conv1(x)))
        #x = F.elu(self.batchnorm2(self.conv2(x)))
        x = self.drop3_dil(self.maxPool1(x))
        #x = self.maxPool1(x)

        # ***MS-Conv Block***
        x1 = x
        x2 = x

        #x1 = self.drop3_dil(x1)
        x1 = F.elu(self.batchnorm3_dil(self.conv3_dil(x1)))
        #x2 = self.drop3_undil(x2)
        x2 = F.elu(self.batchnorm3_undil(self.conv3_undil(x2)))

        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(self.dropDil(x1))))
        x_undilated = F.elu(self.batchnorm4(self.conv4(self.drop4(x2))))
        #x_dilated = F.elu(self.batchnormDil(self.dilatedconv(x1)))
        #x_undilated = F.elu(self.batchnorm4(self.conv4(x2)))
        x = torch.cat((x, x_dilated, x_undilated), dim=1)

        x = self.poolConcatenated(x)

        # ***Classification Block***
        x = F.elu(self.batchnorm5(self.conv5(self.drop5(x))))
        #x = F.elu(self.batchnorm5(self.conv5(x)))
        x = self.maxPool2(x)
        x = x.view(-1, 8325)
        x = F.elu(self.batchnorm6(self.fc(self.fcdrop(x))))

        return F.log_softmax(x, dim=-1)

def get_data_loaders(train_batch_size, val_batch_size):
    #train_dataset = DL.import_EEGData_train(0, 1, 'data/A0')
    X = np.load("processed_data/train_data_BCI_7.npy")
    y = np.load("processed_data/train_label_BCI_7.npy")
    y = y[:, np.newaxis]
    print("shape of train set:", X.shape, y.shape)
    train_dataset = DL.EEGDataset(X, y)


    X = np.load("processed_data/val_data_BCI_7.npy")
    y = np.load("processed_data/val_label_BCI_7.npy")
    y = y[:, np.newaxis]
    print("shape of val set:", X.shape, y.shape)
    val_dataset = DL.EEGDataset(X, y)



    train_loader = DataLoader.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader

def get_test_loader(batch_size):
    test_dataset = DL.import_EEGData_test(6, 7, 'data/A0')
    #X = np.load("processed_data/test_data_BCI_1.npy")
    #y = np.load("processed_data/test_label_BCI_1.npy")
    #y = y[:, np.newaxis]
    #print("shape of test set:", X.shape, y.shape)
    #test_dataset = DL.EEGDataset(X, y)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def score_function(engine):
    val_loss = engine.state.metrics['nll']
    return -val_loss

def run(train_batch_size, val_batch_size, epochs, learning_rate, weight_decay, log_interval, log_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    test_loader = get_test_loader(val_batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))

    model = CP_MixedNet()
    writer = create_summary_writer(model, train_loader, log_dir)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)
    evaluator_val = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0
        writer.add_scalar("test/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("test/avg_accuracy", avg_accuracy, engine.state.epoch)

    handler = EarlyStopping(patience=400, score_function=score_function, trainer=trainer)
    evaluator_val.add_event_handler(Events.COMPLETED, handler)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        
        evaluator_val.run(val_loader)
        metrics = evaluator_val.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll)
        )
        pbar.n = pbar.last_print_n = 0
        writer.add_scalar("val/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("val/avg_accuracy", avg_accuracy, engine.state.epoch)



    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
    writer.close()

    save_model = True
    if (save_model):
        torch.save(model.state_dict(), "weights_BCI.pt")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=210,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 64)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='decay rate during Adam optimizing (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    run(args.train_batch_size, args.val_batch_size, args.epochs, args.learning_rate, args.weight_decay, args.log_interval, args.log_dir )