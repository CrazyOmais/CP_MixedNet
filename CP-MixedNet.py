import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tensorboardX import SummaryWriter
import torchvision
import numpy as np
import torch.utils.data.dataloader as DataLoader
import datasets.DataLoading as DL
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, eps=1e-3, affine=affine),
        )

    def forward(self, x):
        return self.op(x)



class CP_MixedNet(nn.Module):
    def __init__(self):
        super(CP_MixedNet, self).__init__()

        # ***CP-Spatio-Temporal Block***
        # Channel Projection
        self.channelProj = nn.Conv2d(22, 35, 1, stride=1, bias=False)   # (22*1*1125)->(35*1*1125)
        self.batchnorm_proj_tranf = nn.BatchNorm2d(35)
        # Shape Transformation
        self.shapeTrans = nn.Conv2d(35, 35, 1, stride=1, bias=False)    # (35*1*1125)->(35*1*1125)
        # Temporal Convolution
        self.drop1 = nn.Dropout2d(p=0.5)
        self.conv1 = nn.Conv2d(1, 25, (1,11), stride=1, bias=False)     # (1*35*1125)->(25*35*1115)
        self.batchnorm1 = nn.BatchNorm2d(25, False)
        # Spatial Convolution
        self.drop2 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(25, 25, (35,1), stride=1, bias=False)    # (25*35*1115)->(25*1*1115)
        self.batchnorm2 = nn.BatchNorm2d(25, False)
        # Max Pooling
        self.maxPool1 = nn.MaxPool2d((1,3), stride=3, padding=0)    # (25*1*1115)->(25*1*371)

        # ***MS-Conv Block***
        # unDilated Convolution
        self.drop3 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(25, 100, 1, stride=1, bias=False)        # (25*1*371)->(100*1*371)
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(100, 100, (1, 11), stride=1, padding=(0,5), bias=False)  # (100*1*371)->(100*1*371)
        self.batchnorm4 = nn.BatchNorm2d(100)
        # Dilated Convolution
        self.dropDil = nn.Dropout2d(p=0.5)
        self.dilatedconv = nn.Conv2d(100, 100, (1,11), stride=1, padding=(0,10), dilation=2, bias=False)    # (100*1*371)->(100*1*371)
        self.batchnormDil = nn.BatchNorm2d(100)
        # Max pooling after Concatenating
        self.batchnorm_cancat = nn.BatchNorm2d(225)
        self.poolConcatenated = nn.MaxPool2d((1,3), stride=3, padding=0)    # (225*1*371)->(225*1*123)


        # ***Classification Block***
        self.drop5 = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2d(225, 225, (1,11), stride=1)  # (225*1*123)->(225*1*113)
        self.batchnorm5 = nn.BatchNorm2d(225)
        self.maxPool2 = nn.MaxPool2d((1,3), stride=3, padding=0)    # (225*1*113)->(225*1*37)
        self.fc = nn.Linear(8325, 4, bias=False)    # (1*8325)->(1*4)
        # self.softmax = nn.Softmax(dim=1)
        self.batchnorm6 = nn.BatchNorm1d(4)
        self.softmax = nn.Softmax(dim=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = F.elu(self.batchnorm_proj_tranf(self.channelProj(x)))
        #print('Channel Projection:',x.shape)
        x = F.elu(self.batchnorm_proj_tranf(self.shapeTrans(x)))
        #print('before Shape Transformation:',x.shape)
        x = torch.transpose(x, 1, 2)
        #print('after Shape Transformation:',x.shape)
        x = F.elu(self.batchnorm1(self.conv1(self.drop1(x))))
        #print('Temporal convolution:',x.shape)
        x = F.elu(self.batchnorm2(self.conv2(self.drop2(x))))
        #print('Spatial convolution:',x.shape)
        x = self.maxPool1(x)
        #print('Max pooling：',x.shape)

        x1 = F.elu(self.batchnorm3(self.conv3(self.drop3(x))))
        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(self.dropDil(x1))))
        #print('Dilated Convolution1:', x_dilated.shape)
        x_undilated = F.elu(self.batchnorm4(self.conv4(self.drop4(x1))))
        #print('Undilated Convolution2:', x_undilated.shape)

        x = torch.cat((x, x_dilated, x_undilated),dim=1)
        #print('Concatenated:', x.shape)
        x = self.poolConcatenated(self.batchnorm_cancat(x))
        #print('MixedScaleConv:', x.shape)

        x = F.elu(self.batchnorm5(self.conv5(self.drop5(x))))
        #print('Conv5:', x.shape)
        x = self.maxPool2(x)
        #print('maxPool2:', x.shape)
        x = x.view(-1, 8325)
        #print('beforeFC:', x.shape)
        x = F.relu(self.batchnorm6(self.fc(x)))
        #print('FC:', x.shape)
        #x = self.softmax(x)
        #print('softmax:', x.shape)
        return F.log_softmax(x, dim=-1)

def train(model, device, train_loader, optimizer, epoch, log_interval=100, ):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target.squeeze())

        #loss_fun = nn.CrossEntropyLoss()
        #loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        '''
        for i,(name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, 0)
                writer.add_scalar('loss', loss, i)
                loss = loss * 0.5
        '''
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss：{:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))
            '''
            writer.add_scalar(
                "Training loss",
                loss.item(),
                epoch * len(train_loader)
            )
            '''


    print("Trainning accuracy:", 100. * correct / len(train_loader.dataset))

def val(model, device, val_loader, optimizer):
    model.train()
    val_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        #loss = F.nll_loss(output, target.squeeze())

        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        val_loss += loss * batch_size
        pred = output.argmax(dim=1, keepdim=True)
        #correct = output.eq(target.view_as(pred)).sum().item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx == 0:
            print("pred:", output[0])
            print("true:", target[0])
    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            #loss_fun = nn.CrossEntropyLoss()
            #test_loss += loss_fun(output, target) * batch_size

            test_loss += F.nll_loss(output, target.squeeze(), reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))

if __name__ == "__main__":

    # Configs and Hyperparameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))
    batch_size = 128
    val_batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0.01

    #train_dataset = DL.import_EEGData(0, 1, 'data/A0')
    X = np.load("processed_data/train_data_BCI_1.npy")
    y = np.load("processed_data/train_label_BCI_1.npy")
    y = y[:, np.newaxis]
    print(X.shape, y.shape)
    train_dataset = DL.EEGDataset(X, y)

    X = np.load("processed_data/val_data_BCI_1.npy")
    y = np.load("processed_data/val_label_BCI_1.npy")
    y = y[:, np.newaxis]
    print("shape of val set:", X.shape, y.shape)
    val_dataset = DL.EEGDataset(X, y)

    X = np.load("processed_data/test_data_BCI_1.npy")
    y = np.load("processed_data/test_label_BCI_1.npy")
    y = y[:, np.newaxis]
    print("shape of test set:", X.shape, y.shape)
    test_dataset = DL.EEGDataset(X, y)


    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    model = CP_MixedNet().to(device)
    #writer = SummaryWriter('tensorboard_logs')



    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoches = 500
    for epoch in range(1, epoches + 1):
        train(model, device, train_loader, optimizer, epoch)
        val(model, device, val_loader, optimizer)
        test(model, device, test_loader)
    save_model = False
    if (save_model):
        torch.save(model.state_dict(), "weights.pt")




