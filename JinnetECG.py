import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.serialization import *
import torch.utils.data as data_utils

learning_rate = 1e-3
batch_size = 10
epoches = 50

traindataname = '/Users/ping/Downloads/sample-data-model_2016_11_29/traindata.th7'
trainlabelname = '/Users/ping/Downloads/sample-data-model_2016_11_29/trainlabel.th7'

validdataname = '/Users/ping/Downloads/sample-data-model_2016_11_29/validdata.th7'
validlabelname = '/Users/ping/Downloads/sample-data-model_2016_11_29/validlabel.th7'

features = load_lua(traindataname) # nsample x dim 
targets = load_lua(trainlabelname) #1 x nsample

dim = features.size()
trsize = dim[0] 
trdim = dim[1]  

newfeat = torch.Tensor(trsize, 1, trdim)
labels = torch.LongTensor(trsize)
for i in range(trsize):
   newfeat[i][0] = features[i]
   labels[i] = long(targets[0][i])

features = newfeat
targests = labels

train = data_utils.TensorDataset(features, targets.transpose(0,1))
train_loader = data_utils.DataLoader(train, batch_size, shuffle=True)

features = load_lua(validdataname)
targets = load_lua(validlabelname)
dim = features.size()
trsize = dim[0] 
trdim = dim[1]  

newfeat = torch.Tensor(trsize, 1, trdim)
labels = torch.LongTensor(trsize)
for i in range(trsize):
        newfeat[i][0] = features[i]
        labels[i] = long(targets[0][i])

features = newfeat
targests = labels

test = data_utils.TensorDataset(features, targets.transpose(0,1))
test_loader = data_utils.DataLoader(train, batch_size, shuffle=True)

class Jinnet(nn.Module):
    def __init__(self):
        super(Jinnet, self).__init__()
	self.conv = nn.Sequential(
	    	nn.Conv1d(1,32,16,1),
		nn.ReLU(),
		nn.MaxPool1d(4),

		nn.Conv1d(32,64,8,1),
		nn.ReLU(),
		nn.MaxPool1d(4),

		nn.Conv1d(64,128,4,1),
		nn.ReLU(),
		nn.MaxPool1d(2),

		nn.Conv1d(128,256,4,1),
		nn.ReLU(),
		nn.MaxPool1d(2)
		)

        self.fc = nn.Sequential(
			nn.Linear(256*13, 1024),
			nn.ReLU(),
			nn.Linear(1024, 20)
                        #nn.LogSoftmax()
			)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 256*13)
        out = self.fc(out)
        out = nn.functional.log_softmax(out)

        return out

jinnet = Jinnet()

optimizer = optim.SGD(jinnet.parameters(), lr=learning_rate)
log_interval = 10
# train

def train(epoches):
    jinnet.train()
    train_loss = 0
    correct = 0
    for batch_idx, (sample, label) in enumerate(train_loader):
        feat = Variable(sample)
        label = Variable(label)
        
        label = label.type(torch.LongTensor)
        label = label.squeeze() - 1
	
        optimizer.zero_grad()
        output = jinnet(feat)
        
        loss = nn.functional.nll_loss(output, label)
        print loss.data[0]
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()

    train_loss = loss
    train_loss /= len(train_loader)
    print 'train set: average loss is :'
    print train_loss
    print '\n'
    
    print 'accuracy is :'
    print 100.*correct/(trsize)

def test(epoches):
    jinnet.eval()
    test_loss = 0
    correct = 0
    for feat, label in test_loader:
        feat, label = Variable(feat, volatile = True), Variable(label)
        
        label = label.type(torch.LongTensor)
        label = label.squeeze() - 1

        output = jinnet(feat)
        test_loss +=  nn.functional.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        corret += pred.eq(label.data).cpu().sum()
 
    test_loss = test_loss
    test_loss /= len(test_loader)
    print 'test set: average loss is :'
    print test_loss
    print '\n'

    print 'accuracy is :'
    print 100.*correct/(trsize)

for epoch in range(1, epoches+1):
    train(epoch)
    test(epoch)
