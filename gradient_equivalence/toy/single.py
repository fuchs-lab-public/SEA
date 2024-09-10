from PIL import Image
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import time
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--ktiles', type=int)
parser.add_argument('--ntiles', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--workers', type=int)

class dataset(data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        return img, index

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.tile = nn.Sequential(
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.ReLU()
        )
        self.slide = nn.Sequential(
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,2),
            nn.ReLU()
        )
    
    def forward(self, x):
        f = self.tile(x)
        x = self.slide(f)
        x = x.mean(0, keepdims=True)
        return f, x

def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = torch.rand(args.ntiles, 2048)
    
    dset = dataset(data)
    loader = torch.utils.data.DataLoader(dset,
                                         batch_size=args.ktiles,
                                         shuffle=False,
                                         num_workers=args.workers)
    
    net = Net()
    net.to('cuda')
    
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
    for epoch in range(args.epochs):
        
        running_loss = 0.0
        for i, (inputs, index) in enumerate(loader):
            torch.cuda.reset_peak_memory_stats()
            
            inputs = inputs.to('cuda')
            label = torch.LongTensor([1])
            label = label.to('cuda')
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward encoder
            features, outputs = net(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            obj = {
                'rank': 0,
                'index': index,
                'inputs': inputs.cpu(),
                'feats': features.detach().cpu().clone(),
                'w1': net.tile[0].weight.detach().cpu().clone(),
                'grad1': net.tile[0].weight.grad.detach().cpu().clone(),
                'w2': net.slide[2].weight.detach().cpu().clone(),
                'grad2': net.slide[2].weight.grad.detach().cpu().clone(),
                'out': outputs.detach().cpu().clone(),
                'loss': loss.item()
            }
            torch.save(obj, f'single_grads.pth')
            
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f'{epoch+1}/{args.epochs} - [{i+1}/{len(loader)}] - Mem: {mem:.2f}GB - loss: {loss.item():.3f}')
            
            raise Exception('Done')

if __name__ == '__main__':
    main()
