'''
All processes: 1+N. Master process handles the GMA, other N handle features
N processes wrapped in DDP with new group
Custom loss to flow gradients through DDP model
Networks are deterministic to check gradients
'''
from PIL import Image
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import numpy as np
import time
import pdb

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()

parser = argparse.ArgumentParser()
parser.add_argument('--ktiles', type=int)
parser.add_argument('--ntiles', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--workers', type=int)
parser.add_argument("--dist_url", default="env://", type=str)
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

class GradMultiplierFunction(torch.autograd.Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, constant):
        # The forward pass can use ctx.
        ctx.constant = constant
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.constant, None

class GradMultiplier(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant
    
    def forward(self, input):
        return GradMultiplierFunction.apply(input, self.constant)

class dataset(data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        return img, index

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.tile = nn.Sequential(
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.tile(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.slide = nn.Sequential(
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,2),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.slide(x)
        x = x.mean(0, keepdims=True)
        return x

#class MyDistributedBatchSampler(torch.utils.data.distributed.DistributedSampler):
class MyDistributedBatchSampler(object):
    '''
    Ensure that rank 1->N receive the right data splits
    '''
    def __init__(self, dataset, num_replicas=None, rank=None):
        #super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        '''
        num_replicas: processes involved in DDP, N
        rank 0: master process, does not receive data
        rank 1,N+1: receives data splits
        '''
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.rank == 0:
            indices = indices[self.rank:self.total_size:self.num_replicas]
        else:
            indices = indices[self.rank-1:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples

def pseudo_loss(f, fgrad):
    return (f * fgrad).sum()

def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    #data = torch.ones((args.ntiles,3,224,224))*torch.arange(args.ntiles).view(-1,1,1,1)
    data = torch.rand(args.ntiles, 2048)
    
    init_distributed_mode(args)
    # DDP group all ranks except for rank 0
    ddp_group = dist.new_group(ranks=list(range(1,args.world_size)))
    print(args.rank, ddp_group)
    
    dset = dataset(data)
    sampler = MyDistributedBatchSampler(dset, rank=args.rank, num_replicas=args.world_size-1)
    loader = torch.utils.data.DataLoader(dset,
                                         sampler=sampler,
                                         batch_size=args.ktiles,
                                         num_workers=args.workers)
    
    net1 = Net1()
    net2 = Net2()
    if args.rank == 0:
        #net2 = Net2()
        net2.to(args.gpu)
        criterion = nn.CrossEntropyLoss().to(args.gpu)
        optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
    else:
        #net1 = Net1()
        net1.to(args.gpu)
        net1 = nn.parallel.DistributedDataParallel(net1, device_ids=[args.gpu], process_group=ddp_group)
        optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
    
    print(f'rank {args.rank} data, sampler, model initialized')
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, (inputs, index) in enumerate(loader):
            print(f'rank {args.rank} inside loop')
            # rank 0 receives no data
            # ranks 1,N+1 receive split data
            if args.rank == 0:
                torch.cuda.reset_peak_memory_stats()
            
            if args.rank != 0:
                inputs = inputs.to(args.gpu)
                
                # zero the parameter gradients
                optimizer1.zero_grad()
                
                # forward encoder
                features = net1.forward(inputs)
            else:
                features = torch.empty((args.ktiles,2048)).to(args.gpu)
            
            print(f'rank {args.rank} features calculated', features)
            
            # Barrier on main group
            print(f'rank {args.rank} before barrier 1')
            dist.barrier(group=None)
            
            # Gather features
            with torch.no_grad():
                if args.rank == 0:
                    allfeatures = [torch.empty((args.ktiles,2048)).to(args.gpu) for _ in range(args.world_size)]
                    dist.gather(features, allfeatures, dst=0, group=None)
                    allfeatures = torch.cat(allfeatures[1:])
                else:
                    dist.gather(features, None, dst=0, group=None)
            
            print(f'rank {args.rank} after gather')
            
            # Forward pass on aggregator
            if args.rank == 0:
                # Record gradients on features
                allfeatures.requires_grad_()
                
                # Forward / backward for net2
                output = net2(allfeatures)
                label = torch.LongTensor([1])
                label = label.to(args.gpu)
                loss2 = criterion(output, label)
                loss2.backward()
                print(f'rank {args.rank} loss')
                
                # Get grads of features
                grads = allfeatures.grad.detach()
                
                # Split grads
                grads = torch.split(grads, args.ktiles)
                grads = [torch.empty((args.ktiles, 2048)).to(args.gpu)] + list(grads)
            
            else:
                grads = None
            
            # Barrier on main group
            print(f'rank {args.rank} before barrier 2')
            dist.barrier(group=None)
            
            print(f'rank {args.rank} grads calculated', grads)
            # Define output tensor
            grads_recv = torch.zeros((args.ktiles, 2048)).to(args.gpu)
            # Scatter features to other processes
            with torch.no_grad():
                dist.scatter(grads_recv, grads, src=0, group=None)
            
            print(f'rank {args.rank} after scatter', grads_recv)
            
            # Generate loss on ddp group
            # Loss has to be scaled by number of DDP processes
            if args.rank != 0:
                loss1 = pseudo_loss(features, grads_recv) * (args.world_size-1)
                loss1.backward()
            
            # Save some stuff
            if args.rank == 0:
                obj = {
                    'rank': args.rank,
                    'index': index,
                    'inputs': inputs.cpu(),
                    'feats0': features.detach().cpu().clone(),
                    'feats': allfeatures.detach().cpu().clone(),
                    #'grad1': net1.module.tile[0].weight.grad.detach().cpu().clone(),
                    'w2': net2.slide[2].weight.detach().cpu().clone(),
                    'grad2': net2.slide[2].weight.grad.detach().cpu().clone(),
                    'out': output.detach().cpu().clone(),
                    'loss': loss2.item()
                }
            else:
                obj = {
                    'rank': args.rank,
                    'index': index,
                    'inputs': inputs.cpu(),
                    'feats0': features.detach().cpu().clone(),
                    #'feats': allfeatures.detach().cpu().clone(),
                    'w1': net1.module.tile[0].weight.detach().cpu().clone(),
                    'grad1': net1.module.tile[0].weight.grad.detach().cpu().clone(),
                    #'grad2': net2.slide[2].weight.grad.detach().cpu().clone(),
                    #'out': outputs.detach().cpu().clone(),
                    'loss': loss1.item()
                }
            torch.save(obj, f'parallel_grads_rank-{args.rank}.pth')
            
            if args.rank == 0:
                optimizer2.step()
            else:
                optimizer1.step()
            
            # print statistics
            if args.rank == 0:
                running_loss += loss2.item()
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f'{epoch+1}/{args.epochs} - [{i+1}/{len(loader)}] - Mem: {mem:.2f}GB - loss: {loss2.item():.3f}')
            
            raise Exception('Done')

if __name__ == '__main__':
    main()
