import os
import torch

ddp = []
for i in range(5):
    ddp.append(torch.load(f'parallel_grads_rank-{i}.pth', map_location='cpu'))

noddp = torch.load('single_grads.pth', map_location='cpu')

# Check Model
torch.equal(noddp['w1'], ddp[1]['w1'])
torch.equal(noddp['w2'], ddp[0]['w2'])

# Check Inputs
torch.equal(noddp['inputs'][0], ddp[1]['inputs'][0])
torch.equal(noddp['inputs'][1], ddp[2]['inputs'][0])
torch.equal(noddp['inputs'][2], ddp[3]['inputs'][0])
torch.equal(noddp['inputs'][3], ddp[4]['inputs'][0])
torch.equal(noddp['inputs'][4], ddp[1]['inputs'][1])
torch.equal(noddp['inputs'][5], ddp[2]['inputs'][1])
torch.equal(noddp['inputs'][6], ddp[3]['inputs'][1])
torch.equal(noddp['inputs'][7], ddp[4]['inputs'][1])

# Check features gather
torch.equal(
    torch.cat([
        ddp[1]['feats0'],
        ddp[2]['feats0'],
        ddp[3]['feats0'],
        ddp[4]['feats0']
    ]),
    ddp[0]['feats']
)

# Check Features
torch.equal(noddp['feats'][0], ddp[0]['feats'][0])
(noddp['feats'][0] - ddp[0]['feats'][0]).abs().mean()
torch.equal(noddp['feats'][1], ddp[0]['feats'][10])
(noddp['feats'][1] - ddp[0]['feats'][10]).abs().mean()

# Check output
torch.equal(noddp['out'], ddp[0]['out'])
(noddp['out'] - ddp[0]['out']).abs().mean()

# Check loss
abs(ddp[0]['loss'] - noddp['loss'])

# Check grads 2
torch.equal(ddp[0]['grad2'], noddp['grad2'])
(ddp[0]['grad2'] - noddp['grad2']).abs().mean()
ddp[0]['grad2']
noddp['grad2']

# Check grads 1
torch.equal(ddp[1]['grad1'], ddp[2]['grad1'])
torch.equal(ddp[1]['grad1'], ddp[3]['grad1'])
torch.equal(ddp[1]['grad1'], ddp[4]['grad1'])

torch.equal(ddp[1]['grad1'], noddp['grad1'])
(ddp[1]['grad1'] - noddp['grad1']).abs().mean()
# Gradients basically the same
