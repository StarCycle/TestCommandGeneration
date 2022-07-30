import json
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing

class GatedGraphConv(MessagePassing):
    
    def __init__(self, channel_num, num_layers, aggr = 'add', bias = True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)
        self.channel_num = channel_num
        self.num_layers = num_layers
        self.weight = nn.Parameter(Tensor(num_layers, channel_num, channel_num))
        self.rnn = torch.nn.GRUCell(channel_num, channel_num, bias=bias)
        # self.reset_parameters()

    def reset_parameters(self):
        uniform(self.channel_num, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
          m = torch.matmul(x, self.weight[i])
          m = self.propagate(edge_index, x=m, edge_weight=None, size=None)
          x = self.rnn(m, x)
        return x

    def message(self, x_j):
        return x_j

class GNNModel(torch.nn.Module):
    def __init__(self, nodes, edges, codeCount2Label, cmdLength, covLength):
        super(GNNModel, self).__init__()       
        self.nodes = nodes
        self.edges = edges
        self.count2label = codeCount2Label
        nodeNum = nodes.size()[0]
        nodeChannels = nodes.size()[1] + 1 # Add Code cov info 
        self.graphConv = GatedGraphConv(nodeChannels, 3)
        self.mlp = nn.Sequential(
                      nn.Linear(nodeNum*nodeChannels + cmdLength, 1024),
                      nn.Tanh(),
                      nn.Linear(1024, covLength))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.5)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, cmds, cov):
        if len(cmds.shape) == 1:
            batchSize = 1
        else:
            batchSize = cmds.shape[0]
        device=cmds.device
        nodesInput = torch.tensor([]).to(device)
        edgesInput = torch.tensor([], dtype=torch.long).to(device)
        for i in range(batchSize):
            covInGraph = torch.zeros(self.nodes.shape[0]).to(device) 
            covInGraph[self.count2label] = cov[i] 
            covInGraph = covInGraph.reshape((-1, 1))
            temp = torch.cat((self.nodes, covInGraph), 1)
            nodesInput = torch.cat((nodesInput, temp), 0)
            temp = self.edges + i*self.nodes.shape[0]
            edgesInput = torch.cat((edgesInput, temp), 1)
        x = self.graphConv(nodesInput, edgesInput)
        x = x.reshape((batchSize, -1))
        x = torch.cat((x, cmds), 1)
        x = self.mlp(x)
        return x
