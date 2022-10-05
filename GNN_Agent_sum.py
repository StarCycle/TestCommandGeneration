import json
import torch
import numpy as np
from torch import nn
from torch import Tensor 
from torch.distributions.categorical import Categorical
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GatedGraphConv(MessagePassing):
    
    def __init__(self, out_channels, num_layers, aggr = 'add', bias = True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.weight = nn.Parameter(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()
		
    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index):
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')
        if x.size(-1) < self.out_channels:
            fill = x[:,-1].repeat(self.out_channels-x.shape[-1], 1)
            x = torch.cat([x, fill.T], dim=1)
        for i in range(self.num_layers):
          m = torch.matmul(x, self.weight[i])
          m = self.propagate(edge_index, x=m, edge_weight=None, size=None)
          x = self.rnn(m, x)
        return x

    def message(self, x_j):
        return x_j

class GNN_Agent(torch.nn.Module):
    def __init__(self, env, out_channels, gnn_layers, device):
        super(GNN_Agent, self).__init__()
        with open('graph_len127.json', 'r') as f:
            graph = json.load(f)
        self.nodes = torch.tensor(graph['features'], dtype=torch.float32)
        self.edges = torch.tensor(graph['edges'], dtype=torch.long).T
        self.count2label = torch.tensor(graph['count2label'], dtype=torch.long)
        self.historyVecLen = len(env.actions)			
        nodeChannels = self.nodes.shape[1] + 1 # Add Code cov info 
        self.graphConv = GatedGraphConv(out_channels, gnn_layers)
        self.atten_i = nn.Linear(out_channels+nodeChannels, 256)
        self.atten_j = nn.Linear(out_channels+nodeChannels, 256)
        self.nodemlp = nn.Sequential(
            layer_init(nn.Linear(out_channels, out_channels)),
            nn.ReLU(),
        )
        self.value_net = layer_init(nn.Linear(256, 1), 0.01)
        self.advantage_net = layer_init(nn.Linear(256, len(env.actions)), 0.01)

    def attention(self, x, nodesInput):
        x = torch.cat((x, nodesInput), 1)
        x = torch.mul(torch.sigmoid(self.atten_i(x)), torch.relu(self.atten_j(x)))
        return x

    def GNN(self, cov, batch, batchNum):
        device = cov.device
        batch = torch.tensor(batch).to(device)
        nodesInput = torch.tensor([]).to(device)
        edgesInput = torch.tensor([], dtype=torch.long).to(device)
        for i in range(batchNum):
            covInGraph = torch.zeros(self.nodes.shape[0]).to(device) 
            covInGraph[self.count2label] = cov[i] 
            covInGraph = covInGraph.reshape((-1, 1))
            temp = torch.cat((self.nodes, covInGraph), 1)
            nodesInput = torch.cat((nodesInput, temp), 0)
            temp = self.edges + i*self.nodes.shape[0]
            edgesInput = torch.cat((edgesInput, temp), 1)
        x = self.graphConv(nodesInput, edgesInput) 
        x = self.nodemlp(x)
        x = self.attention(x, nodesInput)
        x = global_add_pool(x, batch=batch)
        x = torch.relu(x)
        return x
		
    def forward(self, input):
        if len(input.shape) == 1:
            batchNum = 1
            batch = [0]*self.nodes.shape[0]
            input = input.reshape((1, -1))
        else:
            batchNum = input.shape[0]
            batch = []
            for i in range(batchNum):
                batch = batch + [i]*self.nodes.shape[0]
        cov = input[:, :len(self.count2label)]
        history = input[:, len(self.count2label):]
        graphEmb = self.GNN(cov, batch, batchNum)
        value = self.value_net(graphEmb)
        advantage = self.advantage_net(graphEmb)
        out = value + advantage - torch.mean(advantage)
        return out