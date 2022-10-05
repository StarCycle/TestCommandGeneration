import json
import torch
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing

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
    def __init__(self, env, graphFile, out_channels, gnn_layers, device):
        super(GNN_Agent, self).__init__()
        with open('graph.json', 'r') as f:
            graph = json.load(f)
        self.nodes = torch.tensor(graph['features'], dtype=torch.float32)
        self.edges = torch.tensor(graph['edges'], dtype=torch.long).T
        self.count2label = torch.tensor(graph['count2label'], dtype=torch.long)
        self.historyVecLen = len(env.actions)			
        self.graphConv = GatedGraphConv(out_channels, gnn_layers)
        self.nodemlp = nn.Sequential(
            layer_init(nn.Linear(out_channels, 1)),
            nn.ReLU(),
        )
        self.graphmlp = nn.Sequential(
            layer_init(nn.Linear(self.nodes.shape[0]+self.historyVecLen, 2048)),
            nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(2048, 1), 1)
        self.actor = layer_init(nn.Linear(2048, len(env.actions)), 0.01)
        
    def GNN(self, cov, batchNum):
        device = cov.device
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
        x = x.reshape((batchNum, -1))
        return x
		
    def get_value(self, x):
        if len(x.shape) == 1:
            batchNum = 1
            x = x.reshape((1, -1))
        else:
            batchNum = x.shape[0]
        cov = x[:, :len(self.count2label)]
        history = x[:, len(self.count2label):]
        nodeData = self.GNN(cov, batchNum)
        state = self.graphmlp(torch.cat((nodeData, history), 1))
        return self.critic(state)

    def get_action_and_value(self, x, action=None):
        if len(x.shape) == 1:
            batchNum = 1
            x = x.reshape((1, -1))
        else:
            batchNum = x.shape[0]
        cov = x[:, :len(self.count2label)]
        history = x[:, len(self.count2label):]
        nodeData = self.GNN(cov, batchNum)
        state = self.graphmlp(torch.cat((nodeData, history), 1))
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)