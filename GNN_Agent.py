import json
import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing

class GatedGraphConv(MessagePassing):
    
    def __init__(self, channel_num, num_layers, aggr = 'add', bias = True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)
        self.channel_num = channel_num
        self.num_layers = num_layers
        self.weight = nn.Parameter(Tensor(num_layers, channel_num, channel_num))
        self.rnn = torch.nn.GRUCell(channel_num, channel_num, bias=bias)
        self.reset_parameters()
		
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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GNN_Agent(torch.nn.Module):
    def __init__(self, env, graphFile):
        super(GNN_Agent, self).__init__()
        with open('graph.json', 'r') as f:
            graph = json.load(f)
        self.nodes = torch.tensor(graph['features'], dtype=torch.float32)
        self.edges = torch.tensor(graph['edges'], dtype=torch.long).T
        self.count2label = torch.tensor(graph['count2label'], dtype=torch.long)			
        nodeNum = self.nodes.shape[0]
        nodeChannels = self.nodes.shape[1] + 1 # Add Code cov info 
        self.graphConv = GatedGraphConv(nodeChannels, 3)
        self.critic = nn.Sequential(
                      layer_init(nn.Linear(631, 1024)),
                      nn.ReLU(),
                      layer_init(nn.Linear(1024, 1024)),
					  nn.ReLU(),
					  layer_init(nn.Linear(1024, 1), std=1))
        self.actor = nn.Sequential(
                     layer_init(nn.Linear(631, 1024)),
                     nn.ReLU(),
					 layer_init(nn.Linear(1024, 1024)),
					 nn.ReLU(),
                     layer_init(nn.Linear(1024, len(env.actions)), std=0.01))
        
    def GNN(self, cov):
        if len(cov.shape) == 1:
            batchSize = 1
            cov = cov.reshape((1, -1))
        else:
            batchSize = cov.shape[0]
        device = cov.device
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
        return x
		
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)