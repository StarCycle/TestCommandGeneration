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
    def __init__(self, env, graphFile, gnn_layers, device):
        super(GNN_Agent, self).__init__()
        with open(graphFile, 'r') as f:
            graph = json.load(f)
        self.nodes = torch.tensor(graph['features'], dtype=torch.float32).to(device)
        self.edges = torch.tensor(graph['edges'], dtype=torch.long).T.to(device)
        self.count2label = torch.tensor(graph['count2label'], dtype=torch.long)
        nodeChannels = self.nodes.shape[1] + 1 # Add Code cov info 			
        self.graphConv = GatedGraphConv(nodeChannels, gnn_layers)
        self.atten_i = nn.Linear(2*nodeChannels, 2*nodeChannels)
        self.atten_j = nn.Linear(2*nodeChannels, 2*nodeChannels)
        self.graphmlp = nn.Sequential(
            layer_init(nn.Linear(2*nodeChannels, 256), 0.01),
            nn.ReLU(),
        )
        self.action_space = env.action_space
        self.critic = layer_init(nn.Linear(256, 1), 1)
        self.actor = layer_init(nn.Linear(256, sum(self.action_space)), 0.01)

    def attention(self, x, nodesInput):
        x = torch.cat((x, nodesInput), 1)
        x = torch.mul(torch.sigmoid(self.atten_i(x)), torch.relu(self.atten_j(x)))
        return x

    def GNN(self, cov, batchNum, batch):
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
        x = self.attention(x, nodesInput)
        x = global_add_pool(x, batch=batch)
        x = torch.relu(x)
        return x
		
    def get_value(self, x):
        if len(x.shape) == 1:
            batchNum = 1
            batch = [0]*self.nodes.shape[0]
            x = x.reshape((1, -1))
        else:
            batchNum = x.shape[0]
            batch = []
            for i in range(batchNum):
                batch = batch + [i]*self.nodes.shape[0]
        batch = torch.tensor(batch).to(x.device)
        graphEmb = self.GNN(x, batchNum, batch)
        state = self.graphmlp(graphEmb)
        return self.critic(state)

    def get_action_and_value(self, x, action=None):
        if len(x.shape) == 1:
            batchNum = 1
            batch = [0]*self.nodes.shape[0]
            x = x.reshape((1, -1))
        else:
            batchNum = x.shape[0]
            batch = []
            for i in range(batchNum):
                batch = batch + [i]*self.nodes.shape[0]
        batch = torch.tensor(batch).to(x.device)
        graphEmb = self.GNN(x, batchNum, batch)
        state = self.graphmlp(graphEmb)
        logits = self.actor(state)
        split_logits = torch.split(logits, self.action_space, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.critic(state)