import torch
from torch import nn
from torch.utils import data
from RNN import RNNModel
from RNN import train
import json

def try_gpu():  
    """If there is a GPU, return gpu(0)，otherwise return cpu()"""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')

if __name__ == "__main__":

    with open('dataset.json', 'r') as f:
        embeddings = json.load(f)
    cmds = torch.tensor(embeddings['cmd'], device=try_gpu(), dtype=torch.float32)
    cov= torch.tensor(embeddings['cov'], device=try_gpu(), dtype=torch.float32)
    inputs = torch.cat((cov[0:-1], cmds[1:]), 1) # Sequential input: (last cov, current cmd)
    outputs = cov[1:]  # current cov

    # Preprocessing
    inputs -= torch.mean(inputs, axis=0)
    inputs /= torch.std(inputs, axis=0)
    inputs = torch.nan_to_num(inputs, nan=0)

    # Network
    loss = nn.MSELoss()
    num_inputs = inputs.shape[1]
    num_outputs = outputs.shape[1]
    num_rnn_hiddens = 256
    gru_layer = nn.GRU(num_inputs, num_rnn_hiddens, 1)
    net = RNNModel(gru_layer, num_outputs)
    net = net.to(device=try_gpu())

    # Train
    num_epochs, lr, weight_decay, batch_size, num_step, patience, split = 1000, 1, 0, 64, 10, 5, 0.8
    inputs = inputs.reshape((-1, num_step, inputs.shape[-1]))
    outputs = outputs.reshape((-1, num_step, outputs.shape[-1]))
    train_inputs = inputs[:int(len(inputs)*split)]
    train_outputs = outputs[:int(len(outputs)*split)]
    test_inputs = inputs[int(len(inputs)*split):]
    test_outputs = outputs[int(len(outputs)*split):]
    train(net, loss, train_inputs, train_outputs, test_inputs, test_outputs, num_epochs, lr, weight_decay, batch_size, patience, random=False, device=try_gpu())

    print('here')
