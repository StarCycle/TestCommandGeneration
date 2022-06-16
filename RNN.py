from random import randint, shuffle
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as weight_init
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

def analysis(epoch, train_outputs, train_outputs_hat, test_outputs, test_outputs_hat, loss, log_writer):

    train_true_positive = train_outputs[train_outputs_hat>0.5].sum().item()
    train_true_negative = (train_outputs[train_outputs_hat<0.5]==0).sum().item()
    train_fault_positive = (train_outputs[train_outputs_hat>0.5]==0).sum().item()
    train_fault_negative = train_outputs[train_outputs_hat<0.5].sum().item()
    train_acc = (train_true_positive + train_true_negative) / train_outputs.numel()
    train_prec = train_true_positive / (train_true_positive + train_fault_positive)
    train_recall = train_true_positive / (train_true_positive + train_fault_negative)
    train_f1 = 2*train_true_positive / (2*train_true_positive + train_fault_positive + train_fault_negative)
    train_ls = loss(train_outputs_hat, train_outputs).data.item()

    test_true_positive = test_outputs[test_outputs_hat>0.5].sum().item()
    test_true_negative = (test_outputs[test_outputs_hat<0.5]==0).sum().item()
    test_fault_positive = (test_outputs[test_outputs_hat>0.5]==0).sum().item()
    test_fault_negative = test_outputs[test_outputs_hat<0.5].sum().item()
    test_acc = (test_true_positive + test_true_negative) / test_outputs.numel()
    test_prec = test_true_positive / (test_true_positive + test_fault_positive)
    test_recall = test_true_positive / (test_true_positive + test_fault_negative)
    test_f1 = 2*test_true_positive / (2*test_true_positive + test_fault_positive + test_fault_negative)
    test_ls = loss(test_outputs_hat, test_outputs).data.item()

    '''
    log_writer.add_scalars('loss', {'train loss': train_ls, 'test loss': test_ls}, epoch)
    log_writer.add_scalars('accuracy', {'train acc': train_acc, 'test acc': test_acc}, epoch)
    log_writer.add_scalars('precison', {'train prec': train_prec, 'test prec': test_prec}, epoch)
    log_writer.add_scalars('recall', {'train recall': train_recall, 'test recall': test_recall}, epoch)
    log_writer.add_scalars('f1', {'train f1': train_f1, 'test f1': test_f1}, epoch)
    '''
    return train_ls, test_ls
    
def seq_data_iter(inputs, outputs, batch_size, random=True):
    num_steps = inputs.shape[1]
    # rearange the data according the offset
    offset = randint(0, num_steps)
    tranf_inputs = torch.zeros_like(inputs)
    tranf_outputs = torch.zeros_like(outputs)
    tranf_inputs[:-1, :offset, :] = inputs[1:, :offset, :] 
    tranf_outputs[:-1, :offset, :] = outputs[1:, :offset, :] 
    tranf_inputs = torch.cat((inputs[:, offset:, :], inputs[:, :offset, :]), 1)
    tranf_outputs = torch.cat((outputs[:, offset:, :], outputs[:, :offset, :]), 1)
    tranf_inputs = tranf_inputs[:-1, :, :]
    tranf_outputs = tranf_outputs[:-1, :, :]
    # Throw the data which cannot form a batch
    length = len(tranf_inputs)//batch_size*batch_size
    tranf_inputs = tranf_inputs[:length, :, :]
    tranf_outputs = tranf_outputs[:length, :, :]
    # Generate the indices for random sampling / sequencial sampling
    index = list(range(0, length))
    if random: 
        shuffle(index)
    else:
        index = torch.tensor(index)
        index = index.reshape((batch_size, -1))
        index = index.t()
        index = index.reshape(-1)
        index = index.tolist()
    for i in range(0, len(index), batch_size):
        yield tranf_inputs[index[i:i+batch_size]], tranf_outputs[index[i:i+batch_size]]

def train(net, loss, train_inputs, train_outputs, test_inputs, test_outputs, num_epochs, learning_rate, weight_decay, batch_size, patiance, random, device):
    log_writer = SummaryWriter()
    state = None
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in seq_data_iter(train_inputs, train_outputs, batch_size, random):
            if state is None or random:
                state = net.begin_state(batch_size, device)
            else: # for GRU, not LSTM
                state.detach_()
            y_hat, state = net(X, state)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        with torch.no_grad():
            state_for_check = net.begin_state(train_inputs.shape[0], device)
            train_outputs_hat, _ = net(train_inputs, state_for_check)
            state_for_check = net.begin_state(test_inputs.shape[0], device)
            test_outputs_hat, _ = net(test_inputs, state_for_check)
            cur_train_ls, cur_test_ls = analysis(epoch, train_outputs, train_outputs_hat, test_outputs, test_outputs_hat, loss, log_writer)
            for name, param in net.named_parameters():
                log_writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)
                log_writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch)
            print(f'epoch {epoch}: train loss {cur_train_ls}, test loss {cur_test_ls}')
        # Early stop
        if epoch % 20 == 0:
            if epoch == 0:
                best_test_ls = cur_test_ls
            if cur_test_ls < best_test_ls:
                best_test_ls = cur_test_ls
                torch.save(net.state_dict(), 'Model')
            # elif sum(test_ls[-patiance:])/min(len(test_ls), patiance) > best_test_ls:
                # break

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, num_outputs):
        super().__init__()
        self.rnn = rnn_layer
        self.num_outputs = num_outputs
        self.linear = nn.Linear(self.rnn.hidden_size, self.num_outputs)
        for name, param in self.named_parameters():
          if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=50)
          elif 'bias' in name:
            param.data.fill_(0)

    def forward(self, inputs, state):
        '''
        inputs  3D tensor, shape: (batch size, num_step, command vector per step) 
        outputs 3D tensor, shape: (batch size, num_step, coverage vector per step) 
        '''
        data = inputs.permute(1, 0, 2)                  # shape: (num_step, batch_size, command vector per step)
        data, state = self.rnn(data, state)             # shape: (num_step, batch_size, rnn_hidden_size)
        data = data.reshape((-1, self.rnn.hidden_size)) # shape: (num_step*batch_size, rnn_hidden_size)
        data = self.linear(data)                        # shape: (num_step*batch_size, num_outputs)
        data = data.reshape((inputs.shape[1], inputs.shape[0], self.num_outputs)) # shape: (num_step, batch_size, num_outputs)
        outputs = data.permute(1, 0, 2)                 # shape: (batch_size, num_step, num_outputs)
        return outputs, state

    def begin_state(self, batch_size, device):
        if isinstance(self.rnn, nn.GRU):
            return  torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), device=device)
