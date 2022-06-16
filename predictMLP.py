import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import json

def analysis(net, features, labels):
    true_positive = labels[net(features)>0.5].sum().item()
    true_negative = (labels[net(features)<0.5]==0).sum().item()
    fault_positive = (labels[net(features)>0.5]==0).sum().item()
    fault_negative = labels[net(features)<0.5].sum().item()
    accuracy = (true_positive + true_negative) / labels.numel()
    # Add 0.01 to prevent 0 denominator
    precision = (true_positive + 0.01) / (true_positive + fault_positive + 0.01)
    recall = (true_positive + 0.01) / (true_positive + fault_negative + 0.01)
    f1 = (2*true_positive + 0.01) / (2*true_positive + fault_positive + fault_negative + 0.01)
    return accuracy, precision, recall, f1

def train(net, loss, trainset, testset, num_epochs, learning_rate, weight_decay, batch_size, patiance):
    train_ls, test_ls = [], []
    log_writer = SummaryWriter()
    train_iter = data.DataLoader(trainset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_features, train_labels = trainset[:]
            test_features, test_labels = testset[:]
            cur_train_ls = loss(net(train_features), train_labels).data.item()
            cur_test_ls = loss(net(test_features), test_labels).data.item()
            cur_train_acc, cur_train_prec, cur_train_recall, cur_train_f1 = analysis(net, train_features, train_labels)
            cur_test_acc, cur_test_prec, cur_test_recall, cur_test_f1 = analysis(net, test_features, test_labels)
            train_ls.append(cur_train_ls)
            test_ls.append(cur_test_ls)
            '''
            log_writer.add_scalars('loss', {'train loss': cur_train_ls, 'test loss': cur_test_ls}, epoch)
            log_writer.add_scalars('accuracy', {'train acc': cur_train_acc, 'test acc': cur_test_acc}, epoch)
            log_writer.add_scalars('precison', {'train prec': cur_train_prec, 'test prec': cur_test_prec}, epoch)
            log_writer.add_scalars('recall', {'train recall': cur_train_recall, 'test recall': cur_test_recall}, epoch)
            log_writer.add_scalars('f1', {'train f1': cur_train_f1, 'test f1': cur_test_f1}, epoch)
            for name, param in net.named_parameters():
                log_writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)
                log_writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch)
            '''
            print(f'epoch {epoch}: train loss {cur_train_ls}, test loss {cur_test_ls}')
        # Early stop
        if epoch % 20 == 0:
            if epoch == 0:
                best_test_ls = cur_test_ls
            if cur_test_ls < best_test_ls:
                best_test_ls = cur_test_ls
                torch.save(net.state_dict(), 'Model')
            # elif sum(test_ls[-patiance:])/min(len(test_ls), patiance) > best_test_ls:
            #    break

def my_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def try_gpu():  
    """If there is a GPU, return gpu(0)，otherwise return cpu()"""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')

with open('dataset.json', 'r') as f:
    embeddings = json.load(f)
cmds = torch.tensor(embeddings['cmd'], device=try_gpu(), dtype=torch.float32)
cov= torch.tensor(embeddings['cov'], device=try_gpu(), dtype=torch.float32)

# Preprocess
cmds -= torch.mean(cmds, axis = 0)
cmds /= torch.std(cmds, axis = 0)
cmds = torch.nan_to_num(cmds, nan=0)

# Baseline Network
loss = nn.MSELoss()
in_features = cmds.shape[1]
out_features = cov.shape[1]
net = nn.Sequential(nn.Linear(in_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_features))
net.apply(my_init)
net = net.to(device=try_gpu())

# Train
num_epochs, lr, weight_decay, batch_size, patience, split = 300, 1, 1e-5, 64, 10, 0.8
dataset = data.TensorDataset(cmds, cov)
trainset_size = int(len(dataset)*0.8) 
trainset, testset = data.random_split(dataset, [trainset_size, int(len(dataset)) - trainset_size])
train(net, loss, trainset, testset, num_epochs, lr, weight_decay, batch_size, patience)

print('here')
