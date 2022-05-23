import torch
from torch import nn
from torch.utils import data
import json

def load_array(data_arrays, batch_size, is_train=True):
    '''Consruct a PyTorch Dataloader'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def accuracy(net, features, labels):
    true_positive = labels[net(features)>0.5].sum().item()
    true_negative = (labels[net(features)<0.5]==0).sum().item()
    return (true_positive + true_negative) / labels.numel()

def train(net, loss, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size, patiance):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        if epoch % 50 == 0:
            with torch.no_grad():
                current_train_ls = loss(net(train_features), train_labels).data.item()
                current_test_ls = loss(net(test_features), test_labels).data.item()
                current_train_acc = accuracy(net, train_features, train_labels)
                current_test_acc = accuracy(net, test_features, test_labels)
                if epoch == 0:
                    best_test_ls = current_test_ls
                train_ls.append(current_train_ls)
                test_ls.append(current_test_ls)
                print(f'epoch {epoch}: train loss {current_train_ls}, test loss {current_test_ls}, train acc {current_train_acc}, test acc {current_test_acc}')
                if current_test_ls < best_test_ls:
                    best_test_ls = current_test_ls
                    torch.save(net.state_dict(), 'Model')
                # elif sum(test_ls[-patiance:])/min(len(test_ls), patiance) > best_test_ls:
                    # break # early stop
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    '''
    Input:
            k	total fold number
            i	Which fold is used as testation
            X 	Feature vectors. The first dimension is number of samples
            y 	label vectors.
    Output:
            Training set and testation set of fold i
    '''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_test, y_test = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_test, y_test

def k_fold(k, net, loss, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, patience):
    train_ls_sum, test_ls_sum = 0, 0
    for i in range(k):
        X_train, y_train, X_test, y_test = get_k_fold_data(k, i, X_train, y_train)
        net.apply(my_init)
        train_ls, test_ls = train(net, loss, X_train, y_train, X_test, y_test, num_epochs, learning_rate, weight_decay, batch_size, patience)
        train_ls_sum += train_ls[-1]
        test_ls_sum += test_ls[-1]
    # Return average loss on the training set and testing set
    return train_ls_sum / k, test_ls_sum / k

def my_init(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

def try_gpu():  
    """If there is a GPU, return gpu(0)，otherwise return cpu()"""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')

with open('dataset.json', 'r') as f:
    embeddings = json.load(f)
cmds = torch.tensor(embeddings['commands'], device=try_gpu(), dtype=torch.float32)
replies= torch.tensor(embeddings['replies'], device=try_gpu(), dtype=torch.float32)
cov= torch.tensor(embeddings['cov'], device=try_gpu(), dtype=torch.float32)

# Preprocess
cmds -= torch.mean(cmds, axis = 0)
cmds /= torch.std(cmds, axis = 0)
cmds = torch.nan_to_num(cmds, nan=0)

# Baseline Network
loss = nn.MSELoss()
in_features = cmds.shape[1]
out_features = cov.shape[1]
net = nn.Sequential(nn.Linear(in_features, 500),
                    nn.ReLU(),
                    nn.Linear(500, out_features))
net = net.to(device=try_gpu())

# Train
k, num_epochs, lr, weight_decay, batch_size, patience = 5, 1000, 5, 0, 64, 5
train_l, test_l = k_fold(k, net, loss, cmds, cov, num_epochs, lr, weight_decay, batch_size, patience)

# Plot
predicted_cov = net(cmds)

print('here')
