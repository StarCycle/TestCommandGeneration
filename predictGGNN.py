import json
import torch
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from GNN import GNNModel

def analysis(net, cmds, cov, output):
    true_positive = output[net(cmds, cov)>0.5].sum().item()
    true_negative = (output[net(cmds, cov)<0.5]==0).sum().item()
    fault_positive = (output[net(cmds, cov)>0.5]==0).sum().item()
    fault_negative = output[net(cmds, cov)<0.5].sum().item()
    accuracy = (true_positive + true_negative) / output.numel()
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
        for cmds, cov, output in train_iter:
            optimizer.zero_grad()
            l = loss(net(cmds, cov), output)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_cmds, train_cov, train_output = trainset[:100]
            test_cmds, test_cov, test_output = testset[:100]
            cur_train_ls = loss(net(train_cmds, train_cov), train_output).data.item()
            cur_test_ls = loss(net(test_cmds, test_cov), test_output).data.item()
            cur_train_acc, cur_train_prec, cur_train_recall, cur_train_f1 = analysis(net, train_cmds, train_cov, train_output)
            cur_test_acc, cur_test_prec, cur_test_recall, cur_test_f1 = analysis(net, test_cmds, test_cov, test_output)
            train_ls.append(cur_train_ls)
            test_ls.append(cur_test_ls)
            log_writer.add_scalars('loss', {'train loss': cur_train_ls, 'test loss': cur_test_ls}, epoch)
            log_writer.add_scalars('accuracy', {'train acc': cur_train_acc, 'test acc': cur_test_acc}, epoch)
            log_writer.add_scalars('precison', {'train prec': cur_train_prec, 'test prec': cur_test_prec}, epoch)
            log_writer.add_scalars('recall', {'train recall': cur_train_recall, 'test recall': cur_test_recall}, epoch)
            log_writer.add_scalars('f1', {'train f1': cur_train_f1, 'test f1': cur_test_f1}, epoch)
            print(f'epoch {epoch}: train loss {cur_train_ls}, test loss {cur_test_ls}')
        # Early stop
        if epoch % 20 == 0:
            if epoch == 0:
                best_test_ls = cur_test_ls
            if cur_test_ls < best_test_ls:
                best_test_ls = cur_test_ls
                torch.save(net.state_dict(), 'Model')
            elif sum(test_ls[-patiance:])/min(len(test_ls), patiance) > best_test_ls:
                break

def try_gpu():  
    """If there is a GPU, return gpu(0)，otherwise return cpu()"""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')

def Preprocess(inputs):
    inputs -= torch.mean(inputs, axis=0)
    inputs /= torch.std(inputs, axis=0)
    inputs = torch.nan_to_num(inputs, nan=0)
    return inputs

if __name__ == "__main__":

    with open('dataset.json', 'r') as f:
        embeddings = json.load(f)
    with open('graph.json', 'r') as f:
        graph = json.load(f)
    device = try_gpu()
    cmds = torch.tensor(embeddings['cmd'], device=device, dtype=torch.float32)
    cmdsInput = cmds[:-1, :]
    cov = torch.tensor(embeddings['cov'], device=device, dtype=torch.float32)
    covInput = cov[:-1, 1:]
    covOutput = cov[1:, 1:]
    nodes = torch.tensor(graph['features'], device=device, dtype=torch.float32)
    edges = torch.tensor(graph['edges'], device=device, dtype=torch.long)
    codeCount2Label = torch.tensor(graph['codeCount2Label'], device=device, dtype=torch.long)
    edges = edges.T

    # Preprocessing
    cmdsInput = Preprocess(cmdsInput)

    # Network
    loss = nn.MSELoss()
    net = GNNModel(nodes, edges, codeCount2Label, cmdsInput.size()[1], covOutput.size()[1])
    net = net.to(device)

    # Train
    num_epochs, lr, weight_decay, batch_size, patience, split = 10, 1e-4, 0, 32, 10, 0.8
    dataset = data.TensorDataset(cmdsInput, covInput, covOutput)
    trainset_size = int(len(dataset)*0.8) 
    trainset, testset = data.random_split(dataset, [trainset_size, int(len(dataset)) - trainset_size])
    train(net, loss, trainset, testset, num_epochs, lr, weight_decay, batch_size, patience)

    print('here')
