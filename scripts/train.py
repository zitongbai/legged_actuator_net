import torch 
from torch.utils.data import DataLoader, random_split

from dataset import JointDataset, read_bags, preprocess_data
from actuator_net import ActuatorNet

import os
import copy

def train(bag_file, batch_size, num_epochs, lr, device):
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    bag_dir = os.path.join(this_file_dir, '..', 'bags')
    
    # Read data and preprocess
    data_dict = read_bags(bag_file, bag_dir=bag_dir)
    Xs, Ys = preprocess_data(data_dict)
    
    data_set = JointDataset(Xs, Ys)
    
    # Split data into train, val, test
    num_train = int(0.7 * len(data_set))
    num_val = int(0.15 * len(data_set))
    num_test = len(data_set) - num_train - num_val
    train_set, val_set, test_set = random_split(data_set, [num_train, num_val, num_test])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    net = ActuatorNet(input_size=6, output_size=1, hidden_size=32, layer_num=3, activation='softsign')
    net = net.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-8)
    
    for epoch in range(num_epochs):
        net.train()
        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_hat = net(X)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            
            # if i % 100 == 0:
            #     print(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
        
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (X, Y) in enumerate(val_loader):
                X, Y = X.to(device), Y.to(device)
                Y_hat = net(X)
                loss = criterion(Y_hat, Y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}, val loss: {val_loss}")
    
    test_loss = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            Y_hat = net(X)
            loss = criterion(Y_hat, Y)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    print(f"Test loss: {test_loss}")
    
    return net

if __name__ == '__main__':
    bag_file = 'go2_real_data_50Hz.bag'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = train(bag_file, batch_size=128, num_epochs=30, lr=8e-4, device=device)
    print(net)
    
    # export to jit
    
    this_file_dir = os.path.dirname(os.path.realpath(__file__))
    jit_dir = os.path.join(this_file_dir, '..', 'logs')
    os.makedirs(jit_dir, exist_ok=True)
    jit_file = os.path.join(jit_dir, 'go2_actuator_net.pt')
    print(f"Exporting to {jit_file}")
    model = copy.deepcopy(net).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(jit_file)