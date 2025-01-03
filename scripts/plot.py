import torch
from torch.utils.data import DataLoader, random_split
from dataset import JointDataset, read_bags
import os
import numpy as np
import matplotlib.pyplot as plt

this_file_dir = os.path.dirname(os.path.realpath(__file__))
jit_dir = os.path.join(this_file_dir, '..', 'logs')
jit_file = os.path.join(jit_dir, 'actuator_net.pt')

# load jit
net = torch.jit.load(jit_file)

bag_dir = os.path.join(this_file_dir, '..', 'bags')
bag_file = 'go2_real_data.bag'
Xs, Ys = read_bags(bag_file, bag_dir=bag_dir, dt=0.01)
data_set = JointDataset(Xs, Ys)

Ys_hat = np.zeros_like(Ys)

net.eval()
with torch.no_grad():
    for i in range(len(data_set)):
        X, Y = data_set[i]
        Y_hat = net(X)
        Ys_hat[i] = Y_hat.item()
        
vis_len = 12000
vis_jnt = 3
plt.figure()
plt.plot(Ys[vis_jnt:vis_len:12], label='ground truth')
plt.plot(Ys_hat[vis_jnt:vis_len:12], label='prediction')
plt.legend()
plt.show()





# num_train = int(0.7 * len(data_set))
# num_val = int(0.15 * len(data_set))
# num_test = len(data_set) - num_train - num_val

# train_set, val_set, test_set = random_split(data_set, [num_train, num_val, num_test])
# batch_size = 128
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# # plot the prediction and ground truth in test set
# import matplotlib.pyplot as plt
# import numpy as np

# net.eval()
# with torch.no_grad():
#     for i, (X, Y) in enumerate(test_loader):
#         Y_hat = net(X)
#         break

# Y = Y.cpu().numpy()
# Y_hat = Y_hat.cpu().numpy()

# plt.figure()
# plt.plot(Y, label='ground truth')
# plt.plot(Y_hat, label='prediction')
# plt.legend()

# plt.show()
