import torch
from torch.utils.data import Dataset, DataLoader

import os
from pathlib import Path
import numpy as np
from scipy import interpolate
from rosbags.highlevel import AnyReader

def read_bags(bag_file, bag_dir='../bags', joint_num=12):
    """Read rosbag file and return data

    Args:
        bag_file (str): _description_
        bag_dir (str, optional): bag directory. Defaults to '../bags'.
        joint_num (int, optional): number of joints. Defaults to 12.

    Returns:
        data_dict: dictionary containing the message data
    """
    
    bag_path = os.path.join(bag_dir, bag_file)
    
    print(f"Reading rosbag from {bag_path}...")
    with AnyReader([Path(bag_path)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/actuator_data']

        msg_len = connections[0].msgcount
        print(f"Number of messages: {msg_len}")

        msg_time = np.zeros(msg_len)
        joint_pos = np.zeros((msg_len, joint_num))
        joint_vel = np.zeros((msg_len, joint_num))
        joint_tau_est = np.zeros((msg_len, joint_num))
        joint_pos_des = np.zeros((msg_len, joint_num))
        joint_vel_des = np.zeros((msg_len, joint_num))

        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections)):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # record start time
            if i == 0:
                start_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            msg_time[i] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - start_time
            joint_pos[i] = msg.pos
            joint_vel[i] = msg.vel
            joint_tau_est[i] = msg.tau_est
            joint_pos_des[i] = msg.pos_des
            joint_vel_des[i] = msg.vel_des

    data_dict = {
        'msg_time': msg_time,
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'joint_tau_est': joint_tau_est,
        'joint_pos_des': joint_pos_des,
        'joint_vel_des': joint_vel_des
    }
    
    return data_dict
    

def preprocess_data(data_dict, dt=None, history_len=3, joint_num=12):
    """Preprocess data

    Args:
        data_dict (dictionary): dictionary containing the message data
        dt (float, optional): if not None, interpolate data according to dt. Defaults to None.
        history_len (int, optional): history length. Defaults to 3.
        joint_num (int, optional): number of joints. Defaults to 12.

    Returns:
        Xs (np.array): input data for the neural network
        Ys (np.array): output data for the neural network
    """
    # parse data
    msg_time = data_dict['msg_time']
    joint_pos = data_dict['joint_pos']
    joint_vel = data_dict['joint_vel']
    joint_tau_est = data_dict['joint_tau_est']
    joint_pos_des = data_dict['joint_pos_des']
    joint_vel_des = data_dict['joint_vel_des']
    
    joint_pos_err = joint_pos - joint_pos_des
    joint_vel_err = joint_vel # - joint_vel_des
    
    t_num = len(msg_time)
    history_len = 3
    
    if dt is not None:
        # interpolate data according to dt
        f_joint_pos_err = interpolate.interp1d(msg_time, joint_pos_err, axis=0, kind='linear')
        f_joint_vel_err = interpolate.interp1d(msg_time, joint_vel_err, axis=0, kind='linear')
        f_joint_tau_est = interpolate.interp1d(msg_time, joint_tau_est, axis=0, kind='linear')
        
        new_msg_time = np.arange(0, msg_time[-1], dt)
        t_num = len(new_msg_time)
        
        joint_pos_err = f_joint_pos_err(new_msg_time)
        joint_vel_err = f_joint_vel_err(new_msg_time)
        joint_tau_est = f_joint_tau_est(new_msg_time)
        
    data_num = (t_num - history_len)*joint_num
    
    Xs = np.zeros((data_num, history_len*2)) # 2 because of joint_pos_err and joint_vel_err
    Ys = np.zeros((data_num, 1))

    for i in range(t_num - history_len):
        for j in range(joint_num):
            j_pos_err_hist = joint_pos_err[i:i+history_len, j][::-1]  # newest to oldest
            j_vel_err_hist = joint_vel_err[i:i+history_len, j][::-1]  # newest to oldest
            j_pos_vel_err_hist = np.concatenate((j_pos_err_hist, j_vel_err_hist), axis=0)   # shape: (history_len*2,)
            
            Xs[i*joint_num+j, :] = j_pos_vel_err_hist
            Ys[i*joint_num+j, :] = joint_tau_est[i+history_len, j]
            
    return Xs, Ys


class JointDataset(Dataset):
    def __init__(self, Xs, Ys):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.Ys = torch.tensor(Ys, dtype=torch.float32)
    
    def __len__(self):
        return self.Xs.shape[0]
    
    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]
    
    
if __name__ == '__main__':
    bag_file = 'go2_real_data.bag'
    data_dict = read_bags(bag_file)
    
    msg_time = data_dict['msg_time']
    dts = np.diff(msg_time)
    print(f"dt min: {np.min(dts)}, dt max: {np.max(dts)}")
    
    Xs, Ys = preprocess_data(data_dict)
    print(Xs.shape, Ys.shape)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(Ys[:12000:12], '-x')
    plt.show()
    
    data_set = JointDataset(Xs, Ys)
    num_train = int(0.7 * len(data_set))
    num_val = int(0.15 * len(data_set))
    num_test = len(data_set) - num_train - num_val
    
    train_set, val_set, test_set = torch.utils.data.random_split(data_set, [num_train, num_val, num_test])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    