#!/usr/bin/python3
#@File: train_1.py
#--coding:utf-8--
#@Author:zxqyiyang
#@time: 20-7-5 上午9:02
#说明:
#总结:
import read_data
from sklearn.model_selection import train_test_split # 用于拆分训练、验证集
import torch.utils.data as data_utils

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


"""加载数据"""
input_data, output_data = read_data.read_input(), read_data.read_output()
x, y = [], []
for i in range(len(input_data)):
    if(input_data[i][0] == output_data[i][0]):
        if(output_data[i][1] == 1):
            x.append(input_data[i][1:])
            label = [0, 1]
            y.append(label)
        # elif( i % 20 ==0):
        else:
            x.append(input_data[i][1:])
            label = [1, 0]
            y.append(label)

"""remove nan in data"""
x, y = np.array(x).astype(np.float64), np.array(y).astype(np.float64)
data = np.array(np.argwhere(np.isnan(x))).astype(np.int)
# print(data[:,0])
x = np.delete(x, data[:,0],axis=0)
y = np.delete(y, data[:,0],axis=0)
x, y = x.tolist(), y.tolist()

"""数据归一化"""
# for j in range(0,10,1):
#     list = [data[j] for data in x]
#     list_max = max(list)
#     list_min = min(list)
#     print(list_max, list_min)
#     for i in range(len(list)):
#         list[i] = (list[i]-list_min)/(list_max-list_min)
#         x[i][j] = list[i]
# #
# #
# 数据预处理方法：采用均值、标方差做归一化
for j in range(0,10,1):
    list = [data[j] for data in x]
    np_list = np.array(list)
    u = np.mean(np_list)
    s = np.std(np_list)
    print(u, s)
    for i in range(len(list)):
        list[i] = (list[i] - u) / s
        x[i][j] = list[i]
x, y = np.array(x).astype(np.float64), np.array(y).astype(np.float64)
x = x[:, 0:7]
print(x.shape, y.shape)
# print(x[100], x[57000])
# print(y[100], y[57000])
# x = input()


# x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20)
# print(x_train.shape, y_train.shape)
train = data_utils.TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
train_loader = data_utils.DataLoader(train, batch_size=100, shuffle=True)


class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.line1 = nn.Linear(7, 20)
        self.sm1 = nn.Sigmoid()
        self.d1 = nn.Dropout(0.2)

        self.line2 = nn.Linear(20,10)
        self.sm2 = nn.Sigmoid()
        self.d2 = nn.Dropout(0.2)

        self.line4 = nn.Linear(10,6)
        self.sm4 = nn.Sigmoid()
        self.d4 = nn.Dropout(0.2)

        self.line3 = nn.Linear(6,2)
        self.sm3 = nn.Sigmoid()
        self.d3 = nn.Dropout(0.2)

    def forward(self, x):
        out = self.sm1(self.d1(self.line1(x)))
        out = self.sm2(self.d2(self.line2(out)))
        out = self.sm4(self.d4(self.line4(out)))
        out = self.sm3(self.d3(self.line3(out)))
        return out
net = Line()
criterion = nn.BCELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.01)


for epoch in range(100):
    net.train()
    train_correct = 0.
    train_count = 0
    train_loss = 0
    for batch_id, (inputs, targets) in enumerate(train_loader):
        output = net(inputs)
        loss = criterion(output, targets)
        train_loss += float(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_count += 1
        targets, output = targets.data.numpy(), output.detach().numpy()
        for i in range(output.shape[0]):
            if(output[i,0] > output[i,1]):
                pre = [1,0]
            else:
                pre = [0,1]
            target = targets[i,:].tolist()
            if(pre.index(max(pre))==target.index(max(target))):
                train_correct += 1.
    print(epoch, train_loss/train_count, train_correct/59706.)

path = "./model_line_max_min_all_data"
torch.save(net.state_dict(), path)