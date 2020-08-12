#!/usr/bin/python3
#@File: test_1.py
#--coding:utf-8--
#@Author:zxqyiyang
#@time: 20-7-5 上午9:44
#说明:
#总结:

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math


class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.line1 = nn.Linear(7, 20)
        self.sm1 = nn.Sigmoid()
        self.d1 = nn.Dropout(0.4)

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

"""
mean and std of data
772.6783162328746 1103.830952793306
1.6403462193031103e+23 3.351996202843185e+23
17619733017847.305 23810541574320.91
 88.59402292901886 196.6633921949684
4138574328687.502 7389237439615.587
9.612613615409758e+21 1.3222859985064443e+22
581.6742464534218 704.661909539274
4921.915947872575 4113.651133263181
2.3351563494456173 1.6586153685634057
23.338458165511007 16.71207379003364
"""
number_u_s = np.array([[772.9499192543446, 1103.830952793306],
                  [1.6403462193031103e+23, 3.351996202843185e+23],
                  [17619733017847.305, 23810541574320.91],
                  [88.59402292901886, 196.6633921949684],
                  [4138574328687.502, 389237439615.587],
                  [9.612613615409758e+21, 1.3222859985064443e+22],
                  [581.6742464534218, 704.661909539274],
                  [4921.915947872575, 4113.651133263181],
                  [2.3351563494456173, 1.6586153685634057],
                  [23.338458165511007, 16.71207379003364]])

def read_test_data():
    file_path = "./data/test_para_input.txt"
    f = open(file_path)
    data = f.readlines()
    f.close()
    input_data = []
    name_list = []
    for list in data:
        list = list.replace("\n","").split(" ")
        x = []
        name = []
        # print(list)
        for l in list:
            if(l != ""):
               try:
                   l = float(l)
                   if(math.isnan(l)):
                       l=0
                   x.append(l)
               except:
                   name.append(l)
        # print(x)
        input_data.append(x)
        name_list.append(name)
    # print(input_data)
    input_data = np.array(input_data)
    return input_data, name_list

def data_normalization(x):
    for j in range(0,10,1):
        # x[:, j] = (x[:, j] - number_max_min[j][1])/(number_max_min[j][0] - number_max_min[j][1])
        x[:, j] = (x[:, j] - number_u_s[j][0])/number_u_s[j][1]
    # data = np.array(np.argwhere(np.isnan(x))).astype(np.int)
    # for i in range(data.shape[0]):
    #     x[data[i, 0], data[i, 1]] = 0
    return x

def save_to_txt(list):
    data = list
    f = open("./test_name_out.txt", "w")
    # f = open("./test_out.txt", "w")
    for i in range(len(data)):
        f.write(str(data[i][0]).strip("'") + str(" ") + str(data[i][1]) + str("\n"))
        # f.write(str(data[i]) + str("\n"))
        # print(str(data[i][0]).strip("'") + str(" ") + str(data[i][1]))
    f.close()

def test():
    # net = Line_deep()
    net = Line()
    net.load_state_dict(torch.load("model_line_max_min_all_data"))
    net.eval()
    input_data, name_list = read_test_data()
    input_data = data_normalization(input_data)
    name_list = np.array(name_list)
    print(input_data.shape)
    results = []
    for i, input in enumerate(input_data):
        result = []
        # input = input[0:7]
        input = Variable(torch.tensor(input, dtype=torch.float32))
        output = net(input)
        output = output.detach().numpy()
        output = output.tolist()
        # print(output)
        output = output.index(max(output))
        result.append(name_list[i,0])
        result.append(output)
        print(name_list[i,0], output)
        results.append(result)
        # results.append(output)
    save_to_txt(results)

if __name__ == '__main__':
    test()