#!/usr/bin/python3
#@File: processe_data.py
#--coding:utf-8--
#@Author:zxqyiyang
#@time: 20-7-1 下午1:02
#说明:
#总结:
input_file ="./data/train_para_input.txt"
output_file="./data/train_output.txt"
import numpy as np
import math

def data_normalization(data):
    for j in range(0,10,1):
        data_max = np.max(data[:,j])
        print(data_max)
        data_min = np.min(data[:,j])
        if(data_max !=0 and data_min != 0):
            data[:,j] = (data[:,j]-data_min)/(data_max-data_min)
    return data

def read_input():
    f = open(input_file)
    data = f.readlines()
    f.close()
    input_data = []
    for list in data:
        list = list.replace("\n","").split(" ")
        x = []
        for l in list:
            if(l != ""):
               try:
                   l = float(l)
                   if(math.isnan(l)):
                       l = 0
                   x.append(float(l))
               except:
                   x.append(l)
        input_data.append(x)
    return input_data

def read_output():
    f = open(output_file)
    data = f.readlines()
    f.close()
    output_data = []
    for list in data:
        list = list.replace("\n","").split(" ")
        x = []
        for l in list:
            if(l!=""):
                try:
                    x.append(float(l))
                except:
                    x.append(l)
        output_data.append(x)
    return output_data
    
if __name__ == '__main__':
    read_input()
    # read_output()