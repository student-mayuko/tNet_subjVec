#-*- coding: utf-8 -*-
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import torch
import torch.nn as nn
import gzip
import torch.nn.functional as F
import numpy as np
import sympy as sym
izip=zip
#from itertools import izip
from sklearn.metrics import accuracy_score, recall_score

if __name__ == '__main__':
    #30Kならマルチセンスが30000個、単語数自体は99156
    file_name = './datasets/vectors.MSSG.50D.30K.gz'
    mssg_word_vec = []
    mssg_word_info = []
    single_word_vec = []
    single_word_info =[]
    count = 1
    start_index = 0
    with gzip.open(file_name,"rt","utf-8") as fi:
        for line in fi:
            #初期設定
            if count == 1:
                count +=1
            #初期設定した後の処理
            else:
                if len(line.split())==2:
                    mssg_word_info.append(line.split())
                else:
                    mssg_word_vec.append(line.split())
    #多義語のみ抽出する作業
    for index in range(0,len(mssg_word_info)):
        if mssg_word_info[index][1] == '1':
            single_word_info.append(mssg_word_info[index])
            mssg_word_info[index] = []
            change_index = start_index
            for i in range(3):
                single_word_vec.append(mssg_word_vec[change_index])
                mssg_word_vec[change_index] = []
                change_index += 1
            start_index += 3
        else:
            start_index += 7
    mssg_word_info = [vec for vec in mssg_word_info if vec != []]
    mssg_word_vec = [vec for vec in mssg_word_vec if vec != []]
    with open('./datasets/50D30K_3000small_size.txt',mode='w') as f1:
        for index in range(0,3000):
            f1.write(str(mssg_word_info[index])+"\n")
            for j in range(0,7):
                f1.write(str(mssg_word_vec[index*7+j])+"\n")
    with open('./datasets/50D30K_300small_size.txt',mode='w') as f2:
        for index in range(0,300):
            f2.write(str(mssg_word_info[index])+"\n")
            for j in range(0,7):
                f2.write(str(mssg_word_vec[index*7+j])+"\n")