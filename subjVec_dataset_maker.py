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
from scipy import linalg
izip=zip
#from itertools import izip
from sklearn.metrics import accuracy_score, recall_score
from gensim.models import KeyedVectors
from pprint import pprint

def prototype(w):
    return w[:w.rfind('_s')]

if __name__ == '__main__':
    #30Kならマルチセンスが30000個、単語数自体は99156
    file_name = './vectors.MSSG.50D.30K.gz'
    mssg_word_vec = []
    mssg_word_info = []
    count = 1
    #global.vecとlocal.vecへの分離作業
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
                    mssg_word_vec.append(line)
  
    #global_modelとlocal_modelを作成する
    with open("MSSG.vectors.50D.30K_global.txt","w") as f_g:
        with open("MSSG.vectors.50D.30K_local.txt","w") as f_l:
            f_g.write("30000 50\n")
            f_l.write("90000 50\n")
            for global_index in range(len(mssg_word_info)):
                if mssg_word_info[global_index][1] == '3':
                    f_g.write(mssg_word_info[global_index][0]+' ')
                    f_g.write(mssg_word_vec[7*global_index])
                    f_l.write(mssg_word_info[global_index][0]+'_s0 ')
                    f_l.write(mssg_word_vec[7*global_index+1])
                    f_l.write(mssg_word_info[global_index][0]+'_s1 '+mssg_word_vec[7*global_index+3])
                    f_l.write(mssg_word_info[global_index][0]+'_s2 '+mssg_word_vec[7*global_index+5])


 