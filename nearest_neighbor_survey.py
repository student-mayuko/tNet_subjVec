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
from layers.dynamic_rnn import DynamicLSTM
izip=zip
#from itertools import izip
from sklearn.metrics import accuracy_score, recall_score

def cosine_sim(v1, v2):
    self_v1,self_v2 = np.array(v1,dtype = float),np.array(v2,dtype=float)
    return np.dot(self_v1, self_v2) / (np.linalg.norm(self_v1) * np.linalg.norm(self_v2))

if __name__ == '__main__':
    #30Kならマルチセンスが30000個、単語数自体は99156
    #本番
    '''
    file_name = './datasets/vectors.MSSG.50D.30K.gz'
    '''
    #練習　スモールサイズ用
    file_name = './datasets/50D30K_300small_size.txt'
    
    mssg_word_vec = []
    mssg_word_info = []
    single_word_vec = []
    single_word_info =[]
    count = 1
    start_index = 0
#本番用    
#    with gzip.open(file_name,"rt","utf-8") as fi:
    with open(file_name,"r") as fi:
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
    mssg_word_dict = {}
    #疑似マルチセンス候補のみを抽出する作業。ここではsense cluster center(もしかしたらglobal vectorも)を削除している。
    for i in range(len(mssg_word_vec)):
        if i%7==0 or i%7 ==2 or i%7 ==4 or i%7==6:
            if i%7==0:
                mssg_word_dict[mssg_word_info[i//7][0]] = mssg_word_vec[i] 
            mssg_word_vec[i] = []
    mssg_word_vec = [vec for vec in mssg_word_vec if vec != []]
    
    #subj_vec情報を得る
    subj_vec = []
    vec_cosine = []
    with open("subjVec_result.txt","r") as f:
        for line in f:
            subj_vec.append(line.split())
    #subj_vec =[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    #subj_vecとglobal vectorのcosine類似度を計算していく
    for i in range(len(mssg_word_info)):
        vec_cosine.append(cosine_sim(subj_vec[1],mssg_word_dict[mssg_word_info[i][0]]))
    #cosine類似度が大きい上位20語を選んで出力する
    with open("nearest_neighbor.txt","w") as f:
        for i in range(20):
            print(mssg_word_info[np.argsort(vec_cosine)[::-1][i]][0])
            f.write(mssg_word_info[np.argsort(vec_cosine)[::-1][i]][0])
