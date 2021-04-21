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
from scipy import linalg
izip=zip
#from itertools import izip
from sklearn.metrics import accuracy_score, recall_score


#データの読み込み、実行するプログラムを追加する必要あり。
class SGD:
    def __init__(self, eta=0.00002):
        self.eta = eta                  #学習率
        self.grad = np.array([])      #関数の勾配
        self.loss = 0        #損失関数?

    def choice_vec_by_shrink_rate(self,word_info,word_vec,k_size):
        shrink_rate = []
        x_vec,y_vec = [],[]
        k_xVector_set = []
        k_yVector_set = []
        new_word_info = []
        new_word_vec = []
        for i in range(len(word_info)):
            vec0,vec1,vec2=word_vec[3*i],word_vec[3*i+1],word_vec[3*i+2]
            #shrink_rate01 = (np.linalg.norm((self.M*vec0-self.M*vec1).to('cpu').detach().numpy().copy(),ord=2)**2)/(np.linalg.norm((vec0-vec1).to('cpu').detach().numpy().copy(),ord=2)**2)
            #shrink_rate02 = (np.linalg.norm((self.M*vec0-self.M*vec2).to('cpu').detach().numpy().copy(),ord=2)**2)/(np.linalg.norm((vec0-vec2).to('cpu').detach().numpy().copy(),ord=2)**2)
            #shrink_rate12 = (np.linalg.norm((self.M*vec1-self.M*vec2).to('cpu').detach().numpy().copy(),ord=2)**2)/(np.linalg.norm((vec1-vec2).to('cpu').detach().numpy().copy(),ord=2)**2)
            shrink_rate01 = (torch.norm(self.M*vec0-self.M*vec1)**2)/(torch.norm(vec0-vec1)**2)
            shrink_rate02 = (torch.norm(self.M*vec0-self.M*vec2)**2)/(torch.norm(vec0-vec2)**2)
            shrink_rate12 = (torch.norm(self.M*vec1-self.M*vec2)**2)/(torch.norm(vec1-vec2)**2)
            if min([shrink_rate01,shrink_rate02,shrink_rate12])==shrink_rate01:
                x_vec.append(vec0)
                y_vec.append(vec1)
            elif min([shrink_rate01,shrink_rate02,shrink_rate12])==shrink_rate02:
                x_vec.append(vec0)
                y_vec.append(vec2)                
            else: 
                x_vec.append(vec1)
                y_vec.append(vec2)
            shrink_rate.append(min([shrink_rate01,shrink_rate02,shrink_rate12]))
        #k個分の
        shrink_rate_index = np.argsort(shrink_rate)
        for i in range(k_size):
            k_xVector_set.append(x_vec[shrink_rate_index[i]])
            k_yVector_set.append(y_vec[shrink_rate_index[i]])
            new_word_info.append(word_info[shrink_rate_index[i]])
            new_word_vec.append(x_vec[shrink_rate_index[i]])
            new_word_vec.append(y_vec[shrink_rate_index[i]])
        self.k_size_word_info = new_word_info
        self.k_size_word_vec = new_word_vec
        return torch.tensor(k_xVector_set[0],dtype=torch.float64),torch.tensor(k_yVector_set[0],dtype=torch.float64)

    def sum_calculate(self,word_info,word_vec,key,k_size):
        loss_sum,word_loss =  0,0
        X,Y = torch.tensor([],dtype=torch.float64),torch.tensor([],dtype=torch.float64)
        for i in range(len(word_info)):
            X,Y=word_vec[2*i],word_vec[2*i+1]
            if key == "loss":
                #word_loss = np.linalg.norm((self.M*X-Y).to('cpu').detach().numpy().copy(),ord=2)**2
                word_loss = torch.norm(self.M*X-Y)**2+torch.norm(self.M*Y-Y)**2 
            if key == "grad":
                word_loss = 2*((self.M*X-Y)*X)+2*(self.M*Y-Y)*Y
            loss_sum += word_loss
        return loss_sum

    #訓練事象xとyのサンプリング→損失計算→勾配計算→パラメータ更新
    #wは変換行列Mに当たる
    #パラメータ的なのはx,y,M全部かな！
    #shrink_rate呼び出し多すぎだからタイミング変えようね
    def fit(self, word_info, word_vec):
        #初期設定
        seed_number = 8
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed_all(seed_number)
        torch.backends.cudnn.deterministic = True        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(len(word_vec)):
            for j in range(len(word_vec[i])):
                word_vec[i][j] = float(word_vec[i][j])
        first_word_index = 12174
        first_word_index = np.random.randint(0, len(word_info))
        first_word = word_info[first_word_index][0]
        vec_candidate = word_vec[3*first_word_index:3*(first_word_index+1)]
        vec_index_candidate = random.sample([0,1,2],2)       
        x_index,y_index=vec_index_candidate[0],vec_index_candidate[1]        
        x,y=torch.tensor(vec_candidate[x_index],dtype=torch.float64),torch.tensor(vec_candidate[y_index],dtype=torch.float64)
        self.M = torch.randn((50,50),dtype=torch.float64)
        before_loss,after_loss=0,1000
        before_word_x,before_word_y = x,y
        after_word_x,after_word_y = y,x
        while_count = 0
        self_word_info,self_word_vec = word_info,torch.tensor(word_vec,dtype=torch.float64)
        self.k_size_word_info= self_word_info
        self.k_size_word_vec = []
        k_size = 6000
        #疑似マルチセンスペアの初期設定
        for i in range(len(self_word_info)):
            vec0,vec1,vec2=self_word_vec[3*i],self_word_vec[3*i+1],self_word_vec[3*i+2]
            #shrink_rate01 = (np.linalg.norm((self.M*vec0-self.M*vec1).to('cpu').detach().numpy().copy(),ord=2)**2)/(np.linalg.norm((vec0-vec1).to('cpu').detach().numpy().copy(),ord=2)**2)
            #shrink_rate02 = (np.linalg.norm((self.M*vec0-self.M*vec2).to('cpu').detach().numpy().copy(),ord=2)**2)/(np.linalg.norm((vec0-vec2).to('cpu').detach().numpy().copy(),ord=2)**2)
            #shrink_rate12 = (np.linalg.norm((self.M*vec1-self.M*vec2).to('cpu').detach().numpy().copy(),ord=2)**2)/(np.linalg.norm((vec1-vec2).to('cpu').detach().numpy().copy(),ord=2)**2)
            shrink_rate01 = ((torch.norm(self.M*vec0-self.M*vec1))**2)/(torch.norm(vec0-vec1)**2)
            shrink_rate02 = ((torch.norm(self.M*vec0-self.M*vec2))**2)/(torch.norm(vec0-vec2)**2)
            shrink_rate12 = ((torch.norm(self.M*vec1-self.M*vec2))**2)/(torch.norm(vec1-vec2)**2)
            if min([shrink_rate01,shrink_rate02,shrink_rate12])==shrink_rate01:
                self.k_size_word_vec.append(vec0)
                self.k_size_word_vec.append(vec1)
            elif min([shrink_rate01,shrink_rate02,shrink_rate12])==shrink_rate02:
                self.k_size_word_vec.append(vec0)
                self.k_size_word_vec.append(vec2)                
            else: 
                self.k_size_word_vec.append(vec1)
                self.k_size_word_vec.append(vec2)          
        #学習回数分
        #この部分の終了条件設定を決める
        #while not(all(before_word_x == after_word_x) and all(before_word_y == after_word_y)):
        while while_count < 50:
            while_count += 1
            learn_count = 0
            print(while_count,"回目の更新")
            before_loss = 1
            after_loss = 0
            #損失と勾配を算出。その後Mの更新を行う
            #before_loss == after_lossになってもFalse判定を受けてる。できれば直したい。
            while abs(before_loss - after_loss) >= 0.0001:
                learn_count += 1
                before_loss = self.loss                                    
                #self.loss = self.sum_calculate(self_word_info,self_word_vec,"loss")+np.linalg.norm((self.M*y-y).to('cpu').detach().numpy().copy(),ord=2)**2 
                self.loss = self.sum_calculate(self.k_size_word_info,self.k_size_word_vec,"loss",k_size)
                self.grad = self.sum_calculate(self.k_size_word_info,self.k_size_word_vec,"grad",k_size)
                self.M -= self.eta * self.grad
                print(learn_count,"回目の学習")
                print('loss:',self.loss)
                print('M:',self.M)
                after_loss = self.loss
            #(x,y)の更新を行う
            x,y= self.choice_vec_by_shrink_rate(self_word_info,self_word_vec,k_size) 
            print(self.k_size_word_info[:20])
        eigen_value,subj_vec = linalg.eig(self.M)
        print("first word:",first_word)
        with open('subjVec_result.txt','w') as f2: 
            f2.write(str(self.M)+"\n")
            for i in range(len(self.k_size_word_info)):
                f2.write(self.k_size_word_info[i][0]+" ")
            f2.write("\n"+str(subj_vec))
            f2.write("\n"+str(subj_vec[np.argmin(eigen_value)]))
        return subj_vec[np.argmin(eigen_value)]

'''
class Adam:
    def __init__(self, feat_dim, loss_type='log', alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
        self.weight = np.zeros(feat_dim)  # features weight
        self.loss_type = loss_type  # type of loss function
        self.feat_dim = feat_dim  # number of dimension
        self.x = np.zeros(feat_dim)  # feature
        self.m = np.zeros(feat_dim)  # 1st moment vector
        self.v = np.zeros(feat_dim)  # 2nd moment vector
        self.alpha = alpha  # step size
        self.beta1 = beta1  # Exponential decay rates for moment estimates
        self.beta2 = beta2  # Exponential decay rates for moment estimates
        self.epsilon = epsilon
        self.t = 1  # timestep

    def fit(self, data_fname, label_fname):
        with open(data_fname, 'r') as f_data, open(label_fname, 'r') as f_label:
            for data, label in izip(f_data, f_label):
                self.features = np.array(data.rstrip().split(','), dtype=np.float64)
                y = int(-1) if int(label.rstrip())<=0 else int(1)  # posi=1, nega=-1に統一
                # update weight
                self.update(self.predict(self.features), y)
                self.t += 1
        return self.weight

    def predict(self, features): #margin
        return np.dot(self.weight, features)

    def calc_loss(self,m): # m=py=wxy
        if self.loss_type == 'hinge':
            return max(0,1-m)
        elif self.loss_type == 'log':
            # if m<=-700: m=-700
            return math.log(1+math.exp(-m))

    # gradient of loss function
    def calc_dloss(self,m): # m=py=wxy
        if self.loss_type == 'hinge':
            res = -1.0 if (1-m)>0 else 0.0 # lossが0を超えていなければloss=0.そうでなければ-mの微分で-1になる
            return res
        elif self.loss_type == 'log':
            if m < 0.0:
                return float(-1.0) / (math.exp(m) + 1.0) # yx-e^(-m)/(1+e^(-m))*yx
            else:
                ez = float( math.exp(-m) )
                return -ez / (ez + 1.0) # -yx+1/(1+e^(-m))*yx

    def update(self, pred, y):
        grad = y*self.calc_dloss(y*pred)*self.features  # gradient
        self.m = self.beta1*self.m + (1 - self.beta1)*grad  # update biased first moment estimate
        self.v = self.beta2*self.v + (1 - self.beta2)*grad**2  # update biased second raw moment estimate
        mhat = self.m/(1-self.beta1**self.t)  # compute bias-corrected first moment estimate
        vhat = self.v/(1-self.beta2**self.t)  # compute bias-corrected second raw moment estimate
        self.alpha *= np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)  # update stepsize
        self.weight -= self.alpha * mhat/(np.sqrt(vhat) + self.epsilon)  # update weight

if __name__=='__main__':
    data_fname = 'train800.csv'
    label_fname = 'trainLabels800.csv'
    test_data_fname = 'test200.csv'
    test_label_fname = 'testLabels200.csv'

    adam = Adam(40, loss_type='hinge')
    adam.fit(data_fname, label_fname)
    y_true = []
    y_pred = []
    with open(test_data_fname, 'r') as f_data, open(test_label_fname, 'r') as f_label:
        for data, label in zip(f_data, f_label):
            pred_label = adam.predict(np.array(data.rstrip().split(','), dtype=np.float64))
            y_true.append(int(label))
            y_pred.append( 1 if pred_label>0 else 0)
    print 'accuracy:', accuracy_score(y_true, y_pred)
    print 'recall:', recall_score(y_true, y_pred)

'''
def is_num_judge(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
    
def cosine_sim(v1, v2):
    self_v1,self_v2 = v1,np.array(v2,dtype=float)
    return np.dot(self_v1, self_v2) / (np.linalg.norm(self_v1) * np.linalg.norm(self_v2))

if __name__ == '__main__':
    #30Kならマルチセンスが30000個、単語数自体は99156
    #本番
    file_name = './datasets/vectors.MSSG.50D.30K.gz'
    '''
    #練習　スモールサイズ用
    file_name = './datasets/50D30K_300small_size.txt'
    '''
    mssg_word_vec = []
    mssg_word_info = []
    single_word_vec = []
    single_word_info =[]
    count = 1
    start_index = 0
    mssg_word_dict = {}
#本番用    
    with gzip.open(file_name,"rt","utf-8") as fi:
#   with open(file_name,"r") as fi:
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
    #辞書型データの作成、近傍調査用(多義語＋単一語)
    start_index_list = []
    start_index_content = 0
    for index in range(len(mssg_word_info)):
        start_index_list.append(start_index_content)
        if mssg_word_info[index][1] == '1':
            start_index_content += 3 
        else:
            start_index_content += 7
    for index in range(len(mssg_word_info)):
        mssg_word_dict[mssg_word_info[index][0]] = mssg_word_vec[start_index_list[index]]
    '''
    #辞書型データの作成(多義語のみ) 
    for index in range(len(mssg_word_info)):
        if mssg_word_info[index][1] == '3':
            mssg_word_dict[mssg_word_info[index][0]] = mssg_word_vec[start_index_content] 
            start_index_content += 7
        else:
            start_index_content += 3
    '''
    #多義語のみ抽出する事前作業
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

    #疑似マルチセンス候補のみを抽出する作業。ここではsense cluster center(もしかしたらglobal vectorも)を削除している。
    for i in range(len(mssg_word_vec)):
        if i%7==0 or i%7 ==2 or i%7 ==4 or i%7==6:
            mssg_word_vec[i] = []
    mssg_word_vec = [vec for vec in mssg_word_vec if vec != []]

    sgd = SGD()
    vec_cosine = []
    subj_vec = sgd.fit(mssg_word_info,mssg_word_vec)    
    for i in range(len(mssg_word_info)):
        vec_cosine.append(cosine_sim(subj_vec,mssg_word_dict[mssg_word_info[i][0]]))
    #cosine類似度が大きい上位20語を選んで出力する
    vec_cosine_index = np.argsort(vec_cosine)[::-1]
    with open("nearest_neighbor.txt","w") as f:
        for i in range(20):
            print(mssg_word_info[vec_cosine_index[i]][0])
            f.write(mssg_word_info[np.argsort(vec_cosine)[::-1][i]][0])
   
