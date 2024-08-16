'''
Softmax 回归。计算accuracy。
'''
from answerMultiLayerPerceptron import buildGraph, lr, wd1, wd2, batchsize
import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed

import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/mlp.npy"

X=mnist.trn_X
Y=mnist.trn_Y 

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [StdScaler(mnist.mean_X, mnist.std_X), Linear(mnist.num_feat, mnist.num_class), tanh(), LogSoftmax(), NLLLoss(Y)]
    graph=Graph(nodes)
    return graph

def Train(lr,wd1,wd2,batchsize):
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 60+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        # print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

    # 测试
    with open(save_path, "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(mnist.val_X, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    # print("valid acc", np.average(haty==mnist.val_Y))
    return np.average(haty==mnist.val_Y)

best_superParam=None
best_acc=0
Lr=np.arange(0.001,0,-0.0001)
Wd1=np.arange(0.0001,0,-0.00001)
Wd2=np.arange(0.0001,0,-0.00001)
Batchsize=[128,256]
for lr in Lr:
    for wd1 in Wd1:
        for wd2 in Wd2:
            for batchsize in Batchsize:
                acc=Train(lr,wd1,wd2,batchsize)
                if acc>best_acc:
                    best_superParam=(lr,wd1,wd2,batchsize)
                print(acc)
print(best_superParam)