import numpy as np
from numpy.random import randn
import mnist
import pickle
import matplotlib.pyplot as plt
from util import setseed
from answerMultiLayerPerceptron import buildGraph, lr, wd1, wd2, batchsize
from autograd.utils import PermIterator
import os
import sys
import time

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/your.npy"

def recover_image(image_vector):
    return image_vector.reshape(28,28)

def flatten_image(image):
    return image.reshape(-1)

def shift_image(image,shift:tuple):
    shifted_image=np.roll(image,shift,axis=(0,1))
    return shifted_image

def rotate_image(image,angle):
    angle = np.radians(angle)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    height,width=28,28
    new_width = int(np.ceil(width * np.abs(cos_theta) + height * np.abs(sin_theta)))
    new_height = int(np.ceil(width * np.abs(sin_theta) + height * np.abs(cos_theta)))

    cx = width / 2
    cy = height / 2
    new_cx = new_width / 2
    new_cy = new_height / 2

    rotated_image = np.zeros((height, width))
    for X in range(width):
        for Y in range(height):
            x=int(np.floor(X+new_cx-cx))
            y=int(np.floor(Y+new_cy-cy))
            rx = cos_theta * (x - new_cx) + sin_theta * (y - new_cy) + cx
            ry = -sin_theta * (x - new_cx) + cos_theta * (y - new_cy) + cy
            if 0 <= rx < width and 0 <= ry < height:
                rotated_image[Y, X] = bilinear_interpolation(image, rx, ry)
    return rotated_image

def scale_image(image,scaling):
    height,width=28,28
    new_height=int(np.ceil(height*scaling))
    new_width=int(np.ceil(width*scaling))

    cx = width / 2
    cy = height / 2
    new_cx = new_width / 2
    new_cy = new_height / 2

    scaled_image=np.zeros((height, width))
    for X in range(width):
        for Y in range(height):
            x=int(np.floor(X+new_cx-cx))
            y=int(np.floor(Y+new_cy-cy))
            rx=x/scaling
            ry=y/scaling
            if 0 <= rx < width and 0 <= ry < height:
                scaled_image[Y,X]=bilinear_interpolation(image, rx, ry)
    return scaled_image

def bilinear_interpolation(image, x, y):
    """
    双线性插值
    """
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = x1 + 1, y1 + 1

    # 边界检查
    if x1 >= image.shape[1] - 1:
        x1 = image.shape[1] - 2
        x2 = image.shape[1] - 1
    if y1 >= image.shape[0] - 1:
        y1 = image.shape[0] - 2
        y2 = image.shape[0] - 1

    # 四个像素的值
    q11 = image[y1, x1]
    q21 = image[y1, x2]
    q12 = image[y2, x1]
    q22 = image[y2, x2]

    # 双线性插值计算
    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1) + 1e-6)

def add_noise(image,noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image

def Generator(image_vector):
    old_image=recover_image(image_vector)
    angle=np.random.randint(-45,45)
    new_image=rotate_image(old_image,angle)
    shift=np.random.randint(-4,4,size=2)
    new_image=shift_image(new_image,shift)
    scaling=np.random.uniform(0.9,1.1)
    new_image=scale_image(new_image,scaling)
    noise_level = np.random.randint(5, 10)
    new_image = add_noise(new_image, noise_level)
    return np.expand_dims(new_image,axis=0)

from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

X=np.concatenate((mnist.trn_X,mnist.val_X))
Y=np.concatenate((mnist.trn_Y,mnist.val_Y))
X=mnist.trn_X
Y=mnist.trn_Y

scale=1     #数据集扩充规模
k=4        #k-折
lr = 1e-1   # 学习率 3e-3
wd1 = 0  # L1正则化
wd2 = 0  # L2正则化
batchsize = 1024

def buildGraph(Y,std_X,mean_X):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    """
    valid acc 0.9571428571428572
    test acc 0.96
    """
    # nodes = [StdScaler(mean_X,std_X), Convolution((6,1,5,5)), BatchNorm((6,24,24)), relu(), MaxPooling(), Flatten((6,12,12)), Linear(864,256), relu(), Linear(256,256), relu(), Linear(256,mnist.num_class), relu(), LogSoftmax(), NLLLoss(Y)]
    """
    valid acc 0.9527142857142857
    test acc 0.95
    """
    # nodes = [StdScaler(mean_X,std_X), Convolution((3,1,5,5)), BatchNorm((3,24,24)), relu(), MaxPooling(), Flatten((3,12,12)), Linear(432,256), relu(), Linear(256,256), relu(), Linear(256,mnist.num_class), relu(), LogSoftmax(), NLLLoss(Y)]
    nodes = [StdScaler(mean_X,std_X), convolution((6,1,5,5)), batchNorm((6,24,24)), relu(), MaxPooling(), Flatten((6,12,12)), Linear(864,512), relu(), Linear(512,512), relu(), Linear(512,256), relu(), Linear(256,mnist.num_class), relu(), LogSoftmax(), NLLLoss(Y)]
    graph=Graph(nodes)
    return graph



if __name__ == "__main__":
    #数据增强
    print('开始数据增强')
    TX,TY=[],[]
    if os.path.exists('MNIST/augmentation_data_CNN.npy'):
        TX=np.load('MNIST/augmentation_data_CNN.npy')
        TY=np.load('MNIST/augmentation_targets_CNN.npy')
    else:
        for _ in range(scale):
            for x,y,idx in zip(X,Y,range(len(X))):
                TX.append(Generator(x))
                TY.append(y)
                print(idx)
        TX=np.array(TX)
        TY=np.array(TY)
        np.save('MNIST/augmentation_data_CNN.npy',TX)
        np.save('MNIST/augmentation_targets_CNN.npy',TY)
        # sys.exit()
    print('结束数据增强')
    """
    # #打乱数据
    # print('开始打乱数据')
    # combined=list(zip(TX,TY))
    # np.random.shuffle(combined)
    # TX,TY=zip(*combined)
    # TX=np.array(TX)
    # TY=np.array(TY)
    # print('结束打乱数据')
    #划分数据
    """
    print('开始划分数据')
    L=len(TX)
    p=int(np.ceil(L/k*(k-1)))
    p=50000
    TXT=TX[:p]
    TYT=TY[:p]
    TXV=TX[p:int(1.2*p)]
    TYV=TY[p:int(1.2*p)]

    # TXT,TYT=mnist.trn_X,mnist.trn_Y
    # TXV,TYV=mnist.val_X,mnist.val_Y
    # TXV,TYV=TXT,TYT
    # TXT=np.expand_dims(TXT.reshape(-1,28,28),axis=1)
    # TXV=np.expand_dims(TXV.reshape(-1,28,28),axis=1)
    std_X,mean_X=np.std(TXT, axis=0, keepdims=True)+1e-4, np.mean(TXT, axis=0, keepdims=True)
    print('结束划分数据')
    # 训练
    print('开始训练')
    graph = buildGraph(TYT,std_X,mean_X)
    best_train_acc = 0
    dataloader = PermIterator(TXT.shape[0], batchsize)
    for i in range(1, 100+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = TXT[perm]
            tY = TYT[perm]
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
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        graph.eval()
        graph.flush()
        pred = graph.forward(TXV, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=1)
        "valid acc"
        acc=np.average(haty==TYV)
        print("valid acc", np.average(haty==TYV))
        testX=np.expand_dims(mnist.test_X,axis=1)
        testY=mnist.test_Y
        graph.eval()
        graph.flush()
        pred = graph.forward(testX, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=1)
        print("test acc", np.average(haty==testY))
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)
    print('结束训练')
    # 测试
    print('开始测试')
    with open(save_path, "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(TXV, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    print("valid acc", np.average(haty==TYV))

    testX=np.expand_dims(mnist.test_X,axis=1)
    testY=mnist.test_Y
    # with open(save_path, "rb") as f:
    #     graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(testX, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    print("test acc", np.average(haty==testY))
    testX=testX.reshape(testX.shape[0],28,28)
    plt.figure(figsize=(20, 20))
    for _ in range(testX.shape[0]):
        ax=plt.subplot(10, 10, _+1)
        plt.imshow(testX[_], cmap='gray')
        ax.text(0,0,haty[_])
        plt.axis('off')
    plt.show()