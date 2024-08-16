'''
 回归。计算accuracy。
'''
import numpy as np
from numpy.random import randn
import mnist
import pickle
from util import setseed

# lr = 0.15  # 学习率
# wd = 0.02  # l2正则化项系数

def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    output=X@weight.reshape(-1,1)+bias
    output=output.reshape(-1)
    return output

def sigmoid(x):
    if -x>700:
        return 0
    else:
        return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y,lr,wd):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    haty=predict(X,weight,bias)
    vectorized_sigmoid=np.vectorize(sigmoid)
    yf=haty*Y
    p=vectorized_sigmoid(yf) #(n,)
    # loss=-np.mean(np.log(p))
    def get_loss(x):
        if -x>700:
            return -x+np.log(1+np.exp(x))
        else:
            return np.log(1+np.exp(-x))
    vectorized_get_loss=np.vectorize(get_loss)
    loss=np.mean(vectorized_get_loss(yf))+wd*np.linalg.norm(weight)
    tmp=(1-p)*Y #(n,)
    new_weight=weight-lr*(-np.mean(np.transpose(X*tmp[:,np.newaxis]),axis=1)+2*wd*weight)
    new_bias=bias-lr*(-np.mean(tmp))
    return (haty,loss,new_weight,new_bias)

def train(lr,wd):
    setseed(0) # 固定随机数种子以提高可复现性

    save_path = "model/lr.npy"

    X=mnist.trn_X
    Y=2*(mnist.trn_Y == 0) - 1 
    weight=randn(mnist.num_feat)
    bias=np.zeros((1))

    # if __name__ == "__main__":
        # 训练
    best_train_acc = 0
    for i in range(1, 60+1):
        haty, loss, weight, bias = step(X, weight, bias, Y,lr,wd)
        acc = np.average((haty>0).flatten()==(Y>0))
        # print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump((weight, bias), f)
    # 测试
    with open(save_path, "rb") as f:
        weight, bias = pickle.load(f)
    haty = predict(mnist.val_X, weight, bias)
    haty = (haty>0).flatten()
    y = (mnist.val_Y==0)
    # print(f"confusion matrix: TP {np.sum((haty>0)*(y>0))} FN {np.sum((y>0)*(haty<=0))} FP {np.sum((y<=0)*(haty>0))} TN {np.sum((y<=0)*(haty<=0))}")
    # print(f"valid acc {np.average(haty==y):.4f}")
    return np.average(haty==y)

best_com=(0,0,0)
LR=np.arange(0.15,0,-0.001)
WD=np.arange(0.01,0.1,0.01)
for lr in LR:
    b=(0,0,0)
    for wd in WD:
        acc=train(lr,wd)
        if acc>b[2]:
            b=(lr,wd,acc)
    print(b[0],b[1],f"valid acc {b[2]:.4f}")
    if b[2]>best_com[2]:
        best_com=b
print(best_com[0],best_com[1])
print(f"valid acc {best_com[2]:.4f}")