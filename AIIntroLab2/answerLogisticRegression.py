import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.1  # 学习率
wd = 0.09  # l2正则化项系数


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


def step(X, weight, bias, Y):
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
