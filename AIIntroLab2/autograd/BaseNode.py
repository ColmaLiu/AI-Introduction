from typing import List
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from itertools import product
from .Init import * 

def shape(X):
    if isinstance(X, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")

class Node(object):
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储weight和bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass


class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 

class sigmoid(Node):
    # input X: (*)，即可能是任意维度
    # output sigmoid(X): (*)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, X):
        x=np.exp(-X)
        self.cache.append(x)
        return 1/(1+x)

    def backcal(self, grad):
        x=self.cache[-1]
        return grad*x/(1+x)**2
    
class tanh(Node):
    # input X: (*)，即可能是任意维度
    # output tanh(X): (*)
    def __init__(self):
        super().__init__("tanh")

    def cal(self, X):
        ret = np.tanh(X)
        self.cache.append(ret)
        return ret


    def backcal(self, grad):
        return np.multiply(grad, np.multiply(1+self.cache[-1], 1-self.cache[-1]))
    

class Linear(Node):
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (d2)
    # output Linear(X): (*, d2)
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        self.cache.append(X)
        return X@self.params[0]+self.params[1]

    def backcal(self, grad):
        '''
        需要保存weight和bias的梯度，可以参考Node类和BatchNorm类
        '''
        X=self.cache[-1]
        weight,bias=self.params[0],self.params[1]
        self.grad=[None,None]
        self.grad[0]=X.T@grad
        self.grad[1]=np.sum(grad,axis=0)
        return grad@weight.T


class StdScaler(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-3
    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad/ (self.std + self.EPS)
    


class BatchNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout_Corrected(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            X = X * (1/(1-self.p))
            self.cache.append(mask)
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            grad = grad * (1/(1-self.p))
        return grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True

class Dropout(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            self.cache.append(mask)
        else:
            X = X*(1/(1-self.p))
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            return grad
        else:
            return (1/(1-self.p)) * grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True

class Softmax(Node):
    # input X: (*)
    # output softmax(X): (*), softmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)


class LogSoftmax(Node):
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = X-np.log(expX.sum(axis=self.dim, keepdims=True))
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        logSoftMaxX=self.cache[-1]
        return grad-np.multiply(grad.sum(axis=self.dim, keepdims=True),np.exp(logSoftMaxX))




class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret



class CrossEntropyLoss(Node):
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。它与NLLLoss的区别仅在于后者输入log概率，前者输入概率。
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        probs = np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1)
        loss = -np.sum(np.log(probs))
        return loss
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的cal
        raise NotImplementedError

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        probs = np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1 / probs, axis=-1)
        return grad*ret
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的backcal
        raise NotImplementedError



class Convolution(Node):
    """
    单通道卷积层
    no padding
    """
    def __init__(self,kernel_shape:tuple,padding=0,stride=1):
        """
        kernel_shape:(out_channel,in_channel,ker_height,ker_width)
        """
        self.padding=padding
        self.stride=stride
        kernel=rand(*kernel_shape)
        bias=np.zeros(kernel_shape[0])
        super().__init__("Conv",kernel,bias)

    def cal(self,X):
        """
        X:(batch_size,channel,height,width)
        """
        padding,stride=self.padding,self.stride
        kernel,bias=self.params[0],self.params[1]
        # paddedX=np.pad(X, [(padding, padding), (padding, padding)], mode='constant')
        X_shape=np.array(X.shape)
        kernel_shape=np.array(kernel.shape)
        ret_shape=(X_shape[0],kernel_shape[0],(X_shape[-2]+2*padding-kernel_shape[-2])//stride+1,(X_shape[-1]+2*padding-kernel_shape[-1])//stride+1)
        ret=np.zeros(ret_shape)
        for i in range(ret_shape[-2]):
            for j in range(ret_shape[-1]):
                # for oc in range(kernel_shape[0]):
                #     patch=X[:,:,i*stride:i*stride+kernel_shape[-2],j*stride:j*stride+kernel_shape[-1]]
                #     ret[:,oc,i,j]=np.sum(np.multiply(patch,kernel[oc]),axis=(-2,-1))+bias
                patch=X[:,:,i*stride:i*stride+kernel_shape[-2],j*stride:j*stride+kernel_shape[-1]]
                patch=np.expand_dims(patch,axis=1)
                ret[:,:,i,j]=np.sum(np.multiply(patch,kernel),axis=(-3,-2,-1))+bias
        self.cache.append(X)
        return ret
    
    def backcal(self, grad):
        padding,stride=self.padding,self.stride
        kernel,bias=self.params[0],self.params[1]
        X=self.cache[-1]
        X_shape=np.array(X.shape)
        kernel_shape=np.array(kernel.shape)
        self.grad=[None,None]
        self.grad[1]=np.sum(grad,axis=(0,2,3))
        grad_kernel=np.zeros_like(kernel)
        for i in range(grad.shape[-2]):
            for j in range(grad.shape[-1]):
                grad_kernel += np.sum(np.expand_dims(X[:,:,i * stride:i * stride + kernel_shape[-2], j * stride:j * stride + kernel_shape[-1]],axis=1) * grad[:,:,i, j][:,:,np.newaxis,np.newaxis,np.newaxis],axis=0)

        # 计算上一层的梯度
        pad_grad = np.zeros_like(X)
        for i in range(grad.shape[-2]):
            for j in range(grad.shape[-1]):
                pad_grad[:,:,i * stride:i * stride + kernel_shape[-2], j * stride:j * stride + kernel_shape[-1]] += np.sum(kernel * grad[:,:,i, j][:,:,np.newaxis,np.newaxis,np.newaxis],axis=1)
        
        self.grad[0]=grad_kernel
        return pad_grad



class MaxPooling(Node):
    def __init__(self,pool_size=2,stride=2):
        self.pool_size = pool_size
        self.stride = stride
        super().__init__("Pool")

    def cal(self,X):
        """
        X:(batch_size,channel,height,width)
        """
        pool_size, stride = self.pool_size, self.stride
        X_shape=np.array(X.shape)
        ret_shape=(X_shape[0],X_shape[1],(X_shape[-2]-pool_size)//stride+1,(X_shape[-1]-pool_size)//stride+1)
        ret=np.zeros(ret_shape)
        for i in range(ret_shape[-2]):
            for j in range(ret_shape[-1]):
                patch=X[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
                ret[:,:,i,j]=np.max(patch,axis=(-2,-1))
        self.cache.append(X)
        return ret
    
    def backcal(self, grad):
        pool_size, stride = self.pool_size, self.stride
        X=self.cache[-1]
        X_shape=np.array(X.shape)
        ret_shape=(X_shape[0],X_shape[1],(X_shape[-2]-pool_size)//stride+1,(X_shape[-1]-pool_size)//stride+1)
        grad_X=np.zeros_like(X)
        for i in range(ret_shape[-2]):
            for j in range(ret_shape[-1]):
                patch=X[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
                maxValue=np.max(patch,axis=(-2,-1),keepdims=True)
                mask = (patch == maxValue)
                grad_X[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]+=mask*grad[:,:,i,j][:,:,np.newaxis,np.newaxis]
        return grad_X



class Flatten(Node):
    def __init__(self,shape:tuple):
        """
        shape:(out_channel,height,weight)
        """
        self.shape=shape
        super().__init__("Flatten")

    def cal(self,X):
        return X.reshape(X.shape[0],-1)
    
    def backcal(self, grad):
        return grad.reshape(grad.shape[0],*self.shape)
        return grad.reshape(-1,int(np.sqrt(grad.shape[-1])),int(np.sqrt(grad.shape[-1])))
    


class Recover(Node):
    def __init__(self):
        super().__init__("Recover")

    def cal(self,X):
        return X.reshape(-1,28,28)
    
    def backcal(self, grad):
        return grad.reshape(-1,784)



def split_by_strides(x: np.ndarray, kernel_size, stride=(1, 1)):
    """
    将张量按卷积核尺寸与步长进行分割
    :param x: 被卷积的张量
    :param kernel_size: 卷积核的长宽
    :param stride: 步长
    :return: y: 按卷积步骤展开后的矩阵
    """
    *bc, h, w = x.shape
    out_H, out_W = (h - kernel_size[0]) // stride[0] + 1, (w - kernel_size[1]) // stride[1] + 1
    shape = (*bc, out_H, out_W, kernel_size[0], kernel_size[1])
    strides = (*x.strides[:-2], x.strides[-2] * stride[0],
               x.strides[-1] * stride[1], *x.strides[-2:])
    y = as_strided(x, shape, strides=strides)
    return y
def padding_zeros(x: np.ndarray, padding):
    """
    在张量周围填补0
    @param x: 需要被padding的张量,ndarray类型
    @param padding: 一个二元组,其每个元素也是一个二元组,分别表示竖直、水平方向需要padding的层数
    @return: padding的结果
    """
    if padding == ((0, 0), (0, 0)):
        return x
    n = x.ndim - 2
    x = np.pad(x, ((0, 0),) * n + padding, 'constant', constant_values=0)
    return x
def unwrap_padding(x: np.ndarray, padding):
    """
    padding的逆操作
    @param x:
    @param padding:
    @return:
    """
    if padding == ((0, 0), (0, 0)):
        return x
    p, q = padding
    if p == (0, 0):
        return x[..., :, q[0]:-q[1]]
    if q == (0, 0):
        return x[..., p[0]:-p[1], :]
    return x[..., p[0]:-p[1], q[0]:-q[1]]
def dilate(x: np.ndarray, dilation=(0, 0)):
    """
    膨胀,在各行、列间插入一定数量的0
    """
    if dilation == (0, 0):
        return x
    *bc, h, w = x.shape
    y = np.zeros((*bc, (h - 1) * (dilation[0] + 1) + 1, (w - 1) * (dilation[1] + 1) + 1),dtype=np.float64)
    y[..., ::dilation[0] + 1, ::dilation[1] + 1] = x
    return y
def erode(x: np.ndarray, dilation=(0, 0)):
    """
    腐蚀,与膨胀互为逆运算
    """
    if dilation == (0, 0):
        return x
    y = x[..., ::dilation[0] + 1, ::dilation[1] + 1]
    return y
def rotate180(kernel: np.ndarray, axis=(-1, -2)):
    return np.flip(kernel, axis)
class convolution(Node):
    def __init__(self,kernel_shape:tuple,padding=0):
        self.padding=padding
        kernel=np.random.rand(*kernel_shape)
        bias=np.zeros(kernel_shape[0])
        super().__init__("conv",kernel,bias)
    
    def cal(self,X):
        paddedX=np.pad(X,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        split=split_by_strides(paddedX,self.params[0].shape[-2:])
        ret=np.tensordot(split,self.params[0],axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2))
        self.cache.append(X)
        return ret

    def backcal(self, grad):
        X=self.cache[-1]
        self.grad=[None,None]
        self.grad[1]=np.sum(grad,axis=(0,-2,-1))
        paddedX=np.pad(X,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        s=split_by_strides(paddedX,grad.shape[-2:])
        grad_kernel=np.tensordot(s,grad,axes=[(0, 4, 5), (0, 2, 3)]).transpose((3, 0, 1, 2))
        self.grad[0]=grad_kernel

        padding=self.padding
        p=(self.params[0].shape[-2]-1,self.params[0].shape[-1]-1)
        padded_grad=np.pad(grad,((0,0),(0,0),(p[0],p[0]),(p[1],p[1])), mode='constant')
        s=split_by_strides(padded_grad,self.params[0].shape[-2:])
        grad_X=unwrap_padding(np.tensordot(s,rotate180(self.params[0]),axes=[(1, 4, 5), (0, 2, 3)]).transpose((0, 3, 1, 2)),((padding,padding),(padding,padding)))
        return grad_X



class maxPooling(Node):
    def __init__(self,pool_size=2,stride=2):
        self.pool_size = pool_size
        self.stride = stride
        super().__init__("pool")

    def cal(self,X):
        split=split_by_strides(X,(self.pool_size,self.pool_size),(self.stride,self.stride))
        max_data=np.max(split,axis=(-2,-1))
        self.cache.append(X)
        return max_data

    def backcal(self, grad):
        pool_size, stride = self.pool_size, self.stride
        X=self.cache[-1]
        X_shape=np.array(X.shape)
        ret_shape=(X_shape[0],X_shape[1],(X_shape[-2]-pool_size)//stride+1,(X_shape[-1]-pool_size)//stride+1)
        grad_X=np.zeros_like(X)
        for i in range(ret_shape[-2]):
            for j in range(ret_shape[-1]):
                patch=X[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
                maxValue=np.max(patch,axis=(-2,-1),keepdims=True)
                mask = (patch == maxValue)
                grad_X[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]+=mask*grad[:,:,i,j][:,:,np.newaxis,np.newaxis]
        return grad_X



class batchNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, *self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, *self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True