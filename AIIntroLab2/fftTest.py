from autograd.BaseNode import Convolution,convolution
import numpy as np
from numpy.lib.stride_tricks import as_strided
import math
# x = np.array([[1, 0, 2, 1],
#               [0, 1, 3, 0],
#               [1, 1, 2, 1],
#               [0, 1, 3, 0]],dtype=np.int64)
# output = as_strided(x, shape=(3, 3, 2, 2), strides=(32, 8, 32, 8))
# kernel = np.array([[0, 1],
#                    [2, 0]],dtype=np.int64)
# result = np.tensordot(output, kernel, axes=((2, 3), (0, 1)))
# print(result)


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

node1=Convolution((1,1,2,2))
node2=convolution((1,1,2,2))
node1.params[0]=np.array([[[[1,0],
                           [1,0]]]])
node2.params[0]=np.array([[[[1,0],
                           [1,0]]]])
x=np.array([[[[0,1,2],
              [3,4,5],
              [6,7,8]]]])
ret1=node1.cal(x)
ret2=node2.cal(x)
grad=np.array([[[[1,1],
                 [1,1]]]])
grad1=node1.backcal(grad)
grad2=node2.backcal(grad)
# print(grad1)
# print(grad2)
print(node1.grad)
print(node2.grad)