import numpy as np
import mnist
import matplotlib.pyplot as plt
import copy
from numpy.random import randn

X=mnist.trn_X

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
    return rotated_image.astype(np.uint8)

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
    return scaled_image.astype(np.uint8)

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
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def Generator(image_vector):
    old_image=recover_image(image_vector)
    angle=np.random.randint(-10,10)
    new_image=rotate_image(old_image,angle)
    shift=np.random.randint(-2,2,size=2)
    new_image=shift_image(new_image,shift)
    scaling=np.random.uniform(0.9,1.1)
    new_image=scale_image(new_image,scaling)
    noise_level = np.random.randint(5, 10)
    new_image = add_noise(new_image, noise_level)
    return new_image.reshape(-1)

tmp=copy.deepcopy(X[0])

old_image=recover_image(tmp)
angle=np.random.randint(-10,10)
new_image=rotate_image(old_image,angle)
shift=np.random.randint(-2,2,size=2)
new_image=shift_image(new_image,shift)
scaling=np.random.uniform(0.9,1.1)
new_image=scale_image(new_image,scaling)
noise_level = np.random.randint(5, 10)
new_image = add_noise(new_image, noise_level)

print(tmp)

plt.figure(figsize=(10, 10))
plt.subplot(5, 5, 0+1)
plt.imshow(new_image, cmap='gray')
plt.axis('off')
plt.show()