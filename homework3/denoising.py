"""
@Project ：CV 
@File    ：denoising.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/14 15:32
 找一张灰度噪声图像（也可以自己模拟生成噪声），进行如下处理
（1）降噪
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

test_img = cv2.imread('images/test.png')  # 原始图像
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
# cv2.imwrite('images/gray.png', test_img)  # 保存灰度图像

# 模拟生成高斯噪声，使用了skimage库中的random_noise函数
gaussian_noise = random_noise(test_img, mode='gaussian', seed=None, clip=True)  # 高斯噪声
gaussian_noise = np.array(255 * gaussian_noise, dtype='uint8')  # 转化为uint8类型
# cv2.imwrite('images/gaussian_noise.png', gaussian_noise)  # 保存噪声图像
# 降噪
img_median = cv2.medianBlur(gaussian_noise, 3)  # 中值滤波
img_gaussian = cv2.GaussianBlur(gaussian_noise, (3, 3), 0)  # 高斯滤波
img_bilateral = cv2.bilateralFilter(gaussian_noise, 9, 75, 75)  # 双边滤波
plt.subplot(2, 2, 1), plt.imshow(gaussian_noise, 'gray'), plt.title('Gaussian Noise', fontsize=10)  # 子图1，显示高斯噪声图像
plt.axis('off')  # 不显示坐标轴
plt.subplot(2, 2, 2), plt.imshow(img_median, 'gray'), plt.title('Median Filter', fontsize=10)  # 子图2，显示中值滤波后的图像
plt.axis('off')  # 不显示坐标轴
plt.subplot(2, 2, 3), plt.imshow(img_gaussian, 'gray'), plt.title('Gaussian Filter', fontsize=10)  # 子图3，显示高斯滤波后的图像
plt.axis('off')  # 不显示坐标轴
plt.subplot(2, 2, 4), plt.imshow(img_bilateral, 'gray'), plt.title('Bilateral Filter', fontsize=10)  # 子图4，显示双边滤波后的图像
plt.axis('off')  # 不显示坐标轴
plt.savefig('images/gaussian_noise.png', dpi=400, bbox_inches='tight')
plt.close()

# 模拟生成椒盐噪声，使用了skimage库中的random_noise函数
salt_pepper_noise = random_noise(test_img, mode='s&p', seed=None, clip=True)  # 椒盐噪声
salt_pepper_noise = np.array(255 * salt_pepper_noise, dtype='uint8')  # 转化为uint8类型
# cv2.imwrite('images/salt_pepper_noise.png', salt_pepper_noise)  # 保存噪声图像
# 降噪
img_median = cv2.medianBlur(salt_pepper_noise, 3)  # 中值滤波
img_gaussian = cv2.GaussianBlur(salt_pepper_noise, (3, 3), 0)  # 高斯滤波
img_bilateral = cv2.bilateralFilter(salt_pepper_noise, 9, 75, 75)  # 双边滤波
plt.subplot(2, 2, 1), plt.imshow(salt_pepper_noise, 'gray'), plt.title('Salt & Pepper Noise', fontsize=10)  # 子图1，显示椒盐噪声图像
plt.axis('off')  # 不显示坐标轴
plt.subplot(2, 2, 2), plt.imshow(img_median, 'gray'), plt.title('Median Filter', fontsize=10)  # 子图2，显示中值滤波后的图像
plt.axis('off')  # 不显示坐标轴
plt.subplot(2, 2, 3), plt.imshow(img_gaussian, 'gray'), plt.title('Gaussian Filter', fontsize=10)  # 子图3，显示高斯滤波后的图像
plt.axis('off')  # 不显示坐标轴
plt.subplot(2, 2, 4), plt.imshow(img_bilateral, 'gray'), plt.title('Bilateral Filter', fontsize=10)  # 子图4，显示双边滤波后的图像
plt.axis('off')  # 不显示坐标轴
plt.savefig('images/salt_pepper_noise.png', dpi=400, bbox_inches='tight')
plt.close()
