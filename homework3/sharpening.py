"""
@Project ：CV 
@File    ：sharpening.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/14 16:42
 找一张灰度噪声图像（也可以自己模拟生成噪声），进行如下处理
（2）锐化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

test_img = cv2.imread('images/test.png')  # 原始图像
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图

# 模糊图像
blur_img = cv2.GaussianBlur(test_img, (3, 3), 0)

# 锐化图像
# 1.sobel算子
sobel_x = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_img = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sharp_img = cv2.addWeighted(blur_img.astype(np.int16), 1.5, sobel_img.astype(np.int16), -0.5, 0)
sharp_img = np.clip(sharp_img, 0, 255).astype('uint8')
plt.subplot(2, 2, 1), plt.imshow(test_img, cmap='gray'), plt.title('original', fontsize=10)
plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(blur_img, cmap='gray'), plt.title('blur', fontsize=10)
plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(sobel_img, cmap='gray'), plt.title('sobel', fontsize=10)
plt.axis('off')
plt.subplot(2, 2, 4), plt.imshow(sharp_img, cmap='gray'), plt.title('sharp', fontsize=10)
plt.axis('off')
plt.savefig('images/sobel_sharpening.png', dpi=400, bbox_inches='tight')
plt.close()

# 2.laplacian算子
laplacian_img = cv2.Laplacian(blur_img, cv2.CV_64F)
sharp_img = cv2.addWeighted(blur_img.astype(np.int16), 1.5, laplacian_img.astype(np.int16), -0.5, 0)
sharp_img = np.clip(sharp_img, 0, 255).astype('uint8')
plt.subplot(2, 2, 1), plt.imshow(test_img, cmap='gray'), plt.title('original', fontsize=10)
plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(blur_img, cmap='gray'), plt.title('blur', fontsize=10)
plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(laplacian_img, cmap='gray'), plt.title('laplacian', fontsize=10)
plt.axis('off')
plt.subplot(2, 2, 4), plt.imshow(sharp_img, cmap='gray'), plt.title('sharp', fontsize=10)
plt.axis('off')
plt.savefig('images/laplacian_sharpening.png', dpi=400, bbox_inches='tight')
plt.close()


