"""
@Project ：CV 
@File    ：image_utils.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/7 18:19
 实现图像灰度变换及直方图均衡化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 转化为灰度图
def gray(img):
    # # resize为512*512
    # img = cv2.resize(img, (512, 512))
    # 转化为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# 绘制图像直方图
def draw_hist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    H, W = img.shape
    hist = np.zeros(256, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            hist[img[i, j]] += 1
    # 绘制直方图
    plt.plot(hist)
    plt.title('histogram')
    plt.xlabel('pixel value')
    plt.ylabel('number of pixels')
    plt.savefig('images/hist.png')
    plt.close()


# 直方图均衡化
def equalize_hist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    H, W = img.shape
    hist = np.zeros(256, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            hist[img[i, j]] += 1
    gray_hist = hist.copy()
    # 累积
    s = np.zeros(256, dtype=np.float32)
    sum = 0
    for i in range(256):
        sum += hist[i]
        s[i] = sum
    for i in range(H):
        for j in range(W):
            img[i, j] = s[img[i, j]] * 255 / (H * W)
    hist = np.zeros(256, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            hist[img[i, j]] += 1
    # 四张子图，分别为原图的灰度图、直方图、均衡化后的图、均衡化后的直方图
    plt.subplot(221)
    plt.imshow(gray, cmap='gray')
    plt.title('original image')
    plt.subplot(222)
    plt.plot(gray_hist)
    plt.title('histogram')
    plt.xlabel('pixel value')
    plt.ylabel('number of pixels')
    plt.subplot(223)
    plt.imshow(img, cmap='gray')
    plt.title('equalized image')
    plt.subplot(224)
    plt.plot(hist)
    plt.title('equalized histogram')
    plt.xlabel('pixel value')
    plt.ylabel('number of pixels')
    plt.savefig('images/equalize_hist.png')
    plt.close()
    return img


# 图像灰度翻转
def reverse(img):
    if len(img.shape) == 3:
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_grey.dtype != np.uint8:
        img_grey = img_grey.astype(np.uint8)
    H, W = img_grey.shape
    for i in range(H):
        for j in range(W):
            img_grey[i, j] = 255 - img_grey[i, j]
    # 生成对比图，左边为原图，右边为翻转后的图
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.title('original image')
    plt.subplot(122)
    plt.imshow(img_grey, cmap='gray')
    plt.title('reverse image')
    plt.savefig('images/reverse.png')
    plt.close()
