"""
@Project ：CV 
@File    ：t3.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/2/28 16:02
 第三题，将图片转化为HSI图像，使用matplotlib显示原始图像、H图像、S图像、I图像
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

RGB_img = cv2.imread('test.png')
# 记录图像的高、宽、通道数
row, col, channel = RGB_img.shape
# 创建HSI图像
HSI_img = np.zeros((row, col, channel), dtype=np.float32)
# 分离通道
B, G, R = cv2.split(RGB_img)
# 归一化
B = B / 255.0
G = G / 255.0
R = R / 255.0
# 创建H、S、I通道
H = np.zeros((row, col), dtype=np.float32)
S = np.zeros((row, col), dtype=np.float32)
I = np.zeros((row, col), dtype=np.float32)
# 计算I通道
I = (B + G + R) / 3.0
I = I * 255
# 计算S通道
S = 1 - 3 * np.minimum(np.minimum(B, G), R) / (B + G + R)
S = S * 255
# 计算H通道
theta = np.arccos((2 * R - G - B) / (2 * np.sqrt((R - G) ** 2 + (R - B) * (G - B))))
for i in range(row):
    for j in range(col):
        if B[i, j] <= G[i, j]:
            H[i, j] = theta[i, j]
        else:
            H[i, j] = 2 * np.pi - theta[i, j]
# 将H通道转换到0~255
H = H * 255 / (2 * np.pi)
# 将H、S、I通道转换为uint8类型
H = H.astype(np.uint8)
S = S.astype(np.uint8)
I = I.astype(np.uint8)
# 合并通道
HSI_img = cv2.merge([H, S, I])
# 保存HSI图像
cv2.imwrite('HSI.png', HSI_img)
# # 使用opencv自带的函数转换为HSI图像
# HSI_img_opencv = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2HSV)
# # 保存HSI图像
# cv2.imwrite('HSI_opencv.png', HSI_img_opencv)

# 使用matplotlib显示原始图像、H图像、S图像、I图像
plt.subplot(2, 2, 1)
plt.imshow(RGB_img)
plt.title('RGB')
plt.subplot(2, 2, 2)
plt.imshow(HSI_img[:, :, 0], cmap='gray')
plt.title('H')
plt.subplot(2, 2, 3)
plt.imshow(HSI_img[:, :, 1], cmap='gray')
plt.title('S')
plt.subplot(2, 2, 4)
plt.imshow(HSI_img[:, :, 2], cmap='gray')
plt.title('I')
plt.show()
# plt.savefig('compare.png')
# # 使用opencv自带的函数把HSI图像转换为RGB图像
# RGB_img_opencv = cv2.cvtColor(HSI_img, cv2.COLOR_HSV2BGR)
# # 保存RGB图像
# cv2.imwrite('RGB_opencv.png', RGB_img_opencv)


