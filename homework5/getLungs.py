"""
@Project ：CV 
@File    ：getLungs.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/28 15:08
 通过二值化分割、图像反转、连通分量标记、联通分量属性提取、连通分量排序、形态学处理去除空洞，最终从肺部CT图像中提取肺部区域，但不能用区域增长算法
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from skimage import measure
import operator

CT = cv2.imread('chest.PNG', 0)
# 二值化（图像分割）
ret, thresh = cv2.threshold(CT, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # ret为阈值,thresh为二值化后的图像，使用OTSU算法自动计算阈值
# cv2.imwrite('thresh.png', thresh)
# print(ret)
# 图像反转
thresh1 = thresh  # 简单备份中间过程以便后面展示
thresh = cv2.bitwise_not(thresh)  # 二值化后的图像反转，这个方法是按位取反
# cv2.imwrite('afterInvert.png', thresh)
# 连通分量标记
rst_labels, num_labels = measure.label(thresh, connectivity=2, return_num=True)  # 连通分量标记，返回标记后的图像和连通分量个数
# print(num_labels)
# 连通分量属性提取
props = measure.regionprops(rst_labels)  # 连通分量属性提取，返回一个列表，列表中每个元素是一个连通分量的属性
# 排序
props = sorted(props, key=operator.itemgetter('area'), reverse=True)  # 按连通分量面积从大到小排序
# 去除面积最大的连通分量和其他小的连通分量，只保留第二大的和第三大的连通分量，因为第一大的是身体外面的一圈背景
rst = np.zeros_like(thresh)
rst[rst_labels == props[1].label] = 255
rst[rst_labels == props[2].label] = 255
# cv2.imwrite('rst.png', rst)
# 形态学处理，去除空洞，用cv2自带的方法，闭操作
# 结构元素定义
# 我觉得合适的效果，结构元素尺寸为7*7，两条大沟壑留着，小的空洞去掉了
kernel = cv2.getStructuringElement(0, (7, 7))  # 定义结构元素，0表示矩形结构元素，(3,3)表示结构元素的大小
lungs = cv2.morphologyEx(rst, cv2.MORPH_CLOSE, kernel)  # cv2.MORPH_CLOSE表示闭操作
lungs7 = lungs  # 简单备份中间过程以便后面展示
cv2.imwrite('lungs.png', lungs)
# PPT里的效果，结构元素尺寸为50*50
kernel = cv2.getStructuringElement(0, (50, 50))
lungs = cv2.morphologyEx(rst, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('lungsPPT.png', lungs)
# 把原图、分割后的图、翻转后的图、未去除空洞的图、去除空洞后结构元素尺寸为7*7的图、去除空洞后结构元素尺寸为50*50的图画在一起
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(CT, cmap='gray')
plt.title('Original', fontsize=10)
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(thresh1, cmap='gray')
plt.title('After segmentation', fontsize=10)
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('After invert', fontsize=10)
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(rst, cmap='gray')
plt.title('Remove other compoents', fontsize=10)
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(lungs7, cmap='gray')
plt.title('kernel size=7*7', fontsize=10)
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(lungs, cmap='gray')
plt.title('kernel size=50*50', fontsize=10)
plt.axis('off')
plt.show()


