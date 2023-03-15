"""
@Project ：CV
@File    ：t1.py
@IDE     ：PyCharm
@Author  ：32005231文萧寒
@Date    ：2023/2/28 16:00
 第一题，将图片转化为灰度图，保存为test_gray.png
"""

import cv2

img = cv2.imread('test.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('test_gray.png', img_gray)
