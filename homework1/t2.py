"""
@Project ：CV 
@File    ：t2.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/2/28 16:01
 第二题，将图片resize为224*224，保存为test_resize.tif
"""

import cv2

img = cv2.imread('test.png')
img_resize = cv2.resize(img, (224, 224))
cv2.imwrite('test_resize.tif', img_resize)
