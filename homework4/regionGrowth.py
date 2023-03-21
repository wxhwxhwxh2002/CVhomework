"""
@Project ：CV 
@File    ：regionGrowth.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/21 15:25
 实现区域生长算法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image


# img = cv2.imread('images/test.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# visited = np.zeros(img.shape, dtype=np.uint8)  # 访问标记
# out_mask = np.zeros(img.shape, dtype=np.uint8)  # 输出区域标记
# neighbor = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 四邻域
# seed = (100, 300)  # 种子点
# dis_thres = 30  # 阈值
# seeds = [seed]  # 种子点入队
# while len(seeds) > 0:
#     print(len(seeds))
#     cur_seed = seeds.pop(0)  # 队首元素出队
#     print(cur_seed)
#     if visited[cur_seed] == 1:
#         continue
#     visited[cur_seed] = 1  # 标记为已访问
#     out_mask[cur_seed] = 1  # 标记为前景
#     mean = np.mean(img[out_mask == 1])  # 计算当前区域的均值
#     for i in range(4):  # 四邻域
#         # print(neighbor[i])
#         next_seed = (cur_seed[0] + neighbor[i][0], cur_seed[1] + neighbor[i][1])
#         print(next_seed)
#         # 判断是否越界
#         if next_seed[0] < 0 or next_seed[0] >= img.shape[0] or next_seed[1] < 0 or next_seed[1] >= img.shape[1]:
#             continue
#         # 判断是否已访问
#         if visited[next_seed] == 1:
#             continue
#         # 判断是否满足阈值条件
#         if abs(img[next_seed] - mean) > dis_thres:
#             continue
#         seeds.append(next_seed)  # 满足条件的种子点入队
#
# plt.imshow(out_mask, cmap='gray')
# plt.savefig('images/regionGrowth.png')


def regionGrowth(img, seed, dis_thres):
    print(img.shape)
    print(seed)
    print(dis_thres)
    visited = np.zeros(img.shape, dtype=np.uint8)  # 访问标记
    out_mask = np.zeros(img.shape, dtype=np.uint8)  # 输出区域标记
    neighbor = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 四邻域
    seeds = [seed]  # 种子点入队
    while len(seeds) > 0:
        print(len(seeds))
        cur_seed = seeds.pop(0)  # 队首元素出队
        print(cur_seed)
        if visited[cur_seed] == 1:
            continue
        visited[cur_seed] = 1  # 标记为已访问
        out_mask[cur_seed] = 1  # 标记为前景
        mean = np.mean(img[out_mask == 1])  # 计算当前区域的均值
        for i in range(4):  # 四邻域
            # print(neighbor[i])
            next_seed = (cur_seed[0] + neighbor[i][0], cur_seed[1] + neighbor[i][1])
            # print(next_seed)
            # 判断是否越界
            if next_seed[0] < 0 or next_seed[0] >= img.shape[0] or next_seed[1] < 0 or next_seed[1] >= img.shape[1]:
                continue
            # 判断是否已访问
            if visited[next_seed] == 1:
                continue
            # 判断是否满足阈值条件
            if abs(img[next_seed] - mean) > dis_thres:
                continue
            seeds.append(next_seed)  # 满足条件的种子点入队
    return out_mask


if __name__ == '__main__':
    img = cv2.imread('images/test.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seed = (100, 300)  # 种子点
    dis_thres = 20  # 阈值
    out_mask = regionGrowth(img, seed, dis_thres)
    plt.imshow(out_mask, cmap='gray')
    plt.savefig('images/regionGrowth111.png')
