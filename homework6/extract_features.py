"""
@Project ：CV 
@File    ：extract_features.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/4/4 15:44
    提供: 肺结节三维数据，包含原始图像和对应的分割mask图像
    要求:
    1）调用pyradiomics库提取图像中病灶的影像组学特征；
    2）基于影像组学特征，训练机器学习模型（任选），预测肺结节良恶性；
    提交：
    1）提取影像组学的py脚本以及提取好的影像组学特征（保存为csv格式文件）；
    2）机器学习训练和测试代码，一个py文件执行训练，一个py文件执行性能评估。
    3）PPT简单汇报采用的评估指标，至少包含accuracy、precision、recall以及ROC曲线图。

本文件用于提取训练集和测试集的影像组学特征
训练模型和评估模型的代码见train_and_evaluate.ipynb
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

benign_train_path = '/Users/wxh/nodule_malignancy_db/train_set/benign'
malignant_train_path = '/Users/wxh/nodule_malignancy_db/train_set/malignant'
benign_test_path = '/Users/wxh/nodule_malignancy_db/test_set/benign'
malignant_test_path = '/Users/wxh/nodule_malignancy_db/test_set/malignant'


# 路径中包含原始图像和对应的分割mask图像，文件名示例：
# 1.3.6.1.4.1.14519.5.2.1.6279.6001.135512138260421166189966386929_402_376_362_mask.npy
# 1.3.6.1.4.1.14519.5.2.1.6279.6001.135512138260421166189966386929_402_376_362_raw.npy


# 定义函数，用于加载图像和掩码
def load_image_and_mask(image_path, mask_path):
    image = np.load(image_path)
    mask = np.load(mask_path)
    return image, mask


# 加载良性训练数据
benign_train_images = []
benign_train_masks = []
for file_name in os.listdir(benign_train_path):
    if file_name.endswith('_raw.npy'):
        image_path = os.path.join(benign_train_path, file_name)
        mask_path = image_path.replace('_raw.npy', '_mask.npy')
        image, mask = load_image_and_mask(image_path, mask_path)
        benign_train_images.append(image)
        benign_train_masks.append(mask)

# 加载恶性训练数据
malignant_train_images = []
malignant_train_masks = []
for file_name in os.listdir(malignant_train_path):
    if file_name.endswith('_raw.npy'):
        image_path = os.path.join(malignant_train_path, file_name)
        mask_path = image_path.replace('_raw.npy', '_mask.npy')
        image, mask = load_image_and_mask(image_path, mask_path)
        malignant_train_images.append(image)
        malignant_train_masks.append(mask)

# 加载良性测试数据
benign_test_images = []
benign_test_masks = []
for file_name in os.listdir(benign_test_path):
    if file_name.endswith('_raw.npy'):
        image_path = os.path.join(benign_test_path, file_name)
        mask_path = image_path.replace('_raw.npy', '_mask.npy')
        image, mask = load_image_and_mask(image_path, mask_path)
        benign_test_images.append(image)
        benign_test_masks.append(mask)

# 加载恶性测试数据
malignant_test_images = []
malignant_test_masks = []
for file_name in os.listdir(malignant_test_path):
    if file_name.endswith('_raw.npy'):
        image_path = os.path.join(malignant_test_path, file_name)
        mask_path = image_path.replace('_raw.npy', '_mask.npy')
        image, mask = load_image_and_mask(image_path, mask_path)
        malignant_test_images.append(image)
        malignant_test_masks.append(mask)


# 配置特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor('exampleCT.yaml')

# 提取良性训练数据的特征
benign_train_features = []
for image, mask in zip(benign_train_images, benign_train_masks):
    # 转化成sitk格式
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    # 提取特征
    features = extractor.execute(image, mask)
    # 显示进度
    print('benign', len(benign_train_features), '/', len(benign_train_images))
    benign_train_features.append(features)

# 提取恶性训练数据的特征
malignant_train_features = []
for image, mask in zip(malignant_train_images, malignant_train_masks):
    # 转化成sitk格式
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    # 提取特征
    features = extractor.execute(image, mask)
    # 显示进度
    print('malignant', len(malignant_train_features), '/', len(malignant_train_images))
    malignant_train_features.append(features)

# 提取良性测试数据的特征
benign_test_features = []
for image, mask in zip(benign_test_images, benign_test_masks):
    # 转化成sitk格式
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    # 提取特征
    features = extractor.execute(image, mask)
    # 显示进度
    print('benign', len(benign_test_features), '/', len(benign_test_images))
    benign_test_features.append(features)

# 提取恶性测试数据的特征
malignant_test_features = []
for image, mask in zip(malignant_test_images, malignant_test_masks):
    # 转化成sitk格式
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    # 提取特征
    features = extractor.execute(image, mask)
    # 显示进度
    print('malignant', len(malignant_test_features), '/', len(malignant_test_images))
    malignant_test_features.append(features)

# 将特征保存为CSV文件
train_df = pd.DataFrame(benign_train_features + malignant_train_features)
train_df['Label'] = [0] * len(benign_train_features) + [1] * len(malignant_train_features)
train_df.to_csv('./data/train_features.csv', index=False)
test_df = pd.DataFrame(benign_test_features + malignant_test_features)
test_df['Label'] = [0] * len(benign_test_features) + [1] * len(malignant_test_features)
test_df.to_csv('./data/test_features.csv', index=False)


