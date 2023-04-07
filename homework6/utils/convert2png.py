import numpy as np
import imageio
import os
import os.path as osp
import cv2


def hu2gray(volume, WL=-600, WW=900):
    """
	convert HU value to gray scale[0,255] using lung-window(WL/WW=-500/1200)
	"""

    low = WL - 0.5 * WW
    volume = (volume - low) / WW * 255.0
    volume[volume > 255] = 255
    volume[volume < 0] = 0
    volume = np.uint8(volume)
    return volume


def vis_edges(im, mask):
    vis_im = np.zeros(im.shape + (3,))
    vis_im[..., 0] = im
    vis_im[..., 1] = im
    vis_im[..., 2] = im

    lobe_colors = [(255, 0, 0), (0, 255, 0)]
    for k in range(1, 2):
        tmp_mask = np.zeros_like(mask, dtype=np.uint8)
        tmp_mask[mask == k] = 1

        h = cv2.findContours(tmp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print('counters:', h)
        edge = h[0]

        # print('edge:',edge)
        cv2.drawContours(vis_im, edge, -1, lobe_colors[k - 1], 1)

    return np.uint8(vis_im)


import glob

subset = 'test_set'
target = 'benign'

save_dir = osp.join('viz_pngs', subset, target)
if not osp.exists(save_dir):
    os.makedirs(save_dir)

raw_files = glob.glob(osp.join(subset, target, '*raw.npy'))

for f in raw_files:
    npy_data = np.load(f)
    print(npy_data.shape)
    mask_file = f.replace('raw', 'mask')
    if not osp.exists(mask_file):
        continue
    mask_data = np.load(mask_file)

    save_path = osp.join(save_dir, osp.basename(f))
    if not osp.exists(save_path):
        os.mkdir(save_path)

    # 将每一层图像都可视化出来，保存为png图像
    for z in range(npy_data.shape[0]):
        im = hu2gray(npy_data[z, ...], WL=-500, WW=1000)  # 将CT值转为[0,255]之间的灰度值
        mask = mask_data[z, :, :]

        vis_im = vis_edges(im, mask)
        imageio.imsave(osp.join(save_path, str(z + 1) + '_edge.png'), vis_im)
        imageio.imsave(osp.join(save_path, str(z + 1) + '_mask.png'), mask * 255)
