"""
@Project ：CV 
@File    ：UI.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/21 16:40
 简单的UI界面
"""
import sys

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox, \
    QProgressDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from regionGrowth import regionGrowth

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('作业4 区域生长算法 32005231文萧寒')
        self.setGeometry(100, 100, 600, 600)
        self.seeds = []
        # 默认图片
        self.img = cv2.imread('images/test.png', cv2.IMREAD_GRAYSCALE)
        self.img_height = 512
        self.img_width = 512
        self.img = cv2.resize(self.img, (self.img_width, self.img_height))
        self.qimg = QImage(self.img.data, self.img_width, self.img_height, self.img_width, QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(self.qimg)
        self.label_img = QLabel(self)
        self.label_img.setPixmap(self.pixmap)
        self.label_img.resize(self.img_width, self.img_height)

        # 选择图片按钮
        self.btn_select_img = QPushButton('选择图片', self)
        self.btn_select_img.move(10, self.img_height + 20)
        self.btn_select_img.clicked.connect(self.selectImg)

        # 生成按钮
        self.btn_generate = QPushButton('生成', self)
        self.btn_generate.move(310, self.img_height + 20)
        self.btn_generate.clicked.connect(self.onGenerate)

        # 阈值标签
        self.label_threshold = QLabel('请输入阈值:', self)
        self.label_threshold.move(120, self.img_height + 20)

        # 阈值输入框
        self.edit_threshold = QLineEdit(self)
        self.edit_threshold.move(200, self.img_height + 20)
        self.edit_threshold.setText('20')

        # 标签显示当前选定种子点坐标，默认为"请点击图片"
        self.label_seed = QLabel('请点击图片以选定种子点', self)
        self.label_seed.move(10, self.img_height + 50)
        self.label_seed.setFixedWidth(300)


    def selectImg(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files (*.jpg *.png *.bmp)')
        if fname:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_width, self.img_height))
            self.img = img
            cv2.imwrite('temp.png', img)
            self.qimg = QImage(img.data, self.img_width, self.img_height, self.img_width, QImage.Format_Grayscale8)
            self.pixmap = QPixmap.fromImage(self.qimg)
            self.label_img.setPixmap(self.pixmap)
            # 去除已有的标红的种子点
            for child in self.children():
                if isinstance(child, QLabel) and child.styleSheet() == 'background-color:red':
                    child.deleteLater()
            # 清空种子点列表
            self.seeds.clear()
            # 清空标签
            self.label_seed.setText('请点击图片以选定种子点')

    def onGenerate(self):
        # 如果阈值输入框为空，就不进行处理，并且提示用户
        if self.edit_threshold.text() == '':
            QMessageBox.warning(self, '警告', '请输入阈值')
            return
        threshold = int(self.edit_threshold.text())
        if len(self.seeds) == 0:
            QMessageBox.warning(self, '警告', '请选定种子点')
            return
        progress_dialog = QProgressDialog("正在生成...", None, 0, 0, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()
        for seed in self.seeds:
            x, y = seed.x(), seed.y()
        seed = (y, x)  # 注意这里的坐标是反的
        out_mask = regionGrowth(self.img, seed, threshold)
        out_mask = out_mask * 255
        out_mask = out_mask.astype(np.uint8)
        self.qimg = QImage(out_mask.data, self.img_width, self.img_height, self.img_width, QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(self.qimg)
        self.label_img.setPixmap(self.pixmap)
        progress_dialog.close()





    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 显示原图
            # cv2.imwrite('temp.png', self.img)
            self.qimg = QImage(self.img.data, self.img_width, self.img_height, self.img_width, QImage.Format_Grayscale8)
            self.pixmap = QPixmap.fromImage(self.qimg)
            self.label_img.setPixmap(self.pixmap)
            # 去除已有的标红的种子点
            for child in self.children():
                if isinstance(child, QLabel) and child.styleSheet() == 'background-color:red':
                    child.deleteLater()
            # 获取鼠标点击的坐标
            pos = event.pos()
            x, y = pos.x(), pos.y()
            # 如果点击在图片内部
            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                # 绘制种子点
                label_seed = QLabel(self)
                # 把已有的标红的种子点清除

                label_seed.setGeometry(x - 3, y - 3, 6, 6)
                label_seed.setStyleSheet('background-color:red')
                label_seed.show()
                # 清空种子点
                self.seeds.clear()
                # 记录种子点
                self.seeds.append(QPoint(x, y))
                # 显示当前选定种子点坐标
                self.label_seed.setText(f'当前选定种子点坐标：({y}, {x})')  # 注意这里的坐标是反的
                print(self.seeds)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())