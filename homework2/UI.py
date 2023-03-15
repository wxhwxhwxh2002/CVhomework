"""
@Project ：CV 
@File    ：UI.py
@IDE     ：PyCharm 
@Author  ：32005231文萧寒
@Date    ：2023/3/7 15:12
 使用PyQt5实现GUI界面
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from image_utils import *


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.imageToBeProcessed = 'images/test.png'
        self.initUI()
        self.setWindowTitle('作业2 图像灰度变换及直方图均衡化 32005231文萧寒')
        self.setGeometry(300, 300, 500, 500)
        self.resize(700, 550)
        self.show()

    def initUI(self):
        self.label = QLabel(self)
        self.img_label1 = QLabel(self)
        self.img_label1.setScaledContents(True)
        # self.img_label2 = QLabel(self)
        self.img_label1.setPixmap(QPixmap(self.imageToBeProcessed))
        # self.img_label2.setPixmap(QPixmap("test.png"))
        # 设置五个按钮，分别为：打开图片、灰度变换、灰度反转、绘制直方图、直方图均衡化
        self.btn1 = QPushButton('打开图片', self)
        self.btn1.setGeometry(550, 50, 150, 50)
        self.btn1.clicked.connect(self.open_img)
        self.btn2 = QPushButton('灰度变换', self)
        self.btn2.setGeometry(550, 150, 150, 50)
        self.btn2.clicked.connect(self.gray_img)
        self.btn3 = QPushButton('灰度反转', self)
        self.btn3.setGeometry(550, 250, 150, 50)
        self.btn3.clicked.connect(self.reverse_img)
        self.btn4 = QPushButton('绘制直方图', self)
        self.btn4.setGeometry(550, 350, 150, 50)
        self.btn4.clicked.connect(self.draw_hist)
        self.btn5 = QPushButton('直方图均衡化', self)
        self.btn5.setGeometry(550, 450, 150, 50)
        self.btn5.clicked.connect(self.equalize_hist)

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    def open_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        if openfile_name[0]:
            self.imageToBeProcessed = openfile_name[0]
            print(self.imageToBeProcessed)
            self.img_label1.setPixmap(QPixmap(self.imageToBeProcessed))

    def gray_img(self):
        img = cv2.imread(self.imageToBeProcessed)
        img = gray(img)
        cv2.imwrite('images/gray.png', img)
        self.img_label1.setPixmap(QPixmap('images/gray.png'))
        # self.imageToBeProcessed = 'gray.png'
        # print(self.imageToBeProcessed)

    def reverse_img(self):
        img = cv2.imread(self.imageToBeProcessed)
        reverse(img)
        self.img_label1.setPixmap(QPixmap('images/reverse.png'))
        # self.imageToBeProcessed = 'reverse.png'
        # print(self.imageToBeProcessed)

    def draw_hist(self):
        img = cv2.imread(self.imageToBeProcessed)
        draw_hist(img)
        self.img_label1.setPixmap(QPixmap('images/hist.png'))
        # print(self.imageToBeProcessed)

    def equalize_hist(self):
        img = cv2.imread(self.imageToBeProcessed)
        equalize_hist(img)
        self.img_label1.setPixmap(QPixmap('images/equalize_hist.png'))
        # print(self.imageToBeProcessed)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    x = MyWindow()
    x.show()
    sys.exit(app.exec_())
