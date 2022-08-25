# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import time

import numpy as np
import torch
from PIL import Image
import matplotlib.image as mplimg
import cv2 as cv
from torchvision import transforms

from MyCnn import MyCnn
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene
from matplotlib import pyplot as plt
from pyqt5_plugins.examplebutton import QtWidgets
from pyqt5_plugins.examplebuttonplugin import QtGui

from Ui_my_first import Ui_auto_adjust_tool
from test_my_net import adjust_tool

global_file_path = ''


class MainWindow(QMainWindow, Ui_auto_adjust_tool):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget (defaults to None)
        @type QWidget (optional)
        """
        super().__init__(parent)
        self.setupUi(self)
        self.actionopen.setShortcut('alt+O')

    # @pyqtSlot()
    # def on_pushButton_clicked(self):
    #     """
    #     Slot documentation goes here.
    #     """
    #     # TODO: not implemented yet

    @pyqtSlot()
    def on_actionopen_triggered(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        try:
            file_name, file_type = QFileDialog.getOpenFileNames(self, "选取文件", "./",
                                                                filter="csv Files (*.csv);;All Files (*)")
            print("文件名：")
            global global_file_path
            global_file_path = ''.join(file_name)
            print(global_file_path)
            one_csv_data = np.loadtxt(global_file_path)
            new_size = (512, 512)
            one_csv_data = cv.resize(one_csv_data, new_size)
            global_file_path = global_file_path[:-3] + 'png'
            mplimg.imsave(global_file_path, one_csv_data, cmap='gray')
            pix = QPixmap(global_file_path)
            print("pix is ok")
            self.label_ori.setPixmap(pix)
            self.label_tip.setText("图片加载成功")
        except:
            pass

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        # print(global_file_path)
        global global_file_path
        global_file_path = global_file_path[:-3] + 'csv'
        img_lab = int(global_file_path[-7:-6]) * 4 + int(global_file_path[-6:-5]) * 2 + int(global_file_path[-5:-4])
        print(img_lab)
        # 准备输入神经网络的数据
        img_data = np.loadtxt(global_file_path).astype(np.float32)
        trans_totensor = transforms.ToTensor()
        img_tensor = trans_totensor(img_data)
        # print(img_tensor.shape)
        img_tensor = torch.reshape(img_tensor, (1, 1, 128, 128))
        # 创建网络模型
        my_cnn = MyCnn()
        my_cnn.load_state_dict(torch.load(("my_cnn.pth")))
        # print(my_cnn)

        output = my_cnn(img_tensor)
        label = output.argmax(1).item()
        label_dict = {0: "000", 1: "001", 2: "010", 3: "011", 4: "100", 5: "101", 6: "110", 7: "111"}
        if label == img_lab:
            self.label_rel.setText("识别成功，原图方位标签为{}".format(label_dict[label]))
        else:
            self.label_rel.setText("识别失败")
        one_csv_data = adjust_tool(label, img_data)
        new_size = (512, 512)
        one_csv_data = cv.resize(one_csv_data, new_size)
        global_file_path = global_file_path[:-4] + 'adjusted.png'
        mplimg.imsave(global_file_path, one_csv_data, cmap='gray')
        pix = QPixmap(global_file_path)
        # print("pix is ok")
        self.label_adj.setPixmap(pix)
        self.label_tip_2.setText("方位调整成功")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec())
