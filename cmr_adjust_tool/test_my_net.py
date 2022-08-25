import torch
from PIL import Image
from MyCnn import *
import torchvision
import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms


def adjust_tool(label, one_csv_data):
    if label == 0:
        return one_csv_data
    if label == 1:
        one_csv_data = one_csv_data[:, ::-1]
        return one_csv_data
    if label == 2:
        one_csv_data = one_csv_data[::-1, :]
        return one_csv_data
    if label == 3:
        one_csv_data = one_csv_data[::-1, ::-1]
        return one_csv_data
    if label == 4:
        one_csv_data = one_csv_data.transpose()
        return one_csv_data
    if label == 5:
        one_csv_data = one_csv_data[:, ::-1].transpose()
        return one_csv_data
    if label == 6:
        one_csv_data = one_csv_data.transpose()[:, ::-1]
        return one_csv_data
    if label == 7:
        one_csv_data = one_csv_data.transpose()[::-1, ::-1]
        return one_csv_data


# # 准备输入数据
# img_path = "D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs\\test\\patient41_C0_slice7_111.csv"
# img_data = np.loadtxt(img_path).astype(np.float32)
# trans_totensor = transforms.ToTensor()
# img_tensor = trans_totensor(img_data)
# # print(img_tensor.shape)
# img_tensor = torch.reshape(img_tensor, (1, 1, 128, 128))
#
# # 创建网络模型
# my_cnn = MyCnn()
# my_cnn.load_state_dict(torch.load(("my_cnn.pth")))
# # print(my_cnn)
#
# output = my_cnn(img_tensor)
# label = output.argmax(1).item()
#
# one_csv_data = adjust_tool(label, img_data)
# img = Image.fromarray(one_csv_data)
# plt.title("   ")
# plt.imshow(img, cmap='gray')
# plt.ion()
# plt.pause(6.0)
# plt.close()




