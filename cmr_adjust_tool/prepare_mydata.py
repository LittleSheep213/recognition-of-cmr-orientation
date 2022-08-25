import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mplimg


# print(np.eye(3)[::-1])
# # 显示图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     print(type(one_csv_data))
#     plt.title(one_csv_name)
#     plt.imshow(one_csv_data, cmap='gray')
#     plt.ion()
#     plt.pause(1.0)
#     plt.close()
#
# # 通过切片方式获取target，因此准备数据集时target放在文件名的最后三位
# def get_target(file_name):
#     return file_name[-7:-4]
#
#
# # 保存文件时，删除文件名最后一些无用的字符信息
# def delete_useless_info(file_name):
#     return file_name[:file_name.find(".")]

#
# # 000图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\cmrdata"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs\\{}_slice{}_000.csv".
#                format(one_csv_name[:one_csv_name.find(".")], one_csv_name[-5:-4]), one_csv_data)
#
# # 101图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\cmrdata"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_101\\{}_{}_101.csv".
#                format(one_csv_name[:one_csv_name.find(".")], one_csv_name[one_csv_name.find("s"):-4]), one_csv_data)
#
# # 000图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_101"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data[:, ::-1].transpose()
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000\\{}000.csv".
#                format(one_csv_name[:-7]), one_csv_data)
#
#
# # 001图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data[:, ::-1]
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_001\\{}001.csv".
#                format(one_csv_name[:-7]), one_csv_data)
#
#
# # 010图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data[::-1, :]
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_010\\{}010.csv".
#                format(one_csv_name[:-7]), one_csv_data)
#
#
# # 011图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data[::-1, ::-1]
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_011\\{}011.csv".
#                format(one_csv_name[:-7]), one_csv_data)
#
# # 100图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data.transpose()
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_100\\{}100.csv".
#                format(one_csv_name[:-7]), one_csv_data)
#
#
# # 110图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data[:, ::-1].transpose()
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_110\\{}110.csv".
#                format(one_csv_name[:-7]), one_csv_data)
#
#
# # 111图像
# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_000"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_csv_data = one_csv_data[::-1, ::-1].transpose()
#     # print(type(one_csv_data))
#     # print(one_csv_data.shape)
#     # plt.title(one_csv_name)
#     # plt.imshow(one_csv_data, cmap='gray')
#     # plt.ion()
#     # plt.pause(1.0)
#     # plt.close()
#     np.savetxt("D:\\workfile\\pytorch_project\\pro1\\new_labeled_cmr_cvs_111\\{}111.csv".
#                format(one_csv_name[:-7]), one_csv_data)


# 展示8种方位图像
csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\example"
all_csv_name = os.listdir(csv_dir_path)
for one_csv_name in all_csv_name:
    one_csv_path = os.path.join(csv_dir_path, one_csv_name)
    one_csv_data = np.loadtxt(one_csv_path)
    plt.subplot(3, 3, all_csv_name.index(one_csv_name)+1)
    plt.title(one_csv_name[-7:-4])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(one_csv_data, cmap='gray')
    print(type(one_csv_data))
    print(one_csv_data.shape)
plt.ion()
plt.pause(-1)
plt.close()


# csv_dir_path = "D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs\\train"
# all_csv_name = os.listdir(csv_dir_path)
# for one_csv_name in all_csv_name:
#     one_csv_path = os.path.join(csv_dir_path, one_csv_name)
#     one_csv_data = np.loadtxt(one_csv_path)
#     one_png_path = os.path.join("D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_png\\train", one_csv_name[:-3] + 'png')
#     mplimg.imsave(one_png_path, one_csv_data, cmap='gray')
