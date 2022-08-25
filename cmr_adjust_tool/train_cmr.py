import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from MyCnn import *
from load_my_data import MyData


# 定义训练的设备
device = torch.device("cuda")


# 准备数据集
train_dir = "D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs\\train"
test_dir = "D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs\\test"
train_data = MyData(train_dir)
test_data = MyData(test_dir)

# 获取数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size=10，则输出：训练集的长度为：10
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 创建网络模型
my_cnn = MyCnn()
my_cnn = my_cnn.to(device)

# 损失函数
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(my_cnn.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练轮数
epoch = 10
# 记录训练次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0


for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    # 训练开始
    total_train_accuracy = 0
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # print(imgs.shape)
        outputs = my_cnn(imgs)
        targets = torch.as_tensor(targets)
        loss = loss_fun(outputs, targets)
        train_accuracy = (outputs.argmax(1) == targets).sum()
        total_train_accuracy = total_train_accuracy + train_accuracy

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        # if total_train_step % 100 == 0:
        #     print("训练次数：{}，loss：{}".format(total_train_step, loss))
    print("整体训练集上的正确率：{}".format(total_train_accuracy/train_data_size))

    # 测试开始
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = my_cnn(imgs)
            targets = torch.as_tensor(targets)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy
    # print("测试集loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_test_accuracy/test_data_size))

# 保存训练完的神经网络参数数据
torch.save(my_cnn.state_dict(), "my_cnn.pth")




