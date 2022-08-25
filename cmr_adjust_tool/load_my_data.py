from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.all_img_name = os.listdir(self.img_dir)

    def __getitem__(self, index):
        img_name = self.all_img_name[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = np.loadtxt(img_path).astype(np.float32)
        img_lab = int(img_name[-7:-6])*4+int(img_name[-6:-5])*2+int(img_name[-5:-4])
        trans_to_tensor = transforms.ToTensor()
        img_tensor = trans_to_tensor(img)
        return img_tensor, img_lab

    def __len__(self):
        return len(self.all_img_name)



# img_dir = "D:\\workfile\\pytorch_project\\pro1\\labeled_cmr_cvs\\train"
# Data = MyData(img_dir)
# for img_tensor, img_lab in Data:
#     print(img_tensor.shape)
#     print(img_lab)

