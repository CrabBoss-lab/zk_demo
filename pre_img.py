# -*- codeing = utf-8 -*-
# @Time :2023/5/7 21:35
# @Author :yujunyu
# @Site :
# @File :pre_img.py
# @software: PyCharm
import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 28 * 28, 512)
        self.relu4 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# 定义数据增强和预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])

# 加载模型参数
model = CNN(num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 加载图像
img_path = 'nomask.png'
img = Image.open(img_path)

# 进行预处理
img = transform(img)

# label_map
label_map = ['mask', 'nomask']

# 使用模型进行预测
with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    p, cls = torch.max(probs, 1)
    print('待预测图片:{}\t预测概率:{}\t预测标签:{}'.format(img_path, p.numpy()[0], label_map[cls]))

    # 可视化
    img = cv2.imread(img_path)
    cv2.putText(img, 'p:{:.2f} cls:{}'.format(p.numpy()[0], label_map[cls]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('res', img)
    cv2.waitKey(0)
