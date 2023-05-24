import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

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

# 1、加载模型及保存的模型参数
model = CNN(num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # 预测

# # 2、加载待预测图片
# # img_path='mask.png'
# img = Image.open(img_path)

transfrom = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义标签
label_map = ['mask', 'nomask']  # 根据print(dataset.class_to_idx)确定：{'mask': 0, 'nomask': 1}

# 导入人脸检测级联分类器
detect_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载摄像头
cap = cv2.VideoCapture(0)
while True:
    rer, frame = cap.read()
    # 如果可以使用opencv检测到人脸并画出人脸框，就将人脸框部分进行预测
    try:
        # 检测人脸
        detect = detect_face.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=15)  # [[150 174 200 199]]
        # 画人框
        for (x0, y0, w, h) in detect:
            # 对人脸进行画框
            frame = cv2.rectangle(frame, pt1=(x0, y0), pt2=(x0 + w, y0 + h), color=(255, 0, 0), thickness=2)

        # 根据返回的人脸坐标点，裁剪人脸部分
        face_img = frame[detect[0][1]:detect[0][1] + detect[0][3], detect[0][0]:detect[0][0] + detect[0][2]]
        # cv2.imwrite('face.png', face_img)

        # 将人脸部分进行后续的预测
        img = Image.fromarray(np.uint8(frame))

        # 3、图片预处理
        img = transfrom(img)

        # 4、模型预测
        with torch.no_grad():
            outputs = model(img)
            # 5、后处理
            pro = torch.softmax(outputs, dim=1)
            p, cls = torch.max(pro, 1)
            # print(f'待预测图片:{img_path}\t预测标签:{label_map[cls]}\t预测概率:{p.numpy()[0]}')

            # 6、可视化预测结果
            cv2.putText(frame, 'p:{:.2f} cls:{}'.format(p.numpy()[0], label_map[cls]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)


            cv2.imshow('res', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # 如果opencv检测不到人脸，就无法画人脸框，也就更无法将人脸框部分进行预测，只能将摄像头全部画面进行预测
    except:
        img = Image.fromarray(np.uint8(frame))

        # 3、图片预处理
        img = transfrom(img)

        # 4、模型预测
        with torch.no_grad():
            outputs = model(img)
            # 5、后处理
            pro = torch.softmax(outputs, dim=1)
            p, cls = torch.max(pro, 1)
            # print(f'待预测图片:{img_path}\t预测标签:{label_map[cls]}\t预测概率:{p.numpy()[0]}')

            # 6、可视化预测结果
            cv2.putText(frame, 'p:{:.2f} cls:{}'.format(p.numpy()[0], label_map[cls]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)


            cv2.imshow('res', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print('no face')
cap.release()
cv2.destroyAllWindows()

## 由于摄像头读取的画面太丰富了，因为我们训练的时候是只训练了人脸部分图片，所以这里先借助下opencv的级联检测器先检测人脸，再将人脸部分进行预测
## 由于使用级联检测检测器会出现检测不到人脸的情况，就无法对人头进行裁剪再进行预测，解决办法就是直接使用目标检测，将图像分类转换为目标检测

## 可能由于数据集太小，模型训得不是很好，出现现在的问题：不准确，有误差！！！
## 可以试着数据集多一些，多做一些数据增强，总之多去练练丹，多试试。