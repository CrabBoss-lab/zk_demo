# -*- codeing = utf-8 -*-
# @Time :2023/5/7 18:33
# @Author :yujunyu
# @Site :
# @File :extract_ROI.py
# @software: PyCharm
import os
import xml.etree.ElementTree as ET
import cv2

k = 1

# xml文件夹路径
xml_path = 'preData/anno_nomask'
for filename in sorted(os.listdir(xml_path)):
    # print(filename)

    # 加载xml文件
    annotation_file = os.path.join(xml_path, filename)
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # 原始图片
    img_path = root.find('path').text
    img = cv2.imread(img_path)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 解析标注信息
    for i in root.findall('object'):
        # 处理某些xml没有标注的情况
        try:
            # 解析标签名、标注坐标
            label = i.find('name').text
            xmin = int(i.find('bndbox').find('xmin').text)
            ymin = int(i.find('bndbox').find('ymin').text)
            xmax = int(i.find('bndbox').find('xmax').text)
            ymax = int(i.find('bndbox').find('ymax').text)
            print(label)
            print(xmin, ymin, xmax, ymax)

            # 裁剪标注区域
            img1 = img[ymin:ymax, xmin:xmax]

            # 保存
            save_dir = 'dataset/nomask'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir, f'{label}_' + str(k) + '.jpg'), img1)

            k += 1

        except:
            pass
