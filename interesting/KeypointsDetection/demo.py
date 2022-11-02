## 导入相关模块
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 定义使用COCO数据集对应的每类的名称
"""
    fire hydrant 消防栓，stop sign 停车标志， parking meter 停车收费器， bench 长椅。
    zebra 斑马， giraffe 长颈鹿， handbag 手提包， suitcase 手提箱， frisbee （游戏用）飞盘（flying disc）。
    skis 滑雪板（ski的复数），snowboard 滑雪板（ski是单板滑雪，snowboarding 是双板滑雪。）
    kite 风筝， baseball bat 棒球棍， baseball glove 棒球手套， skateboard 滑板， surfboard 冲浪板， tennis racket 网球拍。
    broccoli 西蓝花，donut甜甜圈，炸面圈(doughnut，空心的油炸面包), cake 蛋糕、饼, couch 长沙发（靠chi)。
    potted plant 盆栽植物。 dining table 餐桌。 laptop 笔记本电脑，remote 遥控器(=remote control), 
    cell phone 移动电话(=mobile phone)(cellular 细胞的、蜂窝状的)， oven 烤炉、烤箱。 toaster 烤面包器（toast 烤面包片）
    sink 洗碗池, refrigerator 冰箱。（=fridge）， scissor剪刀(see, zer), teddy bear 泰迪熊。 hair drier 吹风机。 
    toothbrush 牙刷。
"""
COCO_INSTANCE_CATEGORY_NAMES = [
    '__BACKGROUND__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'trunk', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 定义能够检测出的关键点名称
"""
    elbow 胳膊肘，wrist 手腕，hip 臀部
"""
COCO_PERSON_KEYPOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear',
                              'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
                              'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']



def Object_Detect(model, image_path, COCO_INSTANCE_CATEGORY_NAMES, threshold=0.5):
    # 准备需要检测的图像
    image = Image.open(image_path)
    transform_d = transforms.Compose([transforms.ToTensor()])
    image_t = transform_d(image)    ## 对图像进行变换
    print(image_t.shape)
    pred = model([image_t])         ## 将模型作用到图像上
    print(pred)

    # 检测出目标的类别和得分
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())

    # 检测出目标的边界框
    pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().numpy())]

    ## 只保留识别的概率大约 threshold 的结果。
    pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]

    ## 设置图像显示的字体
    fontsize = np.int16(image.size[1] / 20)
    font1 = ImageFont.truetype("FreeMono.ttf", fontsize)

    ## 可视化对象
    draw = ImageDraw.Draw(image)
    for index in pred_index:
        box = pred_boxes[index]
        draw.rectangle(box, outline="blue")
        texts = pred_class[index]+":"+str(np.round(pred_score[index], 2))
        draw.text((box[0], box[1]), texts, fill="blue", font=font1) 


    pred_keypoint = pred[0]["keypoints"]
    # 检测到实例的关键点
    pred_keypoint = pred_keypoint[pred_index].detach().numpy()
    # 可视化出关键点的位置
    fontsize = np.int16(image.size[1] / 50)
    r = np.int16(image.size[1] / 150)   # 圆的半径
    font1 = ImageFont.truetype("FreeMono.ttf", fontsize)
    # 可视化图像
    image3 = image.copy()
    draw = ImageDraw.Draw(image3)
    # 对实例数量索引
    for index in range(pred_keypoint.shape[0]):
        # 对每个实例的关键点索引
        keypoints = pred_keypoint[index]
        for ii in range(keypoints.shape[0]):
            x = keypoints[ii, 0]
            y = keypoints[ii, 1]
            visi =keypoints[ii, 2]
            if visi > 0:
                draw.ellipse(xy=(x-r, y-r, x+r, y+r), fill=(255, 0, 0))
                texts = str(ii+1)
                draw.text((x+r, y-r), texts, fill="red", font=font1)

    return image3


if __name__ == '__main__':
    # 加载pytorch提供的keypointrcnn_resnet50_fpn()网络模型，可以对17个人体关键点进行检测。
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    image_path = "D:/TestData/cocotest.jpg"
    image = Object_Detect(model, image_path, COCO_INSTANCE_CATEGORY_NAMES)
    plt.imshow(image)
    plt.axis("off")
    # 保存图片，没有白边。
    plt.savefig('./skiing woman.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
