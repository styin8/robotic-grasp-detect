import cv2
import os
import torch
import math
import glob
from options.visual_options import VisualOptions
from models import create_model

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    img:    rgb图像
    grasps: list()	元素是 [row, col, angle, width]
    mode:   line or region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            width = width / 2

            angle2 = calcAngle2(angle)
            k = math.tan(angle)

            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)


        else:
            img[row, col] = [color_b, color_g, color_r]
        #

    return img

def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """

    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


if __name__ == '__main__':
    opt = VisualOptions().parse()
    path = opt.data

    model = create_model(opt)
    model.setup()
    model.net.eval()

    depth_files = glob.glob(os.path.join(path, '*.tiff'))
    depth_files.sort()
    rgb_files = [f.replace('tiff', 'rgb') for f in depth_files]

    
    