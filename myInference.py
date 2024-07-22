# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import cv2
import time
import os
import shutil
import argparse
from mrcnn.config import Config
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib  # mrcnn模型本體
from mrcnn import visualize  # 實體化模組
# ----------------------------------------------------------tunyu存visualize.display_instances結果存成影像檔
import numpy as np
from skimage.measure import find_contours
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import os


def saveimg(image, boxes, masks, class_ids, class_names,
            scores=None, title="", string1=None,
            figsize=(16, 16), ax=None,
            show_mask=True, show_bbox=True,
            colors=None, captions=None, filename=None):
    # 有幾個instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig1, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    if colors:
        colors = colors
    else:
        colors = visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)  # (原本H + 10, -10)
    ax.set_xlim(-10, width + 10)  # (-10,  原本W + 10)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    # plt.text這塊是自行加入的 為了印上文字
    plt.text(x=width * 0.1,  # 不滿意文字出現位置可改xy
             y=height * 0.1,
             s=string1,  # 要印的文字要事先打好
             color='white',
             ha='center',
             va='center',
             fontsize=16,
             bbox=dict(boxstyle='round4', fc='black', ec='k', lw=1, alpha=0.6))
    plt.savefig(filename)
    # img = Image.open(filename)
    # trim(img).save(filename)
    # print(f"{filename} finish")
    # plt.close()


class InferenceConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # 修改 :1背景+ 你標了幾種類別(依照權重檔)
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input Config')
    parser.add_argument("--weights", help="Path of h5")
    parser.add_argument("--input_folder", help="Path of input folder")
    parser.add_argument("--output_folder", help="Path of output folder")
    parser.add_argument("--gpu", type=bool, help="Path of output folder", default=False)
    args = parser.parse_args()

if args.gpu == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference",
                          model_dir=ROOT_DIR,
                          config=config)

model.load_weights(args.weights, by_name=True)

class_names = ['BG', "RoadLane", "ShoulderLine", 'YellowLane', "car", 'pothole']

dir_name = args.input_folder
# dir_name = '.\\images'
for img_name in os.listdir(dir_name):
    img_path = dir_name + '\\' + img_name
    theimage = np.array(Image.open(img_path))
    results3 = model.detect([theimage], verbose=1)
    r = results3[0]  # 到這邊是把圖片讀進來推論得到 字典r

    MASK3 = np.reshape(a=r['masks'],  # MASK3是調整維度變成2維方便算個物件的面積
                       newshape=(-1, r['masks'].shape[-1]))

    dic3 = {}  # 空字典接 該張圖片的 key:物件的類別名稱, value:該物件的占比
    list3 = []  # 空list接 該張圖片有哪些類別(名稱) 等一下做if else
    for i in range(r['masks'].shape[-1]):
        if class_names[r['class_ids'][i]] not in dic3:
            list3.append(class_names[r['class_ids'][i]])
            dic3[class_names[r['class_ids'][i]]] = MASK3[:, i].astype(np.float32).sum()
        else:
            dic3[class_names[r['class_ids'][i]]] += MASK3[:, i].astype(np.float32).sum()
    img_area = (r['masks'].shape[0]) * (r['masks'].shape[1])

    # 這堆 if else是整理要印在圖片左下的字
    if 'pothole' in list3:
        s1 = 'pothole:' + str(round(100 * dic3['pothole'] / img_area, 3)) + '%'
    else:
        s1 = 'pothole:0%'

    if 'car' in list3:
        s2 = 'car:' + str(round(100 * dic3['car'] / img_area, 3)) + '%'
    else:
        s2 = 'car:0%'
    '''
    if 'collapse' in list3:
        s3 = 'collapse:'+str(round(100*dic3['collapse']/img_area, 2))+'%'
    else:
        s3 = 'collapse:0%' 
        '''
    # ss = s1+"\n"+s2+"\n"+s3 # ss就是最後整理出來要印在圖片左下的字 %數
    ss = s1 + "\n" + s2  # ss就是最後整理出來要印在圖片左下的字 %數

    # 這塊就是存圖片 設定最後圖片的路徑檔名
    # img_name[:-4] 是因為不要最後4個字元 .jpg 或 .png
    save_path = args.output_folder + '\\' + img_name[:-4] + '_output.png'
    # save_path =  ROOT_DIR + '\\output_space\\' +img_name[:-4] + '_output.png'
    saveimg(image=theimage,
            boxes=r['rois'],
            masks=r['masks'],
            class_ids=r['class_ids'],
            class_names=class_names,
            scores=r['scores'],
            # title="the title", # 自己設 可加可不加
            string1=ss,
            show_mask=True,  # 要不要印出mask 通常是要
            show_bbox=True,  # bbox可印可不印
            figsize=(16, 16),  # figsize預設(16,16)設越大output越大張
            filename=save_path)
