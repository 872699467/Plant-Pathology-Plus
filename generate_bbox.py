from utils.MyUtils import edge_and_cut
import pandas as pd
from PIL import Image
import numpy as np
import csv
import os
import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches

if __name__ == '__main__':
    TRAIN_DF = 'data/test.csv'
    BASE_PATH = 'data/images/'
    BBOX_PATH = 'data/test_bbox.csv'
    fieldnames = ['image_id', 'x1', 'y1', 'x2', 'y2']

    train_df = pd.read_csv(TRAIN_DF)
    img_list = train_df.iloc[:, 0].values

    bboxs = []
    for i in tqdm.tqdm(range(len(img_list))):
        img_path = os.path.join(BASE_PATH, img_list[i] + '.jpg')
        img = np.array(Image.open(img_path).convert('RGB'))
        (x1, y1, x2, y2) = edge_and_cut(img)
        # fig = plt.figure(figsize=(6, 4), dpi=100)
        # ax: plt.Axes = fig.subplots(1, 1)
        # patch = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='r', facecolor='none', linewidth=2)
        # ax.add_patch(patch)
        # ax.imshow(img)
        # plt.show()
        bboxs.append({'image_id': img_list[i], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    with open(BBOX_PATH, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bboxs)
    print('finish')
