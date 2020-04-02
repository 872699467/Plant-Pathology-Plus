import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn


def show_plant(imgs, each_row=2):
    row = ((len(imgs) + each_row - 1)) // each_row
    fig = plt.figure(figsize=(8, 6), dpi=100)
    axes = fig.subplots(row, each_row)
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        ax.imshow(imgs[i])
    plt.show()


# 获取模型参数量
def get_model_params(model: nn.Module):
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count


def visualize_leaves(train_data, train_images, cond=[0, 0, 0, 0], cond_cols=['healthy'], is_cond=True):
    if not is_cond:
        cols, rows = 3, min([3, len(train_images) // 3])
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, rows * 20 / 3))
        for col in range(cols):
            for row in range(rows):
                ax[row, col].imshow(train_images[train_images.index[-row * 3 - col - 1]])
        return None
    cond_0 = "healthy == {}".format(cond[0])
    cond_1 = "scab == {}".format(cond[1])
    cond_2 = "rust == {}".format(cond[2])
    cond_3 = "multiple_diseases == {}".format(cond[3])

    cond_list = []
    for col in cond_cols:
        if col == "healthy":
            cond_list.append(cond_0)
        if col == "scab":
            cond_list.append(cond_1)
        if col == "rust":
            cond_list.append(cond_2)
        if col == "multiple_diseases":
            cond_list.append(cond_3)

    data = train_data.iloc[:100, :]
    for cond in cond_list:
        data = data.query(cond)
    images = train_images[data.index]
    cols, rows = 3, min([3, len(images) // 3])

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, rows * 20 / 3))
    for col in range(cols):
        for row in range(rows):
            ax[row, col].imshow(images[images.index[row * 3 + col]])
    plt.show()


# Canny边缘检测并进行bounding box切割
def edge_and_cut(img):
    emb_img = img.copy()
    img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
    edges = cv2.Canny(img, 100, 200)

    nonzero = edges.nonzero()
    # 未检测到边缘
    if len(nonzero[0]) == 0 or len(nonzero[1]) == 0:
        col_min = 0
        row_min = 0
        col_max = img.shape[1]
        row_max = img.shape[0]
    else:
        col_min = min(nonzero[1])
        row_min = min(nonzero[0])
        col_max = max(nonzero[1])
        row_max = max(nonzero[0])

    emb_img[row_min - 10:row_min + 10, col_min:col_max] = [255, 0, 0]
    emb_img[row_max - 10:row_max + 10, col_min:col_max] = [255, 0, 0]
    emb_img[row_min:row_max, col_min - 10:col_min + 10] = [255, 0, 0]
    emb_img[row_min:row_max, col_max - 10:col_max + 10] = [255, 0, 0]

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
    # ax[0].imshow(img, cmap='gray')
    # ax[0].set_title('Original Image', fontsize=24)
    # ax[1].imshow(edges, cmap='gray')
    # ax[1].set_title('Canny Edges', fontsize=24)
    # ax[2].imshow(emb_img, cmap='gray')
    # ax[2].set_title('Bounding Box', fontsize=24)
    # plt.show()
    return (col_min, row_min, col_max, row_max)


# 垂直/水平翻转
def invert(img):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(cv2.flip(img, 0))
    ax[1].set_title('Vertical Flip', fontsize=24)
    ax[2].imshow(cv2.flip(img, 1))
    ax[2].set_title('Horizontal Flip', fontsize=24)
    plt.show()


# 卷积操作
def conv(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((7, 7), np.float32) / 64
    conv = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title('Convolved Image', fontsize=24)
    plt.show()


# 模糊操作
def blur(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(cv2.blur(img, (30, 30)))
    ax[1].set_title('Blurred Image', fontsize=24)
    plt.show()
