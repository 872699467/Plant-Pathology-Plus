import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from PIL import Image


def get_train_trans(SIZE):
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_test_trans(SIZE):
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TrainDataSet(Dataset):

    def __init__(self, base_path, img_list, target_list, bbox_list, transformation=None):
        super(TrainDataSet, self).__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.target_list = target_list
        self.bbox_list = bbox_list
        self.trans = transformation

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.base_path, self.img_list[index] + '.jpg')
        img: Image.Image = Image.open(img_path).convert('RGB')
        bbox = self.bbox_list[self.bbox_list[:, 0] == self.img_list[index]][0, 1:]
        crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # fig = plt.figure(figsize=(6, 4), dpi=100)
        # ax: plt.Axes = fig.subplots(1, 1)
        # ax.imshow(crop_img)
        # plt.show()
        crop_img = self.trans(crop_img)
        target = torch.tensor(self.target_list[index])
        return crop_img, target

    def collate_fn(self, item):
        imgs, targets = list(zip(*item))
        imgs = torch.stack(imgs, dim=0)
        targets = torch.stack(targets, dim=0)
        return imgs, targets


class TestDataset(Dataset):

    def __init__(self, base_path, img_list, bbox_list, transformation=None):
        super(TestDataset, self).__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.bbox_list = bbox_list
        self.trans = transformation

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.base_path, self.img_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        bbox = self.bbox_list[self.bbox_list[:, 0] == self.img_list[index]][0, 1:]
        crop_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        crop_img = self.trans(crop_img)
        return crop_img

    def collate_fn(self, item):
        imgs = item
        imgs = torch.stack(imgs, dim=0)
        return imgs
