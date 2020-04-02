import torch
from torch.utils.data import DataLoader
from utils.MyDataset import TrainDataSet, TestDataset, get_train_trans, get_test_trans
from models.MyModel import ResNet18, save_checkpoints, DenseNet121, SqueezeNet
from utils.MyUtils import *
import pandas as pd
import argparse
import tqdm


def parser():
    parser = argparse.ArgumentParser(description='plant pathology')
    # device
    parser.add_argument('--device_num', type=int, default=0)
    # data
    parser.add_argument('--img_path', type=str, default='data/images/')
    parser.add_argument('--test_bbox_df', type=str, default='data/test_bbox.csv')
    parser.add_argument('--submit_df', type=str, default='data/sample_submission.csv')
    # model
    parser.add_argument('--model_type', type=str, choices=['resnet', 'densenet', 'squeezenet'], default='densenet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--checkpoints', type=str, default='checkpoints/densenet_49.pth')
    # result
    parser.add_argument('--result_file', type=str, default='results/densenet_results2.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    print(args)
    # ==========model==========
    device = torch.device('cuda:{}'.format(args.device_num))
    if args.model_type == 'resnet':
        model: nn.Module = ResNet18(pretrained=True, num_class=args.num_class).to(device)
    elif args.model_type == 'densenet':
        model: nn.Module = DenseNet121(pretrained=False, num_class=args.num_class).to(device)
    elif args.model_type == 'squeezenet':
        model: nn.Module = SqueezeNet(pretrained=True, num_class=args.num_class).to(device)
    checkpoints = torch.load(args.checkpoints, map_location=device)
    model.load_state_dict(checkpoints['state_dict'])
    print('the params amount of model：{}'.format(get_model_params(model)))
    print('==========testing===========')
    test_data = pd.read_csv(args.submit_df)
    test_bbox = pd.read_csv(args.test_bbox_df).values
    test_dataset = TestDataset(args.img_path, test_data.iloc[:, 0].values, test_bbox,
                               transformation=get_test_trans(args.size))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    test_pred = None
    for index, (test_imgs) in enumerate(tqdm.tqdm(test_dataloader, desc='testing')):
        test_imgs = test_imgs.to(device)
        model.eval()
        with torch.no_grad():
            test_output = model(test_imgs)
        if test_pred is None:
            test_pred = test_output.data.cpu()
        else:
            test_pred = torch.cat((test_pred, test_output.data.cpu()), dim=0)
    test_data[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_pred, dim=1)
    test_data.to_csv(args.result_file, index=False)
    print('finish')
